import os
import torch
import wandb
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
from transformers import BitsAndBytesConfig, default_data_collator
from peft import get_peft_model, LoraConfig

# ✅ Initialize Weights & Biases
wandb.init(project="mistral_7b_finetuning", name="fine-tune-run")

# ✅ Ensure CUDA Memory Optimization
torch.backends.cuda.matmul.allow_tf32 = True

# ✅ Load Datasets
print("📌 Loading datasets...")
squad_dataset = load_dataset("squad_v2")
sciq_dataset = load_dataset("allenai/sciq")
pubmedqa_dataset = load_dataset("pubmed_qa", "pqa_labeled")

# ✅ Preprocessing Function
def preprocess_function(examples):
    inputs = []
    targets = []
    for i in range(len(examples["question"])):
        if "context" in examples:
            inputs.append(examples["question"][i].strip() + " " + examples["context"][i].strip())
        else:
            inputs.append(examples["question"][i].strip())
        
        if "answers" in examples:
            targets.append(examples["answers"][i]["text"][0].strip() if len(examples["answers"][i]["text"]) > 0 else "unanswerable")
        elif "correct_answer" in examples:
            targets.append(examples["correct_answer"][i].strip())
        elif "long_answer" in examples:
            targets.append(examples["long_answer"][i].strip())
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")
    labels["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ✅ Combine Datasets
print("📌 Combining datasets...")
train_datasets = [
    squad_dataset["train"].map(preprocess_function, batched=True, remove_columns=squad_dataset["train"].column_names),
    sciq_dataset["train"].map(preprocess_function, batched=True, remove_columns=sciq_dataset["train"].column_names),
    pubmedqa_dataset["train"].map(preprocess_function, batched=True, remove_columns=pubmedqa_dataset["train"].column_names),
]
train_dataset = concatenate_datasets(train_datasets)

validation_datasets = [
    squad_dataset["validation"].map(preprocess_function, batched=True, remove_columns=squad_dataset["validation"].column_names),
    sciq_dataset["validation"].map(preprocess_function, batched=True, remove_columns=sciq_dataset["validation"].column_names),
    pubmedqa_dataset["validation"].map(preprocess_function, batched=True, remove_columns=pubmedqa_dataset["validation"].column_names),
]
validation_dataset = concatenate_datasets(validation_datasets)

# ✅ Load Model & Tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Configure QLoRA for Efficient Training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# ✅ Load Mistral 7B Model
print("📌 Loading Mistral 7B Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload"
)
print("✅ Model Loaded Successfully!")

# ✅ Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

peft_model = get_peft_model(model, lora_config)
peft_model.config.use_cache = False
peft_model.config.pretraining_tp = 1

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    report_to=["wandb"],
    run_name="mistral_7b_finetuning",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    resume_from_checkpoint=True if os.path.exists("./mistral_finetuned") else None
)

# ✅ Load Metrics
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
exact_match_metric = evaluate.load("exact_match")

# ✅ Define Compute Metrics Function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=-1)  # Convert logits to predictions

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute metrics
    f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, average="macro")
    precision = precision_metric.compute(predictions=decoded_preds, references=decoded_labels, average="macro")
    recall = recall_metric.compute(predictions=decoded_preds, references=decoded_labels, average="macro")
    exact_match = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "exact_match": exact_match["exact_match"],
    }

# ✅ Custom Trainer Class
class CustomTrainer(Trainer):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            grad_norm = self._compute_gradient_norm()
            logs["grad_norm"] = grad_norm
            wandb.log(logs)  # ✅ Log all values in wandb
            print(f"📊 Step {state.global_step} | Loss: {logs.get('loss', 'N/A')} | Val Loss: {logs.get('eval_loss', 'N/A')} | Grad Norm: {grad_norm}")

    def _compute_gradient_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

# ✅ Initialize Trainer
trainer = CustomTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,  # Add this line
)

# ✅ Start Training 🚀
print("📌 Starting Model Training...")
trainer.train()
print("✅ Training Completed Successfully!")

# ✅ Save Fine-Tuned Model
print("📌 Saving Fine-Tuned Model...")
peft_model.save_pretrained("./mistral_finetuned")
tokenizer.save_pretrained("./mistral_finetuned")
print("🎉 Fine-tuning complete! Model saved in ./mistral_finetuned 🎉")
