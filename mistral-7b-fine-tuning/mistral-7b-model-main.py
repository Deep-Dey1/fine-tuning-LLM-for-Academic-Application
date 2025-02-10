import os
import torch
import wandb
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig, default_data_collator
from peft import get_peft_model, LoraConfig

# ✅ Initialize Weights & Biases
wandb.init(project="mistral_7b_finetuning", name="fine-tune-run")

# ✅ Ensure CUDA Memory Optimization
torch.backends.cuda.matmul.allow_tf32 = True

# ✅ Load Dataset
dataset_name = "squad_v2"
dataset = load_dataset(dataset_name)

# ✅ Load Model & Tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

# ✅ Preprocessing Function
def preprocess_function(examples):
    inputs = [q.strip() + " " + c.strip() for q, c in zip(examples["question"], examples["context"])]
    targets = [a["text"][0].strip() if len(a["text"]) > 0 else "unanswerable" for a in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")
    labels["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ✅ Apply Tokenization
print("📌 Preprocessing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_dataset.set_format("torch")
print("✅ Dataset Preprocessed Successfully!")

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

# ✅ Define Training Arguments (Now Logs More Frequently)
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=1000,
    evaluation_strategy="steps",  # ✅ Log validation loss frequently
    eval_steps=500,  # ✅ Run evaluation every 500 steps
    logging_dir="./logs",
    logging_steps=50,  # ✅ Log training loss & gradient norm every 50 steps
    report_to=["wandb"],
    run_name="mistral_7b_finetuning",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    resume_from_checkpoint=True if os.path.exists("./mistral_finetuned") else None
)

# ✅ Custom Trainer Class (Logs Gradient Norm)
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=default_data_collator
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

# ✅ Function to Plot Loss & Gradient Norm
def plot_metrics():
    logs = trainer.state.log_history
    steps = [entry["step"] for entry in logs if "loss" in entry]
    train_losses = [entry["loss"] for entry in logs if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
    grad_norms = [entry["grad_norm"] for entry in logs if "grad_norm" in entry]

    plt.figure(figsize=(10, 5))
    
    # ✅ Plot Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label='Training Loss', color='blue')
    plt.plot(steps, eval_losses, label='Validation Loss', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    # ✅ Plot Gradient Norm Stability
    plt.subplot(1, 2, 2)
    plt.plot(steps, grad_norms, label='Gradient Norm', color='green')
    plt.xlabel('Steps')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Stability')
    plt.legend()

    plt.show()
