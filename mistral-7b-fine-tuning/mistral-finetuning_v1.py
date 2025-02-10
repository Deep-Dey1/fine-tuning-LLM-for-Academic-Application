import os
import torch
import transformers
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig, default_data_collator
from peft import get_peft_model, LoraConfig
import wandb

# Initialize Weights & Biases (for logging)
wandb.init(project="mistral_7b_finetuning", name="fine-tune-run")

# Ensure CUDA Memory Optimization
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance

dataset_name = "squad_v2"  # Update with correct dataset

# Load dataset
dataset = load_dataset(dataset_name)

# Load model & tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

def preprocess_function(examples):
    inputs = [q.strip() + " " + c.strip() for q, c in zip(examples["question"], examples["context"])]
    targets = [a["text"][0].strip() if len(a["text"]) > 0 else "unanswerable" for a in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")
    labels["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("📌 Preprocessing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_dataset.set_format("torch")
print("✅ Dataset Preprocessed Successfully!")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

print("📌 Loading Mistral 7B Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically places layers on CPU/GPU
    offload_folder="offload"  # Saves CPU-offloaded layers to prevent memory overflow
)
print("✅ Model Loaded Successfully!")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

peft_model = get_peft_model(model, lora_config)
peft_model.config.use_cache = False
peft_model.config.pretraining_tp = 1

output_dir = "./mistral_finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",  # Change from "steps" to "epoch"
    evaluation_strategy="epoch",  # Matches save strategy
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs",
    report_to=["wandb"],
    run_name="mistral_7b_finetuning",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    resume_from_checkpoint=True if os.path.exists(output_dir) else None
)


trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=default_data_collator
)

print("📌 Starting Model Training...")
trainer.train()
print("✅ Training Completed Successfully!")

print("📌 Saving Fine-Tuned Model...")
peft_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("🎉 Fine-tuning complete! Model saved in ./mistral_finetuned 🎉")

def plot_loss():
    logs = trainer.state.log_history
    steps = [entry["step"] for entry in logs if "loss" in entry]
    train_losses = [entry["loss"] for entry in logs if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
    
    plt.plot(steps, train_losses, label='Training Loss', color='blue')
    plt.plot(steps, eval_losses, label='Validation Loss', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()

plot_loss()

