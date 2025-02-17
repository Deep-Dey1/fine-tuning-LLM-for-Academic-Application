import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig, default_data_collator
from peft import get_peft_model, LoraConfig

# Ensure CUDA Memory Optimization
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance

# Step 1: Define Model & Dataset
model_name = "mistralai/Mistral-7B-v0.1"
dataset = load_dataset("squad_v2")

# Step 2: Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Step 3: Preprocess Dataset (Fixes Label Padding Mismatch)
def preprocess_function(examples):
    inputs = [q.strip() + " " + c.strip() for q, c in zip(examples["question"], examples["context"])]
    targets = [a["text"][0].strip() if len(a["text"]) > 0 else "unanswerable" for a in examples["answers"]]

    # Tokenize inputs & ensure consistent length
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize targets (labels) & ensure they match input length
    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")

    # Replace pad_token_id in labels with -100 to ignore during loss calculation
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply Tokenization
print("📌 Preprocessing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_dataset.set_format("torch")
print("✅ Dataset Preprocessed Successfully!")

# Step 4: Configure QLoRA for Efficient Training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# Step 5: Load Model with Offloading Enabled
print("📌 Loading Mistral 7B Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically places layers on CPU/GPU
    offload_folder="offload"  # Saves CPU-offloaded layers to prevent memory overflow
)
print("✅ Model Loaded Successfully!")

# Step 6: Configure LoRA (PEFT Adapters for Fine-Tuning)
lora_config = LoraConfig(
    r=8,  # LoRA Adapter Size (Lower r uses less VRAM)
    lora_alpha=16,  # Scaling Factor
    lora_dropout=0.1,  # Dropout for Regularization
    bias="none"
)

# Attach LoRA Adapters to Model
peft_model = get_peft_model(model, lora_config)

# Step 7: Ensure LoRA Model Outputs Logits in Correct Shape
peft_model.config.use_cache = False  # Ensures correct batch processing
peft_model.config.pretraining_tp = 1  # Prevents batch size mismatch issues

# Step 8: Define Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",  # Save fine-tuned model here
    per_device_train_batch_size=1,  # Reduce batch size to prevent OOM
    gradient_accumulation_steps=8,  # Accumulate gradients to reduce VRAM usage
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=1000,
    eval_strategy="epoch",  # Updated from deprecated evaluation_strategy
    save_total_limit=2,
    fp16=True,  # Enable Mixed Precision for Speed & Efficiency
    logging_dir="./logs",
    report_to="wandb",  # Use Weights & Biases for tracking
    run_name="mistral_7b_finetuning",  # Avoids duplicate run names in W&B
)

# Step 9: Initialize Trainer with Proper Data Collator
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=default_data_collator  # Ensures batch consistency
)

# Step 10: Start Training 🚀
print("📌 Starting Model Training...")
trainer.train()
print("✅ Training Completed Successfully!")

# Step 11: Save Fine-Tuned Model
print("📌 Saving Fine-Tuned Model...")
peft_model.save_pretrained("./mistral_finetuned")
tokenizer.save_pretrained("./mistral_finetuned")
