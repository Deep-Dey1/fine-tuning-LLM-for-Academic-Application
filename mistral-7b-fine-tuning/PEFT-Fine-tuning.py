from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# Step 1: Define the model name and load dataset
model_name = "mistralai/Mistral-7B-v0.1"  # Model name for Mistral 7B

# Load dataset (replace with your actual dataset if needed)
dataset = load_dataset("squad_v2")

# Step 2: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure tokenizer has padding token (if necessary)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Preprocess the dataset
def preprocess_function(examples):
    inputs = [q.strip() + " " + c.strip() for q, c in zip(examples["question"], examples["context"])]
    targets = [a["text"][0].strip() if len(a["text"]) > 0 else "unanswerable" for a in examples["answers"]]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch")

# Step 4: Configure QLoRA for fine-tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# Step 5: Load model with QLoRA configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Step 6: Configure PEFT (trainable adapters)
lora_config = LoraConfig(
    r=8,  # Size of adapter (adjust based on model size)
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias handling (default: "none")
)

# Attach PEFT adapters to the quantized model
peft_model = get_peft_model(model, lora_config)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",  # Directory to save the fine-tuned model
    per_device_train_batch_size=1,  # Lower batch size due to 8GB VRAM
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=1000,  # Save model checkpoint every 1000 steps
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_total_limit=2,  # Keep only the last 2 checkpoints
    fp16=True,  # Use mixed precision for faster training
    logging_dir="./logs",  # Directory for training logs
    report_to="wandb"  # Track training with Weights & Biases
)

# Step 8: Set up the Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Step 9: Start training
trainer.train()

# Step 10: Save the fine-tuned model
peft_model.save_pretrained("/content/mistral_finetuned")
tokenizer.save_pretrained("/content/mistral_finetuned")
