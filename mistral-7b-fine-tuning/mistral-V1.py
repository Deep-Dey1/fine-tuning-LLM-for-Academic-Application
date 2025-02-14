import transformers
import torch
import wandb
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support

# ✅ Initialize W&B
wandb.init(project="mistral_7b_finetuning", name="multi-dataset-finetune")

# ✅ Load Datasets
print("📌 Loading datasets...")
squad_dataset = load_dataset("squad_v2")
sciq_dataset = load_dataset("allenai/sciq")
medqa_dataset = load_dataset("bigbio/med_qa")

# ✅ Ensure datasets have validation splits
def create_validation_split(dataset, test_size=0.1):
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=test_size)
        dataset = datasets.DatasetDict({
            "train": dataset["train"],
            "validation": dataset["test"]
        })
    return dataset

squad_dataset = create_validation_split(squad_dataset)
sciq_dataset = create_validation_split(sciq_dataset)
medqa_dataset = create_validation_split(medqa_dataset)

# ✅ Combine datasets into one
train_dataset = concatenate_datasets([
    squad_dataset["train"], 
    sciq_dataset["train"], 
    medqa_dataset["train"]
])

val_dataset = concatenate_datasets([
    squad_dataset["validation"], 
    sciq_dataset["validation"], 
    medqa_dataset["validation"]
])

print(f"✅ Combined Training Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
max_seq_length = 1024  # Increased sequence length

# ✅ Preprocessing function
def preprocess_function(examples):
    inputs, targets = [], []

    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        context = examples.get("context", [""] * len(examples["question"]))[i]

        # Handle MedQA-specific format
        if "answer" in examples:
            answer_text = examples["answer"][i]
        else:
            answer_text = "unanswerable"

        context = context.strip() if isinstance(context, str) else ""
        answer_text = answer_text.strip() if isinstance(answer_text, str) else ""

        input_text = f"{question} {context}".strip()
        inputs.append(input_text)
        targets.append(answer_text)

    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True, padding="longest")
    labels = tokenizer(text_target=targets, max_length=max_seq_length, truncation=True, padding="longest")

    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# ✅ Apply preprocessing
print("📌 Preprocessing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# ✅ Define evaluation metrics
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    precision, recall, f1, _ = precision_recall_fscore_support(label_texts, pred_texts, average="macro", zero_division=1)

    # Calculate Exact Match (EM)
    em = np.mean([int(pred == label) for pred, label in zip(pred_texts, label_texts)])

    wandb.log({"Precision": precision, "Recall": recall, "F1-score": f1, "EM": em})
    return {"precision": precision, "recall": recall, "f1": f1, "em": em}

# ✅ Load Mistral 7B model
print("📌 Loading Mistral 7B model...")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="cuda:0",  # Explicitly use GPU 0
    torch_dtype=torch.float16  # Use FP16 for memory efficiency
)

# Enable Gradient Checkpointing (saves memory)
model.gradient_checkpointing_enable()

# ✅ Set training arguments
training_args = TrainingArguments(
    output_dir="./mistral-multi-dataset",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=2,  # Increased batch size
    per_device_eval_batch_size=4,   # Increased batch size
    gradient_accumulation_steps=4,
    learning_rate=2e-5,  # Adjusted learning rate
    weight_decay=0.01,
    warmup_steps=500,
    max_grad_norm=1.0,  # Gradient clipping
    logging_dir="./logs",
    logging_steps=100,
    report_to="wandb",
    load_best_model_at_end=True,
    fp16=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_first_step=True,
)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ✅ Train model
print("🚀 Starting training...")
trainer.train()

# ✅ Save model
print("📌 Saving model...")
trainer.save_model("./mistral-multi-dataset")

# ✅ Finish W&B run
wandb.finish()

print("✅ Training complete! Model saved at './mistral-multi-dataset'")
