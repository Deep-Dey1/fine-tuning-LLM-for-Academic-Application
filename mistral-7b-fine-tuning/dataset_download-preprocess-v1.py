from datasets import load_dataset
from transformers import AutoTokenizer

# Load SQuAD v2 dataset
dataset = load_dataset("squad_v2")

# Load tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure tokenizer has padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
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


