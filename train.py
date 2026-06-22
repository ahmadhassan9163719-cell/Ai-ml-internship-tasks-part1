import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load data directly from public parquet links to bypass downloading script bugs
train_url = "https://huggingface.co/datasets/ag_news/resolve/main/data/train-00000-of-00001.parquet"
test_url = "https://huggingface.co/datasets/ag_news/resolve/main/data/test-00000-of-00001.parquet"

train_df = pd.read_parquet(train_url)
test_df = pd.read_parquet(test_url)

# Convert pandas dataframes directly into hugging face dataset formats
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Store datasets in a dataset dictionary structure
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Select a small slice of the data so it trains fast on your laptop
train_slice = dataset["train"].shuffle(seed=42).select(range(1000))
test_slice = dataset["test"].shuffle(seed=42).select(range(200))

# Load the tokenizer for the base BERT model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Create a function to tokenize the text data columns
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Apply the tokenization process to our data slices
tokenized_train = train_slice.map(tokenize_function, batched=True)
tokenized_test = test_slice.map(tokenize_function, batched=True)

# Load the pre-trained BERT model configuration
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=4
)


# Create a function to calculate accuracy and F1-score during training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


# Set up the training arguments and configurations
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
)

# Initialize the Trainer object to handle the training process
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Start fine-tuning the model
print("Starting BERT model fine-tuning...")
trainer.train()

# Use a clean alphanumeric name for the folder matching your app.py configuration
output_dir = "savedbertmodel"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model training complete and saved successfully!")