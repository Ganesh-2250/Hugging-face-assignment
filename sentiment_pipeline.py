# sentiment_pipeline.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Step i: Load the IMDb dataset
print("Step i: Loading IMDb dataset...")
try:
    dataset = load_dataset("imdb")
    print("IMDb dataset loaded successfully.")
    print(f"Dataset structure: {dataset}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step ii: Preprocess the dataset, including tokenization
print("\nStep ii: Preprocessing dataset and tokenization...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    # Truncate and pad to ensure consistent input length for BERT
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Apply tokenization to the entire dataset
# Use batched=True for faster processing
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename the 'label' column to 'labels' as required by Trainer
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set the format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Create train and test splits
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]
print("Dataset tokenization and formatting complete.")

# Step iii: Fine-tune the BERT model for sentiment analysis
print("\nStep iii: Fine-tuning BERT model for sentiment analysis...")

# Load pre-trained BERT model for sequence classification
# num_labels=2 for binary classification (positive/negative)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",               # Directory to save checkpoints and logs
    num_train_epochs=3,                   # Total number of training epochs
    per_device_train_batch_size=8,        # Batch size per GPU/CPU for training
    per_device_eval_batch_size=8,         # Batch size per GPU/CPU for evaluation
    warmup_steps=500,                     # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                    # Strength of weight decay
    logging_dir="./logs",                 # Directory for storing logs
    logging_steps=100,                    # Log every N update steps
    evaluation_strategy="epoch",          # Evaluate every epoch
    save_strategy="epoch",                # Save model every epoch
    load_best_model_at_end=True,          # Load the best model at the end of training
    metric_for_best_model="f1",           # Metric to use to compare models
    greater_is_better=True,               # Whether the metric is better when greater
    report_to="none",                     # Disable reporting to W&B etc.
)

# Step iv: Evaluate the model's performance using accuracy and F1-score metrics
print("\nStep iv: Setting up evaluation metrics...")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary") # For binary classification
    return {"accuracy": accuracy, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting model training...")
trainer.train()
print("Model training complete.")

# Evaluate the model on the test set
print("\nEvaluating model performance on the test set...")
results = trainer.evaluate()
print(f"Evaluation Results: {results}")

# Step v: Save the fine-tuned model and demonstrate how to load it for inference
print("\nStep v: Saving the fine-tuned model...")
model_save_path = "./my_sentiment_model"
trainer.save_model(model_save_path)
print(f"Fine-tuned model saved to {model_save_path}")

print("\nDemonstrating how to load the model for inference...")
# Load the saved tokenizer and model
loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)

# Example inference
sample_text = "This movie was absolutely fantastic! I loved every minute of it."
# sample_text = "This movie was terrible. A complete waste of time."

print(f"\nSample Text for Inference: '{sample_text}'")

# Prepare input for the model
inputs = loaded_tokenizer(sample_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Move inputs to the same device as the model (e.g., GPU if available)
if torch.cuda.is_available():
    loaded_model.to("cuda")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

# Perform inference
with torch.no_grad():
    outputs = loaded_model(**inputs)

# Get predicted probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_class_id = torch.argmax(probabilities).item()

# Map predicted ID back to sentiment
sentiment_map = {0: "Negative", 1: "Positive"}
predicted_sentiment = sentiment_map[predicted_class_id]
confidence = probabilities[0][predicted_class_id].item()

print(f"Predicted Sentiment: {predicted_sentiment} (Confidence: {confidence:.4f})")

# Basic error handling example (though Trainer handles many internally)
if not dataset:
    print("Dataset could not be loaded. Please check your internet connection or dataset name.")
if not tokenizer:
    print("Tokenizer could not be loaded. Ensure 'bert-base-uncased' is correctly spelled.")
if not model:
    print("Model could not be loaded. Ensure 'bert-base-uncased' is correctly spelled.")

print("\nPipeline execution complete.")
