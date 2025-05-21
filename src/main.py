# @Time     : 2025/5/20 13:46
# @Author   : Kun Lin
# @Filename : main.py
# Corrected version for fine-tuning GPT-2 on math word problems

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
import os
import datetime
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# --- Configuration ---
custom_cache_dir = "../models"
dataset_cache_dir = "../datasets/orca-math-word-problems-200k"
output_dir = "../res"
logging_dir = '../logs'
model_name = "gpt2" # Or "gpt2-medium", "gpt2-large", etc.

# Create cache directories if they don't exist
if not os.path.exists(custom_cache_dir):
    os.makedirs(custom_cache_dir)
if not os.path.exists(dataset_cache_dir):
    os.makedirs(dataset_cache_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)

# --- Load Model and Tokenizer ---
print(f"Loading model and tokenizer for {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)

# Set pad token for GPT-2 if it doesn't have one (common for GPT-like models)
# This is important for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# --- Load and Process Data ---
print(f"Loading dataset...")
# Ensure the dataset column names match what's expected.
# The 'orca-math-word-problems-200k' dataset has 'question' and 'answer' columns.
dataset = load_dataset("microsoft/orca-math-word-problems-200k", cache_dir=dataset_cache_dir)

# Define the maximum sequence length
max_length = 512 # Keep consistent with your original code

# Function to format the problem and solution and tokenize
def format_and_tokenize_function(examples):
    # Combine question and answer into a single text string with clear separators
    # This helps the model learn the structure of question -> answer
    # Use tokenizer's padding and truncation
    # Corrected column names from 'problem' and 'solution' to 'question' and 'answer'
    texts = [f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}" for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length" # Pad to max_length for consistent batching
    )

# Apply the formatting and tokenization
print("Formatting and tokenizing dataset...")
# Remove original columns as they are combined into input_ids
# Corrected column names to remove
tokenized_dataset = dataset.map(
    format_and_tokenize_function,
    batched=True,
    remove_columns=["question", "answer"] # Remove original columns after processing
)

# The DataCollatorForLanguageModeling will handle shifting labels for language modeling
# We don't need a separate group_texts function when padding to max_length
# and using the data collator. The collator prepares the input_ids and labels.

# --- Training Setup ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,          # Adjust based on dataset size and convergence
    per_device_train_batch_size=4, # Adjust based on GPU memory
    save_steps=10_000,           # Save checkpoint every 10,000 steps
    save_total_limit=2,          # Keep only the last 2 checkpoints
    # prediction_loss_only=True is the default for Trainer with CausalLM
    logging_dir=logging_dir,
    logging_steps=500,           # Log training metrics every 500 steps
    # evaluation_strategy="epoch", # Deprecated
    eval_strategy="no",          # Set evaluation strategy to 'no'
    # Add a small learning rate
    learning_rate=5e-5,
)

# Data collator for causal language modeling
# This collator handles padding and creates the labels (input_ids shifted by one)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # Use the tokenized training dataset
    # eval_dataset=tokenized_dataset["validation"], # Uncomment if you have a validation set and want evaluation
    processing_class=tokenizer,
    data_collator=data_collator, # Use the data collator
)

# --- Train and Save ---
print("Starting training...")
trainer.train()

# Save the fine-tuned model
final_save_path = os.path.join(custom_cache_dir, "fine_tuned_gpt2_math")
print(f"Saving fine-tuned model to {final_save_path}...")
trainer.save_model(final_save_path)

print("Training complete and model saved.")
