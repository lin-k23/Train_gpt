# @Time     : 2025/5/23 00:42
# @Author   : Kun Lin
# @Filename : ft_LmsysChat.py


from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
import os
import datetime
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# --- Configuration ---
custom_cache_dir = "../models"
dataset_cache_dir = "../datasets/lmsys_chat" # Updated dataset cache directory
output_dir = "../res"
logging_dir = '../logs'
model_name = "gpt2" # Or "gpt2-medium", "gpt2-large", etc.

# New configuration for subset selection
# Set to None to use the full dataset, or an integer for the number of samples
subset_size = None # Example: use 100,000 samples for training. Set to None for full dataset.

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

# Add special tokens for conversation roles if they don't exist
# This helps the model distinguish between different speakers in a conversation.
# You can define your own special tokens based on the dataset's roles.
# For lmsys-chat-1m, typical roles are 'user' and 'assistant'.
special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} new special tokens to tokenizer.")
# Resize model embeddings to account for new tokens
model.resize_token_embeddings(len(tokenizer))


# Set pad token for GPT-2 if it doesn't have one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# --- Load and Process Data ---
print(f"Loading dataset...")
# Load the lmsys/lmsys-chat-1m dataset
dataset = load_dataset("lmsys/lmsys-chat-1m", cache_dir=dataset_cache_dir)

# Apply subset selection if subset_size is defined
if subset_size is not None and subset_size > 0:
    print(f"Selecting a subset of {subset_size} samples from the training dataset...")
    # Ensure that subset_size does not exceed the actual size of the training split
    actual_subset_size = min(subset_size, len(dataset["train"]))
    dataset["train"] = dataset["train"].select(range(actual_subset_size))
    print(f"Selected {actual_subset_size} samples for training.")


# Define the maximum sequence length
max_length = 512

def format_and_tokenize_function(examples):
    """
    Formats and tokenizes the chat data for causal language modeling.
    Each conversation is joined into a single string, with special tokens
    to delineate roles and turns, then tokenized.
    """
    formatted_conversations = []
    for conversation in examples["conversation"]:
        full_text = ""
        for turn in conversation:
            role = turn['role']
            content = turn['content']
            # Use specific role tokens and end-of-sequence token to structure the conversation
            if role == 'user':
                full_text += f"<|user|>{content}{tokenizer.eos_token}"
            elif role == 'assistant':
                full_text += f"<|assistant|>{content}{tokenizer.eos_token}"
            else:
                # Handle other potential roles if they exist, or ignore them
                full_text += f"{content}{tokenizer.eos_token}"
        formatted_conversations.append(full_text)

    # Tokenize the formatted conversations
    tokenized_inputs = tokenizer(
        formatted_conversations,
        max_length=max_length,
        truncation=True,
        padding=False # DataCollatorForLanguageModeling will handle padding
    )
    return tokenized_inputs

print("Formatting and tokenizing dataset...")
# Apply the formatting and tokenization function to the dataset
# Remove original columns to avoid issues with different lengths after tokenization
# Corrected remove_columns list based on available columns in lmsys/lmsys-chat-1m
tokenized_dataset = dataset.map(
    format_and_tokenize_function,
    batched=True,
    remove_columns=["conversation_id", "model", "turn", "language", "openai_moderation", "redacted"]
)

# --- Training Setup ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,            # Adjust based on dataset size and convergence
    per_device_train_batch_size=1, # Adjust based on GPU memory. Start small.
    save_steps=10_000,             # Save checkpoint every 10,000 steps
    save_total_limit=2,            # Keep only the last 2 checkpoints
    logging_dir=logging_dir,
    logging_steps=500,             # Log training metrics every 500 steps
    eval_strategy="no",            # Set evaluation strategy to 'no'
    learning_rate=1e-5,
    report_to="none",            # Disable reporting to wandb/tensorboard
    # fp16=True,                     # Enable mixed precision training for memory efficiency
    # gradient_accumulation_steps=4, # Uncomment and adjust if you need to simulate a larger batch size
                                   # without increasing per_device_train_batch_size

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
    tokenizer=tokenizer,
    data_collator=data_collator, # Use the data collator
    # report_to="none", # Disable reporting to wandb/tensorboard
)

# --- Train and Save ---
print("Starting training...")
trainer.train()

# Save the fine-tuned model
final_save_path = os.path.join(custom_cache_dir, "lmsys_chat_gpt2_2") # Updated save path
print(f"Saving fine-tuned model to {final_save_path}...")
trainer.save_model(final_save_path)

print("Training complete and model saved.")
