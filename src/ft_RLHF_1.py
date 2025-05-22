# @Time     : 2025/5/21 21:07
# @Author   : Kun Lin
# @Filename : ft_RLHF_1.py


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


print("Loading model and tokenizer...")
model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../models")

print("Setting pad token...")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # GPT-2 doesn't have a pad token by default

# Explicitly set a chat template for the GPT-2 tokenizer
# This is important because GPT-2 doesn't have a default chat template,
# and DPOTrainer will use it to format chat-like data from ultrafeedback.
# This template uses simple "Role: Content" formatting and adds eos_token after assistant messages.
# The `eos_token` is available in the Jinja context when `apply_chat_template` is called.
if tokenizer.chat_template is None:
    print("Setting a custom chat template for GPT-2 tokenizer...")
    gpt2_chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ 'User: ' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ 'Assistant: ' + message['content'] + eos_token + '\\n' }}"
            "{% else %}"  # Fallback for other roles like 'system' or if roles are different
                "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
            "{% endif %}"
        "{% endfor %}"
    )
    tokenizer.chat_template = gpt2_chat_template

print(f"Tokenizer chat template set to: {tokenizer.chat_template}")

print("Loading dataset...")
# Using a smaller subset for quicker testing, you can use the full split "train"
# For actual training, use split="train"
# dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train_sft[:1%]", cache_dir="../datasets/ultrafeedback_binarized")
# For DPO, the dataset usually has 'prompt', 'chosen', 'rejected'.
# ultrafeedback_binarized has 'prompt', 'chosen' (list of msgs), 'rejected' (list of msgs)
# Corrected split name based on the ValueError
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train", cache_dir="../datasets/ultrafeedback_binarized")
# You might want to select a smaller portion for initial testing:
# dataset = dataset.select(range(1000))


print("Training setup...")
# Ensure the output directory exists or can be created
training_args = DPOConfig(
    output_dir="../models/gpt2-DPO",
    beta=0.1, # Default DPO beta
    # Add other relevant DPOConfig arguments if needed, e.g., learning_rate, num_train_epochs
    # For testing, you might want to reduce epochs, batch size, etc.
    num_train_epochs=1,
    per_device_train_batch_size=1, # Adjust based on your GPU memory
    gradient_accumulation_steps=4, # Adjust based on your GPU memory
    # report_to="none", # Uncomment if you don't want to log to wandb/tensorboard
)

trainer = DPOTrainer(
    model=model,
    # ref_model=None, # DPOTrainer can create a reference model automatically if not provided
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer, # Corrected argument name from processing_class to tokenizer
    # formatting_func=None, # Not needed if dataset has 'prompt', 'chosen', 'rejected' in expected format
    # max_prompt_length=512, # Optional: set max prompt length
    # max_length=1024, # Optional: set max length for chosen/rejected
)

print("Starting training...")
trainer.train()
print("Training completed successfully!")

# To save the model
print("Saving model...")
trainer.save_model("../models/gpt2-DPO-final")
tokenizer.save_pretrained("../models/gpt2-DPO-final")