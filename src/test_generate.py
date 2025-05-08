# @Time     : 2025/5/8 22:42
# @Author   : Kun Lin
# @Filename : test_generate.py


# coding: utf-8
import torch
import argparse # Although we load args from file, keep it for potential overrides
import pickle
import os

# Assume data.py and model.py are in the same directory or correctly imported
import data
import model

# --- Configuration ---
# Make sure this matches the model_slt used in train.py
# It's better to get this from train_args, but keeping it here for clarity
model_slt= "transformer" # <-- It's safer to load this from args if possible

# Define the path to the saved model and args
# Assuming models are saved in a 'models' directory relative to the script's parent dir
# Or, if you run from src/, this path might need adjustment like "./models/..."
# Let's refine the paths to be relative to the script's directory for robustness
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, os.pardir) # Assumes src/ is directly under project root
ARGS_PATH = os.path.join(script_dir, "training_args.pkl") # Assuming args are saved in src/
# MODEL_PATH will be determined after loading args
DATA_PATH = os.path.join(project_root, "data", "ptb") # Path to your PTB data directory
model_dir=os.path.join(project_root, "models")


# --- Load Training Arguments ---
try:
    with open(ARGS_PATH, 'rb') as f:
        train_args = pickle.load(f)
    print(f"Loaded training arguments from {ARGS_PATH}")
    # Now we can get the model type from the loaded args
    # model_slt = train_args.model_slt # <-- Get model_slt from args!

    # Determine model path based on the model type
    MODEL_PATH = os.path.join(model_dir, f"trained_lm_model_{model_slt}.pth") # Assuming model is saved in src/ with this name

except FileNotFoundError:
    print(f"Error: Training arguments file not found at {ARGS_PATH}")
    print("Please run train.py first to generate this file.")
    # Provide instructions on how to save args in train.py if they haven't added it
    print("Make sure you added code like 'with open(\'training_args.pkl\', \'wb\') as f: pickle.dump(args, f)' in your train.py")
    exit()
except Exception as e:
    print(f"Error loading training arguments: {e}")
    exit()

# Set device
# Use the GPU ID from training args if CUDA is available and was used
if torch.cuda.is_available() and train_args.cuda:
    # Ensure the GPU ID from training args is valid
    if train_args.gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(train_args.gpu_id)
        device = torch.device(train_args.gpu_id)
        print(f"Using GPU device {train_args.gpu_id}")
    else:
        print(f"Warning: GPU device {train_args.gpu_id} not available ({torch.cuda.device_count()} devices found). Using CPU.")
        device = torch.device("cpu")
else:
    print("CUDA not available or not used during training. Using CPU.")
    device = torch.device("cpu")


# --- Load Data (for vocabulary) ---
# We only need the vocabulary/word mappings here
try:
    # Use a batch size of 1 as generation is done sample by sample
    test_batch_size = 1
    # The data_loader object itself holds the word_id and vocabulary list
    data_loader = data.Corpus(DATA_PATH, {'train': test_batch_size, 'valid': test_batch_size}, train_args.max_sql)

    # --- FIX: Get the correct mappings from data_loader ---
    word_to_id = data_loader.word_id # This is your word2idx dict
    id_to_word = data_loader.vocabulary # This is your idx2word list

    nvoc = len(id_to_word) # Vocabulary size is the length of the list
    print(f"Vocabulary loaded. Size: {nvoc}")
    # Also check if <unk> is in the vocabulary, it's needed for handling unknown words in prompt
    if '<unk>' not in word_to_id:
         print("Warning: '<unk>' token not found in vocabulary. Prompts with unknown words will cause errors.")
         # You might want to add '<unk>' to your data.py or handle this case.
         # Based on your data.py, <unk> is added, so this warning might be unnecessary.
         pass


except FileNotFoundError:
     print(f"Error: Data not found at {DATA_PATH}. Please check the path.")
     exit()
except Exception as e:
    print(f"Error loading data/vocabulary: {e}")
    exit()


# --- Load Model ---
# Re-create the model architecture based on loaded arguments
if model_slt == "transformer":
    model = model.LMModel_transformer(nvoc=nvoc, num_layers=train_args.num_layers,
                                      dim=train_args.emb_dim, nhead=train_args.num_heads)
elif model_slt == "rnn":
    model = model.LMModel_RNN(nvoc=nvoc, num_layers=train_args.num_layers,
                              dim=train_args.emb_dim)
elif model_slt == "lstm":
    model = model.LMModel_LSTM(nvoc=nvoc, num_layers=train_args.num_layers,
                               dim=train_args.emb_dim)
else:
    print(f"Error: Unknown model type '{model_slt}' from training args.")
    exit()

model = model.to(device)

# Load the saved state dictionary
try:
    # Load using map_location to ensure it loads correctly regardless of saved device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}.")
    print(f"Expected path: {MODEL_PATH}")
    print("Please run train.py first to save the model.")
    print(f"Make sure train.py saves the model state_dict to {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval() # Set model to evaluation mode

# --- Text Generation Function ---
# --- FIX: Updated signature to accept word_to_id and id_to_word ---
def generate_text(model, word_to_id, id_to_word, prompt_text, max_length=100, temperature=1.0):
    """
    Generates text using the trained language model.

    Args:
        model: The trained PyTorch model.
        word_to_id (dict): Dictionary mapping words to indices (from data_loader.word_id).
        id_to_word (list): List mapping indices to words (from data_loader.vocabulary).
        prompt_text (str): The initial text prompt.
        max_length (int): The maximum number of tokens to generate.
        temperature (float): Controls randomness. Higher temperature = more random.
                             Use 1.0 for standard sampling, <1.0 for less random, >1.0 for more random.
                             Set temperature very low (e.g., 1e-8) for argmax (most likely token).
    Returns:
        str: The generated text.
    """
    # Tokenize the prompt
    # Use the same simple split as often used for PTB
    tokens = prompt_text.lower().split() # PTB is typically lowercased

    # Convert tokens to indices using the word_to_id map
    # --- FIX: Use word_to_id instead of vocabulary.word2idx ---
    # Use get with a default value for <unk>
    unk_id = word_to_id.get('<unk>')
    if unk_id is None:
         print("Warning: '<unk>' not found in word_to_id map. Handling unknown words may fail.")
         # Fallback: If <unk> not explicitly in dict, perhaps assume index 0 or handle differently
         # For PTB, <unk> is usually handled, so this might not be needed if data.py is correct
         unk_id = 0 # Fallback - risky if 0 is a real word ID

    indices = [word_to_id.get(token, unk_id) for token in tokens]

    # Prepare input tensor
    # Model expects (seq_len, batch_size). Batch size is 1 for generation.
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(device) # Shape: (prompt_len, 1)

    # Start generated sequence with prompt indices/tokens
    generated_indices = indices[:]
    generated_tokens = tokens[:]

    print("\n--- Generating Text ---")
    print("Prompt:", prompt_text)

    with torch.no_grad():
        for _ in range(max_length):
            # Feed the current sequence (prompt + generated) to the model
            # The model predicts the *next* token based on the entire sequence fed
            if model_slt != "transformer":
                 # RNN/LSTM stateful generation would need hidden state management,
                 # but feeding the whole sequence is simpler for demonstration.
                 output, _ = model(input_tensor)
                 # The prediction for the next token is based on the *last* output token
                 logits = output[-1, 0, :] # Shape: (nvoc,) - last token, batch 0, all vocab logits
            else:
                 # Transformer takes the whole sequence and predicts for each position.
                 # We only care about the prediction for the *last* token in the input.
                 output = model(input_tensor) # Shape: (seq_len, batch_size, nvoc)
                 logits = output[-1, 0, :] # Shape: (nvoc,) - last token, batch 0, all vocab logits

            # Apply temperature to logits and sample
            if temperature < 1e-8: # Use argmax for deterministic output
                 next_token_probs = torch.softmax(logits, dim=-1)
                 next_token_idx = torch.argmax(next_token_probs, dim=-1).item()
            else: # Sample from distribution
                 # Ensure logits are float for softmax and sampling
                 next_token_probs = torch.softmax(logits.float() / temperature, dim=-1)
                 # Use .item() to get the scalar index
                 next_token_idx = torch.multinomial(next_token_probs, num_samples=1).item()

            # Convert index back to word using the id_to_word list
            # --- FIX: Use id_to_word list instead of vocabulary.idx2word ---
            # Direct list indexing; assuming next_token_idx is within bounds [0, nvoc-1]
            if 0 <= next_token_idx < len(id_to_word):
                next_token = id_to_word[next_token_idx]
            else:
                # This case should ideally not happen if model output corresponds to vocabulary size
                print(f"Warning: Generated index {next_token_idx} out of vocabulary bounds ({len(id_to_word)}). Using <unk>.")
                next_token = '<unk>' # Fallback token

            # Append to the sequence
            generated_indices.append(next_token_idx)
            generated_tokens.append(next_token)

            # Break if end of sequence token is generated
            if next_token == '<eos>':
                break

            # Update input tensor for the next step
            # We need to feed the model the sequence including the newly generated token
            # Reshape to (seq_len, 1)
            input_tensor = torch.tensor(generated_indices, dtype=torch.long).unsqueeze(1).to(device)

            # Optional: Print token by token as it generates
            # print(next_token, end=' ', flush=True)


    print("\n--- Generation Complete ---")
    # Join tokens to form the final text
    # The generated_tokens list now contains prompt tokens + generated tokens.
    # We can just join the whole list and potentially clean up later.
    final_text = " ".join(generated_tokens)

    # Clean up prompt tokens if they should not be part of the final joined output string
    # A better way is to just join the generated tokens *after* the prompt
    prompt_len = len(tokens)
    generated_only_tokens = generated_tokens[prompt_len:]
    final_text = " ".join(generated_only_tokens)


    # Remove the last '<eos>' if it exists
    if final_text.endswith(" <eos>"):
        final_text = final_text[:-len(" <eos>")]

    return prompt_text + " " + final_text.strip() # Add the original prompt back


# --- Main Execution ---
if __name__ == "__main__":
    print("Language Model Text Generation Tool")
    print("-" * 30)

    # --- FIX: Pass the correct mappings to generate_text ---
    # We need word_to_id (dict) and id_to_word (list)
    # These are now correctly loaded as data_loader.word_id and data_loader.vocabulary
    # Let's rename them locally for clarity:
    word_map = data_loader.word_id
    id_list = data_loader.vocabulary


    while True:
        prompt = input("Enter a starting prompt (or 'quit' to exit): ").strip()
        if prompt.lower() == 'quit':
            break

        try:
            length = int(input("Enter number of tokens to generate (e.g., 50): "))
            if length <= 0:
                print("Please enter a positive number.")
                continue
        except ValueError:
            print("Invalid number. Please enter an integer.")
            continue

        # Optional: Add temperature control
        # try:
        #     temp = float(input("Enter generation temperature (e.g., 1.0, 0.5=less random, 1.5=more random): "))
        #     if temp <= 0:
        #          print("Temperature must be positive. Using 1.0.")
        #          temp = 1.0
        # except ValueError:
        #      print("Invalid temperature. Using 1.0.")
        #      temp = 1.0
        temp = 0.8 # Default temperature for some variability

        # --- FIX: Call generate_text with the correct arguments ---
        generated_sequence = generate_text(model, word_map, id_list, prompt, max_length=length, temperature=temp)

        print("\n" + "="*30)
        print("Generated Text:")
        print(generated_sequence)
        print("="*30 + "\n")

    print("Exiting generator.")