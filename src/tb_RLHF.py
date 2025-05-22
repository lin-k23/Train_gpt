import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_response(model, tokenizer, prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_k=50,
                      top_p=0.95):
    """
    Generates text from a given model using the specified tokenizer and generation parameters.

    Args:
        model: The Hugging Face causal language model (e.g., AutoModelForCausalLM).
        tokenizer: The Hugging Face tokenizer corresponding to the model.
        prompt (str): The input prompt for text generation.
        max_length (int): The maximum length of the generated sequence.
        num_return_sequences (int): The number of independent sequences to generate.
        temperature (float): Controls the randomness of predictions. Higher values mean more random.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): The cumulative probability for nucleus sampling (top-p-filtering).

    Returns:
        str or list: The generated text(s). Returns a single string if num_return_sequences is 1,
                     otherwise a list of strings.
    """
    # Apply the chat template to format the prompt according to the model's expected input format.
    # This returns a BatchEncoding object, which is like a dictionary of tensors.
    encoded_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True  # Add the special tokens that indicate the start of generation
    )

    # Ensure pad_token_id is set for generation. GPT-2 often uses eos_token as pad_token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if encoded_inputs is a BatchEncoding (dictionary-like) or a raw tensor
    if isinstance(encoded_inputs, dict):  # This covers BatchEncoding
        input_ids = encoded_inputs["input_ids"].to(model.device)
        attention_mask = encoded_inputs.get("attention_mask", None)  # attention_mask might not always be present
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    elif isinstance(encoded_inputs, torch.Tensor):
        # If it's a raw tensor, assume it's just input_ids
        input_ids = encoded_inputs.to(model.device)
        attention_mask = None  # No attention mask if it's just a raw tensor

        generate_kwargs = {
            "input_ids": input_ids
        }
    else:
        # If it's neither, raise an error with more information
        raise TypeError(
            f"Unexpected type for encoded_inputs: {type(encoded_inputs)}. Expected dict-like (BatchEncoding) or torch.Tensor.")

    output_sequences = model.generate(
        **generate_kwargs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,  # Enable sampling for more diverse outputs
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_texts = []
    # Use input_ids.shape[1] from the actual input_ids tensor used for generation
    for seq in output_sequences:
        # Ensure the slice is valid: seq should be at least as long as input_ids.shape[1]
        start_index = input_ids.shape[1]
        if start_index >= seq.shape[0]:
            # This means the generated sequence is shorter than or equal to the input prompt.
            # In this case, the model generated nothing new or only very short output.
            # We can just decode the whole sequence or handle it as an empty generation.
            decoded_text = ""  # Or tokenizer.decode(seq, skip_special_tokens=True) if you want to see the prompt too
        else:
            decoded_text = tokenizer.decode(seq[start_index:], skip_special_tokens=True)
        generated_texts.append(decoded_text.strip())  # .strip() removes leading/trailing whitespace

    return generated_texts[0] if num_return_sequences == 1 else generated_texts


def main():
    """
    Main function to run the RLHF model evaluation.
    Loads models, generates responses, facilitates human evaluation, and reports results.
    """
    # --- Configuration ---
    # Path to your fine-tuned DPO model saved from ft_RLHF_1.py
    FINE_TUNED_MODEL_PATH = "../models/gpt2-DPO-final"
    # Name of the original base model (e.g., "gpt2", "gpt2-medium", etc.)
    BASE_MODEL_NAME = "gpt2"
    # Directory to cache downloaded models and tokenizers
    CACHE_DIR = "../models"

    # Ensure the cache directory exists to prevent errors
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- 1. Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR)
    # If GPT-2 tokenizer doesn't have a pad token, set it to the end-of-sequence token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Re-apply the custom chat template. This is critical to ensure that the
    # evaluation script formats prompts in the same way the model was trained.
    if tokenizer.chat_template is None:
        print("Setting a custom chat template for GPT-2 tokenizer...")
        gpt2_chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: ' + message['content'] + eos_token + '\\n' }}"
            "{% else %}"  # Fallback for other roles if they appear
            "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = gpt2_chat_template
    print(f"Tokenizer chat template: {tokenizer.chat_template}")

    # --- 2. Load Models ---
    print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_PATH}...")
    try:
        # Load the fine-tuned model from the specified path.
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH)
        fine_tuned_model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
        print("Fine-tuned model loaded successfully.")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        print(f"Please ensure the model was saved correctly at '{FINE_TUNED_MODEL_PATH}' "
              "after your RLHF training script completed.")
        return  # Exit if the fine-tuned model cannot be loaded

    print(f"Loading base model '{BASE_MODEL_NAME}'...")
    # Load the original base GPT-2 model.
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR)
    base_model.eval()  # Set the model to evaluation mode
    print("Base model loaded successfully.")

    # Determine the device (GPU if available, otherwise CPU) and move models to it for faster inference.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)
    base_model.to(device)
    print(f"Models moved to {device} for inference.")

    # --- 3. Define Evaluation Prompts ---
    # A set of diverse prompts to test different capabilities and areas of interest.
    evaluation_prompts = [
        "Tell me a short story about a brave knight who encounters a friendly dragon.",
        "Explain the concept of photosynthesis in simple terms for a 10-year-old.",
        "What are the key benefits of regular exercise for mental and physical health?",
        "Write a short, encouraging message for someone feeling down about their progress.",
        "Describe a beautiful sunset over a calm ocean, focusing on colors and feelings.",
        "Give me a recipe for a simple chocolate chip cookie.",
        "What is the capital of France and why is it famous?",
        "Discuss the importance of renewable energy sources for the future.",
        "Write a short poem about autumn leaves.",
        "Summarize the plot of 'Romeo and Juliet' in a few sentences."
    ]

    # --- 4. Generate Responses and Store for Comparison ---
    print("\n--- Generating Responses for Comparison ---")
    results = []  # List to store generated responses and prompts
    for i, prompt in enumerate(evaluation_prompts):
        print(f"\n--- Processing Prompt {i + 1}/{len(evaluation_prompts)} ---")
        print(f"Prompt: {prompt}")

        # Generate response from the fine-tuned model
        print("Generating response from fine-tuned model...")
        fine_tuned_response = generate_response(fine_tuned_model, tokenizer, prompt)
        print(f"Fine-tuned Model Output: {fine_tuned_response}")

        # Generate response from the base model
        print("Generating response from base model...")
        base_response = generate_response(base_model, tokenizer, prompt)
        print(f"Base Model Output: {base_response}")

        # Store the results for later human evaluation
        results.append({
            "prompt": prompt,
            "fine_tuned_response": fine_tuned_response,
            "base_response": base_response
        })

    # --- 5. Human Evaluation Loop (Interactive) ---
    print("\n--- Starting Human Evaluation ---")
    print("For each prompt, you will see responses from both models.")
    print("Please compare them and indicate which response you prefer:")
    print("  Type 'f' for the Fine-tuned Model's response.")
    print("  Type 'b' for the Base Model's response.")
    print("  Type 'e' if both responses are equally good (or equally bad).")

    human_preferences = {"fine_tuned": 0, "base": 0, "equal": 0}

    for i, res in enumerate(results):
        print(f"\n--- Evaluation Prompt {i + 1} of {len(results)} ---")
        print(f"Prompt: {res['prompt']}")
        print(f"\n[F] Fine-tuned Model Response:")
        print(f"   {res['fine_tuned_response']}")
        print(f"\n[B] Base Model Response:")
        print(f"   {res['base_response']}")

        while True:
            preference = input("Your preference (f/b/e): ").lower().strip()
            if preference == 'f':
                human_preferences["fine_tuned"] += 1
                break
            elif preference == 'b':
                human_preferences["base"] += 1
                break
            elif preference == 'e':
                human_preferences["equal"] += 1
                break
            else:
                print("Invalid input. Please type 'f', 'b', or 'e'.")

    # --- 6. Reporting and Summary ---
    print("\n--- Evaluation Summary ---")
    print("\nAutomated Metrics (Response Length in Words):")
    for i, res in enumerate(results):
        fine_tuned_len = len(res['fine_tuned_response'].split())
        base_len = len(res['base_response'].split())
        print(f"Prompt {i + 1}:")
        print(f"  Fine-tuned Model Length: {fine_tuned_len} words")
        print(f"  Base Model Length: {base_len} words")

    print("\nHuman Preference Results:")
    total_evaluations = sum(human_preferences.values())
    if total_evaluations > 0:
        print(f"Total Evaluations: {total_evaluations}")
        print(f"Fine-tuned Model Preferred: {human_preferences['fine_tuned']} times "
              f"({(human_preferences['fine_tuned'] / total_evaluations) * 100:.1f}%)")
        print(f"Base Model Preferred: {human_preferences['base']} times "
              f"({(human_preferences['base'] / total_evaluations) * 100:.1f}%)")
        print(f"Responses Judged Equal: {human_preferences['equal']} times "
              f"({(human_preferences['equal'] / total_evaluations) * 100:.1f}%)")
    else:
        print("No human evaluations were performed. Please run the script and provide preferences.")

    print(
        "\nEvaluation complete. Review the detailed generated responses and human preferences to assess the impact of RLHF.")
    print(
        "Consider running this evaluation with more diverse prompts and multiple human evaluators for robust results.")


if __name__ == "__main__":
    main()
