# @Time     : 2025/5/22 13:24
# @Author   : Kun Lin
# @Filename : tb_LmsysChat.py


# @Time     : 2025/5/22 15:00
# @Author   : Gemini
# @Filename : test_lmsys_chat_model.py

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


# --- Configuration and Argument Parsing ---
def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned model for dialogue generation based on lmsys/lmsys-chat-1m dataset and compare with the base model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/lmsys_chat_gpt2_1",  # Default path for the fine-tuned model
        help="Path to the fine-tuned model and tokenizer directory."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="gpt2",  # Base model name from Hugging Face Hub
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../models",  # Default cache directory
        help="Directory to cache Hugging Face models and tokenizers."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,  # Max length for dialogue, potentially longer for chat models
        help="Maximum length of the generated answer sequence (prompt + generated text)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation. Lower values make output more deterministic."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter for text generation."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter for text generation."
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=2,
        help="If set to int > 0, all ngrams of that size can only occur once."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run the model on. 'auto' will use CUDA if available, else CPU."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../res",  # Output directory for results
        help="Directory to save the test results."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fine_tuned_model_path = args.model_path
    base_model_name = args.base_model_name
    custom_cache_dir = args.cache_dir
    max_length = args.max_length
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    no_repeat_ngram_size = args.no_repeat_ngram_size
    output_dir = args.output_dir

    # Determine the device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Check if the fine-tuned model exists
    if not os.path.exists(fine_tuned_model_path):
        print(f"Error: Fine-tuned model directory not found at '{fine_tuned_model_path}'.")
        print("Please ensure the training program has been run and the model saved correctly.")
        exit(1)

    # --- Load models and tokenizers ---
    print("Loading fine-tuned model and tokenizer...")
    try:
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, cache_dir=custom_cache_dir)
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, cache_dir=custom_cache_dir)
    except Exception as e:
        print(f"Error loading fine-tuned model or tokenizer from '{fine_tuned_model_path}': {e}")
        exit(1)

    print(f"Loading base model '{base_model_name}' and tokenizer...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=custom_cache_dir)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=custom_cache_dir)
    except Exception as e:
        print(f"Error loading base model or tokenizer '{base_model_name}': {e}")
        exit(1)

    # Ensure pad_token exists for consistent generation
    for tokenizer, model_config in [(fine_tuned_tokenizer, fine_tuned_model.config),
                                    (base_tokenizer, base_model.config)]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if model_config.pad_token_id is None:
                model_config.pad_token_id = tokenizer.eos_token_id
        elif model_config.pad_token_id is None:
            model_config.pad_token_id = tokenizer.pad_token_id

    # Move models to the determined device
    fine_tuned_model.to(device)
    fine_tuned_model.eval()

    base_model.to(device)
    base_model.eval()

    # --- Generation function ---
    def generate_dialogue_response(model, tokenizer, prompt_text):
        """Generates a dialogue response using the specified model and tokenizer."""
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=False)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=False
            )

        num_prompt_tokens = input_ids.shape[1]
        generated_ids = output_sequences[0][num_prompt_tokens:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response

    # --- Test cases (Dialogue Prompts) ---
    test_prompts = [
        "Hello, how can I help you today?",
        "What are your thoughts on artificial intelligence?",
        "Can you tell me a short story about a brave knight?",
        "I'm feeling a bit down. Do you have any advice?",
        "Explain quantum physics in simple terms.",
        "What's the weather like in Tokyo right now?",
        "Tell me about the history of the internet.",
        "What is the meaning of life?",
        "How do I cook a perfect pasta?",
        "Recommend a good book for me to read."
    ]

    # --- Run tests and save results ---
    print("\n===== Running Dialogue Tests and Saving Results =====")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"lmsys_chat_test_results_{timestamp}.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(f"LMSYS Chat Model Test Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Fine-tuned Model: {fine_tuned_model_path}\n")
        f.write(f"Base Model: {base_model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(
            f"Max Length: {max_length}, Temp: {temperature}, Top-p: {top_p}, Top-k: {top_k}, NoRepeatNgram: {no_repeat_ngram_size}\n")
        f.write("-" * 70 + "\n\n")

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nProcessing Prompt {i}...")
            f.write(f"===== Prompt {i} =====\n")
            f.write(f"Prompt: {prompt}\n\n")

            print("Generating response with Base Model...")
            base_response = generate_dialogue_response(base_model, base_tokenizer, prompt)
            f.write(f"Base Model Response: {base_response}\n\n")
            print(f"Base Model Response: {base_response}")

            print("Generating response with Fine-tuned Model...")
            fine_tuned_response = generate_dialogue_response(fine_tuned_model, fine_tuned_tokenizer, prompt)
            f.write(f"Fine-tuned Model Response: {fine_tuned_response}\n")
            print(f"Fine-tuned Model Response: {fine_tuned_response}")

            f.write("-" * 70 + "\n\n")

    print(f"\n===== Testing Complete. Results saved to {output_filepath} =====")

    # --- Interactive test ---
    print("\n===== Interactive Dialogue Mode (using Fine-tuned Model) =====")
    print("Enter your dialogue prompt below. Type 'q' or 'quit' to exit.")
    while True:
        user_input = input("\nYour prompt: ").strip()
        if user_input.lower() in ['q', 'quit']:
            print("Exiting interactive mode. Goodbye!")
            break
        if not user_input:
            print("Input cannot be empty.")
            continue

        response = generate_dialogue_response(fine_tuned_model, fine_tuned_tokenizer, user_input)
        print(f"Model Response: {response}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")


if __name__ == "__main__":
    main()
