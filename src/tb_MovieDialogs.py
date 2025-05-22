# @Time     : 2025/5/21 15:34
# @Author   : Kun Lin
# @Filename : tb_MovieDialogs.py

# @Time     : 2025/05/21
# @Author   : Gemini
# @Filename : test_cornell_model.py

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


# --- Configuration and Argument Parsing ---
def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned GPT-2 model for dialogue generation and compare with the base model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/cornell_movie_dialog_gpt2_finetuned_local_zip",  # Default path from your finetune script
        help="Path to the fine-tuned model and tokenizer directory."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="gpt2",
        help="Name of the original base model from Hugging Face Hub (e.g., 'gpt2', 'gpt2-medium')."
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
        default=256,  # Max length for dialogue
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
        default="../res",  # Changed output directory
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
        print("Please ensure the training program (finetune2.py) has been run and the model saved correctly.")
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
            if model_config.pad_token_id is None:  # Check model_config as well
                model_config.pad_token_id = tokenizer.eos_token_id
        elif model_config.pad_token_id is None:  # If pad_token exists but not in config
            model_config.pad_token_id = tokenizer.pad_token_id

    # Move models to the determined device
    fine_tuned_model.to(device)
    fine_tuned_model.eval()

    base_model.to(device)
    base_model.eval()

    # --- Generation function ---
    def generate_dialogue_response(model, tokenizer, prompt_text):
        """Generates a dialogue response using the specified model and tokenizer."""
        # For dialogue, the prompt is usually just the user's turn.
        # The model should have learned to continue the conversation.
        # No special "Question:" or "Answer:" formatting needed unless your fine-tuning specifically used it.

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                           padding=False)  # Padding False, generate handles it
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,  # Crucial for open-ended generation
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

        # Sometimes the model might generate the EOS token immediately or very early.
        # You might want to post-process the response, e.g., remove partial next turns if it generates for multiple speakers.
        # For simplicity, we'll take the raw output after the prompt.
        return response

    # --- Test cases (Dialogue Prompts) ---
    test_prompts = [
        # 1. Casual greeting / checking in, with a movie-dialogue flavor
        #    (Replaces: "Hey, how are you doing today?")
        "You alright there? You seem a little distracted.",

        # 2. Seeking opinion or information, phrased as it might be in a script
        #    (Replaces: "What do you think about this new movie?")
        "What's your take on this whole mess?",

        # 3. Expressing a problem or seeking advice, in a conversational, movie-like manner
        #    (Replaces: "I'm not sure what to do about this problem.")
        "I'm kind of stuck here. Any brilliant ideas on how to proceed?",

        # 4. Request for elaboration or a story, more engaging than a simple joke request
        #    (Replaces: "Can you tell me a joke?")
        "That sounds like quite a story. Care to fill me in on the details?",

        # 5. A statement that invites a reaction or continuation, common in dialogue flow
        #    (Replaces: "That's an interesting point of view.")
        "Well, that certainly complicates things, doesn't it?",

        # 6. Explicit turn-taking, using character names and a line that sets up a response,
        #    inspired by actual lines from the Cornell dataset (e.g., "10 Things I Hate About You" - movie m0).
        #    (Replaces: "User A: Want to grab some dinner later?<|endoftext|>User B:")
        #    Line l199 from m0: BIANCA: You're asking me out? That's so cute. What's your name again?
        "BIANCA: You're asking me out? That's so cute. What's your name again?<|endoftext|>CAMERON:",

        # 7. A specific, potentially challenging or intriguing line directly from the Cornell dataset,
        #    maintaining the spirit of your original example.
        #    Line l204 from m0: BIANCA: So you're the kind of guy who likes to watch people cry?
        "BIANCA: So you're the kind of guy who likes to watch people cry?"
    ]

    # --- Run tests and save results ---
    print("\n===== Running Dialogue Tests and Saving Results =====")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"cornell_dialogue_test_results_{timestamp}.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(f"Cornell Movie Dialog Test Results\n")
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