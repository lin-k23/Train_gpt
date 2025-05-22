# @Time     : 2025/5/21 00:00
# @Author   : Kun Lin
# @Filename : tb_main_optimized.py

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime # Import datetime to add a timestamp to the output file

# --- Configuration and Argument Parsing ---
def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Test a fine-tuned GPT-2 model for math problem-solving and compare with the base model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/fine_tuned_gpt2_math_2",
        help="Path to the fine-tuned model and tokenizer directory."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="gpt2", # Assuming the fine-tuned model was based on gpt2
        help="Name of the original base model from Hugging Face Hub (e.g., 'gpt2', 'gpt2-medium')."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../models",
        help="Directory to cache Hugging Face models and tokenizers. "
             "This will be passed to from_pretrained if specified."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run the model on. 'auto' will use CUDA if available, else CPU."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../res",
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
        # Load fine-tuned model
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, cache_dir=custom_cache_dir)
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, cache_dir=custom_cache_dir)
    except Exception as e:
        print(f"Error loading fine-tuned model or tokenizer from '{fine_tuned_model_path}': {e}")
        print("Please verify the path and ensure the directory contains valid model files.")
        exit(1)

    print(f"Loading base model '{base_model_name}' and tokenizer...")
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=custom_cache_dir)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=custom_cache_dir)
    except Exception as e:
        print(f"Error loading base model or tokenizer '{base_model_name}': {e}")
        print("Please check the model name and your internet connection.")
        exit(1)


    # Ensure pad_token exists for consistent generation behavior for both tokenizers
    if fine_tuned_tokenizer.pad_token is None:
        fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
        if fine_tuned_model.config.pad_token_id is None:
            fine_tuned_model.config.pad_token_id = fine_tuned_model.config.eos_token_id

    # Base model might already have a pad_token, but let's ensure consistency if needed
    if base_tokenizer.pad_token is None:
         base_tokenizer.pad_token = base_tokenizer.eos_token
         if base_model.config.pad_token_id is None:
             base_model.config.pad_token_id = base_model.config.eos_token_id


    # Move models to the determined device
    fine_tuned_model.to(device)
    fine_tuned_model.eval()  # Set to evaluation mode

    base_model.to(device)
    base_model.eval() # Set to evaluation mode

    # --- Generation function ---
    def generate_answer(model, tokenizer, question):
        """Generates an answer to a math problem using the specified model and tokenizer."""
        # Format input text, consistent with training format
        input_text = f"Question: {question}\nAnswer: "

        # Encode input and get attention_mask
        # Using the tokenizer directly provides both input_ids and attention_mask
        # Add padding=True and return_attention_mask=True explicitly for clarity
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, return_attention_mask=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) # Get the attention mask

        # Generate answer
        with torch.no_grad(): # Disable gradient calculation for inference
            # max_length in generate refers to the total length of the sequence (prompt + generated text)
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                max_length=max_length,          # Use the configured max_length
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,                 # Enable sampling
                temperature=temperature,        # Use the configured temperature
                top_p=top_p,                    # Use the configured top_p
                early_stopping=False            # Set to False as num_beams=1 (default) and do_sample=True
            )

        # Decode only the newly generated tokens
        # output_sequences contains the input_ids followed by the generated_ids
        num_prompt_tokens = input_ids.shape[1] # Get the length of the prompt tokens
        # Slice the output to get only the generated token IDs, excluding the input prompt
        # Ensure we don't slice beyond the generated sequence length
        generated_ids = output_sequences[0][num_prompt_tokens:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return answer


    # --- Test cases ---
    test_problems = [
        "If the length of a rectangle is 8 meters and the width is 5 meters, what is its area?",
        "Ming has 12 apples. He gives 3 to Hong and 2 to Gang. How many apples does Ming have left?",
        "The radius of a circle is 6 centimeters. What is its circumference? (Take π as 3.14)",
        "A train travels at a speed of 72 kilometers per hour. How many kilometers can it travel in 2.5 hours?",
        "A shop has a batch of goods. On the first day, 1/3 of the total was sold. On the second day, 40% of the remainder was sold. If 120 items are left, how many items were there originally in this batch?"
    ]

    # --- Run tests and save results ---
    print("\n===== Running Tests and Saving Results =====")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"math_test_results_{timestamp}.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(f"Math Problem Test Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Fine-tuned Model: {fine_tuned_model_path}\n")
        f.write(f"Base Model: {base_model_name}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n\n")

        for i, problem in enumerate(test_problems, 1):
            print(f"\nProcessing Problem {i}...")
            f.write(f"===== Problem {i} =====\n")
            f.write(f"Question: {problem}\n\n")

            # Generate answer from base model
            print("Generating answer with Base Model...")
            base_answer = generate_answer(base_model, base_tokenizer, problem)
            f.write(f"Base Model Answer: {base_answer}\n\n")
            print(f"Base Model Answer: {base_answer}") # Also print to console

            # Generate answer from fine-tuned model
            print("Generating answer with Fine-tuned Model...")
            fine_tuned_answer = generate_answer(fine_tuned_model, fine_tuned_tokenizer, problem)
            f.write(f"Fine-tuned Model Answer: {fine_tuned_answer}\n")
            print(f"Fine-tuned Model Answer: {fine_tuned_answer}") # Also print to console

            f.write("-" * 50 + "\n\n")

    print(f"\n===== Testing Complete. Results saved to {output_filepath} =====")

    # --- Interactive test ---
    print("\n===== Interactive Mode (using Fine-tuned Model) =====")
    print("Enter math problems below. Type 'q' or 'quit' to exit.")
    while True:
        user_input = input("\nPlease enter a math problem (or 'q' to quit): ").strip()
        if user_input.lower() in ['q', 'quit']:
            print("Exiting interactive mode. Goodbye!")
            break

        if not user_input:
            print("Input cannot be empty. Please enter a problem.")
            continue

        # Use the fine-tuned model for interactive mode
        answer = generate_answer(fine_tuned_model, fine_tuned_tokenizer, user_input)
        print(f"Model Answer: {answer}")

    # Optional: Clear CUDA cache if using GPU, to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

if __name__ == "__main__":
    main()