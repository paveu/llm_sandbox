"""
This script demonstrates the implementation of Chat Templates using the Hugging Face Transformers library.

Purpose:
    - Loads a subset of the 'HuggingFaceTB/smoltalk' dataset.
    - Utilizes the SmolLM2-135M-Instruct tokenizer to apply the ChatML format.
    - Converts raw conversation dictionaries into a single formatted string that
      the model can use for Supervised Fine-Tuning (SFT).

Expected Behavior:
    - Downloads the 'everyday-conversations' configuration of the smoltalk dataset.
    - Prints a sample of the formatted text to the console.
    - The output should contain specific ChatML control tokens such as <|im_start|>
      and <|im_end|> wrapping the system, user, and assistant roles.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

# --- Expected Behavior: Downloads the 'everyday-conversations' configuration ---
# (Line 26 fulfills the requirement to load the specific dataset and config)
dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:10]")

# 2. Load the tokenizer for a model that uses ChatML
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Processing function
def convert_to_chatml(example):
    # --- Purpose: Accesses raw conversation dictionaries ---
    messages = example["messages"]

    # --- Expected Behavior: Output contains ChatML control tokens (<|im_start|>, etc.) ---
    # (Line 39 applies the Jinja2 template defined in the tokenizer_config)
    formatted_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"formatted_text": formatted_chat}

# 4. Apply the transformation
processed_dataset = dataset.map(convert_to_chatml)

# 5. Output the result
if __name__ == "__main__":
    # --- Expected Behavior: Prints a sample of the formatted text to the console ---
    # (Line 50 fulfills the final requirement for visual verification)
    print(f"--- Formatted Example (ChatML) using {model_id} ---\n")
    print(processed_dataset[0]["formatted_text"])