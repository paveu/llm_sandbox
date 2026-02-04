import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ==============================================================================
# LLM NEWCOMER CONCEPTS:
# 1. THE CHALLENGE: Models like BERT/DistilBERT have a fixed "Context Window"
#    (usually 512 tokens). They physically cannot see text beyond that limit.
# 2. THE SOLUTION: Sliding Windows. If a book is 1000 words, we show the model
#    words 1-100, then 90-190, then 180-280. The overlap (STRIDE) ensures the
#    model doesn't lose context at the edges of the cuts.
# 3. LOGITS: These are "raw scores." A high Logit at position 5 means the model
#    is very confident that the answer starts at Token #5.
# ==============================================================================

print("--- Step 1: Loading the Brain (Model) and the Translator (Tokenizer) ---")
# This model is a 'DistilBERT' - a smaller, faster version of BERT,
# specifically trained to find answers in text (SQuAD dataset).
model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

question = "Where does Sylvain work?"
context = """My name is Sylvain and I am a machine learning engineer at Hugging Face. 
Hugging Face is a company based in New York City. The headquarters are in DUMBO, Brooklyn. 
Sylvain has been working there since 2018 and enjoys building open-source tools for the community."""

# ---------------------------------------------------------
print("\n--- Step 2: Slicing the Text (The Sliding Window) ---")
# In LLMs, we 'truncate' text that is too long. But in QA, we don't want to
# lose data, so we use 'return_overflowing_tokens' to create multiple chunks.

inputs = tokenizer(
    question,
    context,
    max_length=32,      # We use a tiny max_length (32) to FORCE the windowing effect.
    truncation="only_second", # IMPORTANT: Only cut the context, never the question.
    stride=10,          # The overlap. Chunks will share 10 tokens to maintain context.
    padding=True,       # Ensures all chunks are exactly 32 tokens (needed for Tensors).
    return_overflowing_tokens=True, # The 'Special Power': allows creating multiple chunks.
    return_offsets_mapping=True,    # The 'Special Power': remembers char positions.
    return_tensors="pt"             # 'pt' stands for PyTorch.
)

# Each chunk is called a 'feature'.
print(f"Text split into {len(inputs['input_ids'])} overlapping chunks.")
print(f"Tensor Shape: {inputs['input_ids'].shape} (Chunks x Tokens)")

# ---------------------------------------------------------
print("\n--- Step 3: Neural Network Processing ---")
# The model is a function. We pass the numbers (input_ids) and
# the attention_mask (which tells the model to ignore padding zeros).
# We exclude 'offset_mapping' because it's metadata for us, not for the math.
model_inputs = {
    k: v for k, v in inputs.items()
    if k not in ["offset_mapping", "overflow_to_sample_mapping"]
}

with torch.no_grad(): # Disable gradient math to save RAM during prediction.
    outputs = model(**model_inputs)

# start_logits: Probability of each token being the START of the answer.
# end_logits: Probability of each token being the END of the answer.
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# ---------------------------------------------------------
print("\n--- Step 4: Finding the Answer (Decoding) ---")

# Let's look at chunk #0.
# argmax finds the index of the highest score in the list of logits.
sequence_idx = 0
start_token_pos = torch.argmax(start_logits[sequence_idx])
end_token_pos = torch.argmax(end_logits[sequence_idx])

# We retrieve the 'Map' for this specific chunk.
offsets = inputs["offset_mapping"][sequence_idx]

# Map the Token Index (e.g., Token #15) to Character Index (e.g., Char #45)
# offsets[idx] gives a tuple: (start_character, end_character)
start_char = offsets[start_token_pos][0]
end_char = offsets[end_token_pos][1]

# Now we go back to the original string and slice it.
# This handles all '##' subword cleaning automatically!
predicted_answer = context[start_char:end_char]

print(f"Question: {question}")
print(f"Found Answer: '{predicted_answer}'")
print(f"Location: Context characters {start_char} through {end_char}")

# ---------------------------------------------------------
# PRO-TIP FOR THE JOURNEY:
# In a real app, you would check ALL chunks (sequence_idx 0, 1, 2...)
# and pick the one where (start_logit + end_logit) is the highest sum.
# This ensures you find the best answer even if it's in the middle of the text!