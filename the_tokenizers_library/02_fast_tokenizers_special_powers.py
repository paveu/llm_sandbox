import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==============================================================================
# BEGINNER'S LLM GLOSSARY & CONTEXT:
# 1. CHECKPOINT: A pre-trained "brain." It contains the weights (learned patterns).
# 2. INFERENCE: The process of using a trained model to make predictions.
# 3. OFFSET MAPPING: The map that connects "Token #5" back to "Characters 12-18".
# 4. TENSORS: The math format (multi-dimensional arrays) that models require.
# ==============================================================================

print("--- Step 1: Loading the Model and Tokenizer (Checkpoint) ---")
# We use a model fine-tuned on the CoNLL-03 dataset, which is the "gold standard"
# for recognizing People (PER), Organizations (ORG), and Locations (LOC).
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"

# AutoTokenizer: A smart factory that picks the right tokenizer for the model.
# By default, it loads the "Fast" version (written in Rust) which is much
# more capable than the pure Python "Slow" version.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Our raw input string. Note the mixed names, organizations, and punctuation.
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
print(f"Input Text: {example}\n")

# ---------------------------------------------------------
print("--- Step 2: Verifying Tokenizer Capabilities ---")
# is_fast: This boolean confirms if we have access to the Rust-backed features.
# If this were False, the 'return_offsets_mapping' feature below would crash.
print(f"Is this a 'Fast' tokenizer? {tokenizer.is_fast}")

# ---------------------------------------------------------
print("\n--- Step 3: Encoding with Offset Mapping (The Magic Map) ---")
# return_offsets_mapping=True: This is the "Special Power". It tells the tokenizer
# to remember the exact character spans for every single token it creates.
# return_tensors="pt": Returns PyTorch tensors (the format the model expects).
inputs = tokenizer(example, return_offsets_mapping=True, return_tensors="pt")

# BatchEncoding: The object returned by the tokenizer.
# We extract 'offset_mapping' which is a list of (start_char, end_char) tuples.
offsets = inputs["offset_mapping"][0]
# tokens(): Converts the numerical IDs back into human-readable strings for us to see.
tokens = inputs.tokens()

print(f"Tokens created: {tokens}")
print(f"Sample offsets (first 5): {offsets[:5].tolist()}")
# Note: [0, 0] represents special tokens like [CLS] that don't map to actual text.

# Word IDs:
# This is crucial! If 'Sylvain' is broken into ['S', '##yl', '##va', '##in'],
# word_ids() assigns the same index (e.g., 3, 3, 3, 3) to all of them.
# This helps us know they are part of the same original word.
print(f"Word IDs (which tokens belong to which word): {inputs.word_ids()}")

# ---------------------------------------------------------
print("\n--- Step 4: Model Inference (The Prediction Phase) ---")
# The model is a neural network. It doesn't know what "offsets" are.
# We create 'model_inputs' by filtering out the offset metadata so it doesn't cause errors.
model_inputs = {k: v for k, v in inputs.items() if k != "offset_mapping"}

# torch.no_grad(): A memory-saving trick. We aren't training (updating weights),
# so we don't need to track gradients. This makes inference faster.
with torch.no_grad():
    outputs = model(**model_inputs)

# outputs.logits: The raw scores for each of the 9 possible NER classes for EVERY token.
# .argmax(dim=-1): We pick the index of the highest score (the model's "best guess").
predictions = outputs.logits.argmax(dim=-1)[0].tolist()

# softmax: Converts raw logits into 0.0-1.0 probabilities (confidence levels).
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

print(f"Model generated labels for all {len(predictions)} tokens.")

# ---------------------------------------------------------
print("\n--- Step 5: Reconstruction using Offset Mapping ---")
# The model gives us a prediction for "##yl" and "##va". We want to combine
# them back into "Sylvain". Offsets allow us to do this perfectly.
results = []
idx = 0

# We iterate through the sequence of tokens and their predicted labels.
while idx < len(predictions):
    pred_idx = predictions[idx]
    label = model.config.id2label[pred_idx]  # Map ID (e.g., 4) to label (e.g., 'I-PER')

    # 'O' (Outside) means the token is not a named entity (like the word 'is' or 'at').
    if label != "O":
        # label[2:]: Removes the BIO prefix (e.g., 'B-PER' -> 'PER').
        entity_type = label[2:]
        start, _ = offsets[idx]  # Mark the starting character of this entity.

        print(f"Found entity part: '{tokens[idx]}' predicted as {label}")

        # Grouping Logic:
        # We look ahead. If the next tokens belong to the same entity type,
        # we group them into a single result until we hit a different label or 'O'.
        all_scores = []
        while (idx < len(predictions) and
               model.config.id2label[predictions[idx]].endswith(entity_type)):
            # We keep track of the scores to calculate an average confidence.
            all_scores.append(probabilities[idx][predictions[idx]])
            _, end = offsets[idx]  # Update the 'end' character with every new token.
            idx += 1

        # THE POWER OF OFFSETS:
        # Instead of manually cleaning "##" or handling whitespace, we use the
        # 'start' and 'end' coordinates to slice the ORIGINAL 'example' string.
        final_word = example[start:end]

        results.append({
            "entity_group": entity_type,
            "confidence": np.mean(all_scores).item(),  # Average probability of the group.
            "word": final_word,
            "start": start.item(),
            "end": end.item(),
        })
    else:
        # If the label is 'O', just move to the next token.
        idx += 1

# ---------------------------------------------------------
print("\n--- Final Results (Cleanly Grouped Entities) ---")
import pprint

# This mimics the output of the high-level 'pipeline("ner", aggregation_strategy="simple")'
pprint.pprint(results)