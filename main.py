import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

# 1. Przygotowanie (to co już znasz)
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Ładujemy model z "głową" do klasyfikacji sekwencji (automatycznie wie, że MRPC ma 2 klasy)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")
print("raw_datasets", raw_datasets)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("tokenized_datasets", tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 2. Wybieramy 2 pary zdań do testu (żeby było czytelnie)
samples = tokenized_datasets["train"][:2]
print("samples", samples)
# Zapamiętujemy oryginalne teksty, żeby móc je potem wyświetlić
raw_texts = raw_datasets["train"][:2]
print("raw_texts", raw_texts)
# Usuwamy kolumny tekstowe dla modelu
model_inputs = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print("model_inputs", model_inputs)
# 3. Pakujemy w batch (dodanie paddingu)
batch = data_collator(model_inputs)
print("batch", batch)

# 4. PRZEKAZUJEMY DO MODELU
model.eval()  # Przełączamy model w tryb oceny (wyłącza np. Dropout)
with torch.no_grad():  # Mówimy PyTorchowi, że nie będziemy teraz trenować, więc nie musi liczyć gradientów
    outputs = model(**batch)

# 5. WYCIĄGAMY WNIOSKI (Human Readable)
predictions = F.softmax(outputs.logits, dim=-1)  # Zamieniamy logity na prawdopodobieństwo (0-1)

print("-" * 30)
for i in range(2):
    print(f"PARA ZDAŃ NR {i + 1}:")
    print(f"Z1: {raw_texts['sentence1'][i]}")
    print(f"Z2: {raw_texts['sentence2'][i]}")

    # Wyciągamy prawdopodobieństwo dla klasy "Parafraza" (index 1)
    is_paraphrase_prob = predictions[i][1].item()
    actual_label = "Parafraza" if raw_texts['label'][i] == 1 else "To nie parafraza"

    print(f"WERDYKT MODELU: {'Parafraza' if is_paraphrase_prob > 0.5 else 'To nie parafraza'}")
    print(f"PEWNOŚĆ MODELU: {is_paraphrase_prob:.2%}")
    print(f"STAN FAKTYCZNY: {actual_label}")
    print("-" * 30)