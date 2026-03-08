import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ==========================================
# 1. PRZYGOTOWANIE DANYCH
# ==========================================
checkpoint = "bert-base-uncased"
print(f"\n[1/6] Pobieranie modelu i danych: {checkpoint}...")

raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# --- DIAGNOSTYKA: SUROWE DANE ---
sample = raw_datasets["train"][0]
print(
    f"👉 Przykład surowego tekstu:\n   Zdanie 1: {sample['sentence1']}\n   Zdanie 2: {sample['sentence2']}\n   Etykieta (Label): {sample['label']} (1=Parafraza, 0=Różne)"
)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


print("\n[2/6] Tokenizacja danych...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# --- DIAGNOSTYKA: TOKENY ---
tokenized_sample = tokenized_datasets["train"][0]
print(f"👉 Tekst po zamianie na ID (input_ids - pierwsze 10): {tokenized_sample['input_ids'][:10]}...")
print(f"👉 Co model widzi (dekodowanie): {tokenizer.decode(tokenized_sample['input_ids'][:10])}")

# Skracamy zbiór dla CPU (bezpiecznik)
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle(seed=42).select(range(50))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ==========================================
# 2. METRYKI
# ==========================================
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # --- DIAGNOSTYKA: LOGITY ---
    print(f"\n--- Raport z Ewaluacji ---")
    print(f"Kształt logitów: {logits.shape} (Przykłady x Klasy)")
    print(f"Pierwsze 3 przewidywania modelu: {predictions[:3]}")
    print(f"Pierwsze 3 prawdziwe etykiety:  {labels[:3]}")

    return metric.compute(predictions=predictions, references=labels)


# ==========================================
# 3. MODEL I TRENING
# ==========================================
print("\n[3/6] Ładowanie wag modelu...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

training_args = TrainingArguments(
    output_dir="test-trainer-cpu",
    use_cpu=True,  # Wymuszamy CPU
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=5,  # Print co 5 kroków treningu
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n[4/6] Start treningu...")
# Trainer automatycznie wypisuje tabelkę z 'loss' (stratą)
trainer.train()

print("\n[5/6] Zapisywanie modelu...")
trainer.save_model("./moj_model_cpu")

# ==========================================
# 4. TEST PRAKTYCZNY (INFERENCJA)
# ==========================================
print("\n[6/6] Test praktyczny na nowych zdaniach:")
from transformers import pipeline

classifier = pipeline("text-classification", model="./moj_model_cpu", device=-1)  # device=-1 wymusza CPU

s1 = "The cat sits on the mat."
s2 = "A feline is resting on the rug."

wynik = classifier(f"{s1} [SEP] {s2}")  # Ręczne połączenie zdań dla pipeline
print(f"Zdanie A: {s1}")
print(f"Zdanie B: {s2}")
print(f"Wynik klasyfikacji: {wynik}")
