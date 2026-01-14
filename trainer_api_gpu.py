import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ==========================================
# 1. KONFIGURACJA I DANE
# ==========================================

# Checkpoint to nazwa "mózgu" modelu, który pobieramy.
# 'bert-base-uncased' to klasyk – rozumie angielski, nie rozróżnia wielkości liter.
checkpoint = "bert-base-uncased"

# Ładujemy zbiór MRPC z benchmarku GLUE.
# Ten zbiór zawiera pary zdań – musimy ocenić, czy znaczą to samo (parafraza).
raw_datasets = load_dataset("glue", "mrpc")

# Tokenizer zamienia tekst na liczby. Musi być ten sam, na którym trenowano model!
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# ==========================================
# 2. PRZYGOTOWANIE (PREPROCESSING)
# ==========================================

def tokenize_function(example):
    # Podajemy dwa zdania. Tokenizer doda specjalne separatory (np. [SEP] w BERT),
    # aby model wiedział, gdzie kończy się pierwsze zdanie, a zaczyna drugie.
    # truncation=True skraca zbyt długie teksty, by nie przekroczyły limitu modelu.
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# map() nakłada tokenizację na cały zbiór danych (wszystkie wiersze).
# batched=True przyspiesza proces, przetwarzając wiele wierszy naraz.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# DataCollatorWithPadding odpowiada za "wyrównywanie" długości zdań w paczkach (batchach).
# Modele LLM wymagają, by w danej paczce wszystkie dane miały tę samą długość.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ==========================================
# 3. METRYKI I OCENA
# ==========================================

def compute_metrics(eval_preds):
    # Ładujemy funkcję oceniającą specyficzną dla zbioru MRPC (Accuracy i F1).
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds

    # Logity to surowe wyniki z ostatniej warstwy modelu (np. [-2.1, 1.5]).
    # argmax(-1) wybiera indeks z najwyższą wartością (tu: indeks 1).
    predictions = np.argmax(logits, axis=-1)

    # Porównujemy predykcje modelu z prawdziwymi etykietami (labels).
    return metric.compute(predictions=predictions, references=labels)


# ==========================================
# 4. INICJALIZACJA MODELU
# ==========================================

# AutoModelForSequenceClassification dodaje na górze BERTa nową, "pustą" warstwę
# (klasyfikator), którą będziemy douczać. num_labels=2, bo mamy klasy: "Tak" lub "Nie".
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# ==========================================
# 5. PARAMETRY TRENINGU (TrainingArguments)
# ==========================================

training_args = TrainingArguments(
    output_dir="test-trainer",  # Gdzie zapisywać postępy i gotowy model.
    eval_strategy="epoch",  # Wykonaj test (ewaluację) po każdej epoce.
    save_strategy="epoch",  # Zapisz "punkt kontrolny" po każdej epoce.
    learning_rate=2e-5,  # Jak bardzo zmieniać wagi (mała liczba = stabilna nauka).
    per_device_train_batch_size=8,  # Ile przykładów model widzi naraz (zależy od RAMu Twojej karty).
    num_train_epochs=3,  # Ile razy model przeczyta cały zbiór danych.
    weight_decay=0.01,  # Zapobiega "przeuczeniu" (overfitting), by model nie kulił danych na pamięć.
    fp16=torch.cuda.is_available(),  # Jeśli masz kartę NVIDIA, użyj 16-bitowej precyzji (szybszy trening).
    logging_steps=10,  # Co ile kroków wypisywać informację o stracie (loss) w konsoli.
)

# ==========================================
# 6. TWORZENIE TRAINERA I START
# ==========================================

# Trainer to "dyrygent" – łączy model, dane, parametry i metryki w jeden mechanizm.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # Dane do nauki.
    eval_dataset=tokenized_datasets["validation"],  # Dane do sprawdzianu.
    data_collator=data_collator,  # Narzędzie do robienia paczek danych.
    processing_class=tokenizer,  # Przekazujemy tokenizer do obsługi danych.
    compute_metrics=compute_metrics,  # Nasza funkcja licząca skuteczność.
)

print("\n--- Rozpoczynam trening. Jeśli masz GPU, zobaczysz pasek postępu. ---")

# Główna komenda uruchamiająca proces nauki.
trainer.train()

print("\n--- Trening zakończony! Model znajduje się w folderze 'test-trainer'. ---")