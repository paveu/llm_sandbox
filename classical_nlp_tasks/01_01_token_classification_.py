"""
FILE: complete_custom_ner_final.py
DESCRIPTION:
    Kompletny skrypt budujący model NER od zera z pełną dokumentacją.
    Cel: Rozpoznawanie ilości (QTY) i składników (ING).
    Integracja: Weights & Biases (W&B) dla monitoringu.
"""

import os
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# ==============================================================================
# 0. KONFIGURACJA ŚRODOWISKA (W&B)
# ==============================================================================
# Ustawienie klucza API, abyś mógł widzieć wykresy na wandb.ai
os.environ['WANDB_API_KEY'] = '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'

# ==============================================================================
# 1. DEFINICJA ETYKIET (TWOJE "NAKLEJKI")
# ==============================================================================
# CO TRZYMAJĄ POSZCZEGÓLNE ETYKIETY (Standard BIO):
# "O" (Outside)     -> Słowa nieistotne, tło (np. "dodaj", "wymieszaj"). Indeks: 0
# "B-QTY" (Begin)   -> Pierwsze słowo określające ilość (np. "200", "pół"). Indeks: 1
# "B-ING" (Begin)   -> Pierwsze słowo nazwy składnika (np. "mąka", "cukier"). Indeks: 2
# "I-ING" (Inside)  -> Kontynuacja nazwy składnika (np. "pszenna" w "mąka pszenna"). Indeks: 3
label_list = ["O", "B-QTY", "B-ING", "I-ING"]

# Mapowania ID <-> Tekst (Model w środku widzi tylko liczby 0-3)
# ZWRACA: {0: 'O', 1: 'B-QTY', ...}
id2label = {i: label for i, label in enumerate(label_list)}
# ZWRACA: {'O': 0, 'B-QTY': 1, ...}
label2id = {label: i for i, label in enumerate(label_list)}



# ==============================================================================
# 2. PRZYGOTOWANIE DANYCH
# ==============================================================================
data = {
    "tokens": [
        ["Add", "200g", "of", "fresh", "spinach"],
        ["Mix", "5ml", "of", "water"]
    ],
    "ner_tags": [
        [0, 1, 0, 2, 3], # Add(O), 200g(B-QTY), of(O), fresh(B-ING), spinach(I-ING)
        [0, 1, 0, 2]     # Mix(O), 5ml(B-QTY), of(O), water(B-ING)
    ]
}

# ZWRACA: Obiekt Dataset (Zoptymalizowana tabela danych dla AI)
raw_dataset = Dataset.from_dict(data)

# ==============================================================================
# 3. TOKENIZACJA (TŁUMACZENIE TEKSTU NA LICZBY)
# ==============================================================================
model_checkpoint = "bert-base-cased"
# ZWRACA: Obiekt Tokenizera (Zamienia słowa na unikalne ID ze słownika BERT)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



def tokenize_and_align_labels(examples):
    # ZWRACA: Słownik z listami 'input_ids' (liczbowe reprezentacje słów)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True # Informujemy, że sami pocięliśmy tekst na słowa
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        # word_ids() ZWRACA: Listę wskazującą, który token należy do którego słowa.
        # Np. [None, 0, 1, 2, 2, None] (None to tokeny specjalne [CLS] i [SEP])
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # 1. Tokeny specjalne dostają -100 (Sygnał: "Zignoruj ten element przy nauce")
            if word_idx is None:
                label_ids.append(-100)
            # 2. Jeśli to nowe słowo (pierwszy subword):
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # ZWRACA: 0, 1, 2 lub 3
            # 3. Jeśli to kontynuacja pociętego słowa:
            else:
                label_ids.append(-100) # Nie uczymy modelu dwa razy na tym samym słowie

            previous_word_idx = word_idx

        labels.append(label_ids)

    # ZWRACA: Zaktualizowany słownik z kolumną 'labels' dopasowaną do tokenów
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



# map() ZWRACA: Gotowy do nauki tokenized_dataset
tokenized_datasets = raw_dataset.map(tokenize_and_align_labels, batched=True)

# ==============================================================================
# 4. BUDOWA I KONFIGURACJA MODELU
# ==============================================================================
# ZWRACA: Model BERT z nową warstwą wyjściową (4 neurony dla naszych klas)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# TrainingArguments ZWRACA: Komplet technicznych instrukcji (pogoda dla treningu)
args = TrainingArguments(
    "recipe-ner-final-project",
    eval_strategy="no",
    learning_rate=2e-5, # Prędkość nauki (bardzo małe kroki dla stabilności)
    num_train_epochs=15,
    weight_decay=0.01,
    logging_steps=1,
    report_to="wandb"   # Włącza przesyłanie wykresów do Weights & Biases
)

# data_collator ZWRACA: Funkcję wyrównującą długości zdań w paczkach (Padding)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer ZWRACA: Silnik treningowy łączący wszystko w całość
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    processing_class=tokenizer, # Używamy aktualnej nazwy zamiast starego 'tokenizer'
)

# ==============================================================================
# 5. START NAUKI I TEST KOŃCOWY
# ==============================================================================
print("--- Start nauki i synchronizacji z W&B ---")
# ZWRACA: Podsumowanie treningu (czas, strata błędu)
trainer.train()

test_text = "Take 300g of sugar"
# ZWRACA: Słownik tensorów PyTorch gotowy do obliczeń
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

with torch.no_grad(): # Wyłączenie gradientów dla oszczędności pamięci
    # ZWRACA: Macierz logitów (surowe punkty dla każdej klasy)
    logits = model(**inputs).logits

# ZWRACA: Listę ID klas (0-3) z najwyższym wynikiem dla każdego słowa
predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
# ZWRACA: Listę kawałków tekstu (tokenów)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

print("\n--- Wynik testu ---")
for token, pred in zip(tokens, predictions):
    if token not in tokenizer.all_special_tokens:
        label = id2label[pred]
        print(f"Token: {token:10} | Klasa: {label}")