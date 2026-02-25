"""
FILE: complete_custom_ner_final.py
DESCRIPTION:
    Kompletny skrypt budujący model NER od zera z pełną dokumentacją.
    Cel: Rozpoznawanie ilości (QTY) i składników (ING).
    Integracja: Weights & Biases (W&B) dla monitoringu wizualnego.
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

"""
================================================================================
CO ROBI TEN SKRYPT? (CUSTOM NER DLA PRZEPISÓW)
================================================================================
1. DEFINIUJE WŁASNY JĘZYK (BIO TAGGING): 
   Uczy model, że "200g" to ilość (QTY), a "fresh spinach" to składnik (ING). 
   Wykorzystuje format BIO (Begin, Inside, Outside), aby odróżnić początek nazwy 
   składnika od jego kontynuacji.

2. SYNCHRONIZUJE MONITORING (W&B): 
   Każdy krok nauki (Loss) jest wysyłany do zewnętrznego panelu Weights & Biases, 
   co pozwala na analizę wykresów zbieżności modelu w czasie rzeczywistym.

3. DOPASOWUJE ETYKIETY DO SUB-TOKENÓW: 
   Jeśli tokenizer BERT rozbije słowo "spinach" na fragmenty, skrypt dba o to, 
   by etykieta została przypisana tylko do pierwszej części, a reszta była ignorowana 
   (-100), co zapobiega konfuzji modelu.
================================================================================
"""

# ==============================================================================
# 0. KONFIGURACJA ŚRODOWISKA (W&B)
# ==============================================================================
# Ustawienie klucza API pozwala na automatyczną synchronizację wyników z Twoim kontem.
os.environ['WANDB_API_KEY'] = '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'

# ==============================================================================
# 1. DEFINICJA ETYKIET (TWOJE "NAKLEJKI")
# ==============================================================================
# Standard BIO (Begin, Inside, Outside) pozwala na precyzyjne oznaczanie wielowyrazowych nazw.
label_list = ["O", "B-QTY", "B-ING", "I-ING"]

# Mapowania ID <-> Tekst (Model w środku operuje na liczbach 0-3).
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# ==============================================================================
# 2. PRZYGOTOWANIE DANYCH
# ==============================================================================
# Tworzymy mini-zbiór treningowy. W profesjonalnym projekcie wczytasz tu plik JSON/CSV.
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

# Dataset to wysoce zoptymalizowany format danych używany przez bibliotekę Transformers.
raw_dataset = Dataset.from_dict(data)

# ==============================================================================
# 3. TOKENIZACJA (TŁUMACZENIE TEKSTU NA LICZBY)
# ==============================================================================
model_checkpoint = "bert-base-cased"
# Ładujemy tokenizer BERT, który zamieni nasze słowa na wektory liczbowe.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



def tokenize_and_align_labels(examples):
    # Przygotowanie wejścia dla modelu (input_ids, attention_mask).
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True # Ważne: dane wejściowe są już listami słów.
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        # word_ids() mapuje każdy wygenerowany token z powrotem do indeksu oryginalnego słowa.
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Tokeny specjalne ([CLS], [SEP]) otrzymują -100, aby funkcja straty je pomijała.
            if word_idx is None:
                label_ids.append(-100)
            # Przypisujemy etykietę tylko dla pierwszego sub-tokena danego słowa.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Kolejne fragmenty tego samego słowa ignorujemy (-100).
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Uruchomienie mapowania tokenizacji na całym zbiorze danych.
tokenized_datasets = raw_dataset.map(tokenize_and_align_labels, batched=True)

# ==============================================================================
# 4. BUDOWA I KONFIGURACJA MODELU
# ==============================================================================
# Inicjalizujemy model BERT z warstwą klasyfikacyjną o rozmiarze 4 (liczba etykiet).
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)



# TrainingArguments definiuje "strategię bitwy" dla procesu nauki.
args = TrainingArguments(
    "recipe-ner-final-project",
    eval_strategy="no",
    learning_rate=2e-5, # Małe kroki czasowe zapobiegają gwałtownym zmianom wag (stabilność).
    num_train_epochs=15, # Powtarzamy proces 15 razy, aby model utrwalił wzorce.
    weight_decay=0.01,
    logging_steps=1,
    report_to="wandb"   # Włącza integrację z panelem Weights & Biases.
)

# Collator zajmuje się wyrównywaniem długości (padding) wewnątrz grup (batchy).
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer to główny silnik, który łączy model, dane i parametry treningowe.
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ==============================================================================
# 5. START NAUKI I TEST KOŃCOWY
# ==============================================================================
print("--- Start nauki i synchronizacji z W&B ---")
# Rozpoczynamy proces optymalizacji wag modelu.
trainer.train()

test_text = "Take 300g of sugar"
# Przenosimy dane testowe na to samo urządzenie, na którym pracuje model (CPU/GPU).
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

with torch.no_grad(): # Tryb ewaluacji (wyłączamy zapamiętywanie gradientów).
    logits = model(**inputs).logits

# Wybieramy indeks etykiety z najwyższym wynikiem (Argmax).
predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

print("\n--- Wynik testu ---")
print("-" * 40)
print(f"{'TOKEN':<12} | {'WYKRYTA KLASA'}")
print("-" * 40)
for token, pred in zip(tokens, predictions):
    # Pomijamy tokeny techniczne BERT-a przy wyświetlaniu wyniku.
    if token not in tokenizer.all_special_tokens:
        label = id2label[pred]
        print(f"{token:<12} | {label}")

"""
CZEGO SIĘ NAUCZYLIŚMY?
- Model nauczył się korelacji między liczbami/jednostkami a etykietą B-QTY.
- Dzięki Weights & Biases możesz teraz wejść na stronę wandb.ai i zobaczyć, 
  jak funkcja straty (Loss) spadała z każdą epoką.
- Skrypt jest gotowy do rozbudowy o tysiące nowych przykładów kulinarnych.
"""