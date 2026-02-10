import torch
import os
import numpy as np
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset

# Pobranie zasobów do tokenizacji zdań (wymagane dla metryki ROUGE w streszczaniu)
nltk.download("punkt", quiet=True)

"""
================================================================================
CO ROBI TEN SKRYPT? (SUMMARIZATION - ROZDZIAŁ 7.5)
================================================================================
1. ZADANIE STRESZCZANIA (Summarization):
   Model mT5 przyjmuje długą recenzję i generuje krótki tytuł.
   To zadanie typu tekst-na-tekst (Seq2Seq).

2. POPRAWKA mT5 (Sentinel Tokens):
   mT5 używa tokenów <extra_id_0> do uzupełniania luk. Aby generował tekst,
   musi zostać dostrojony na parach (tekst -> streszczenie).

3. ZGODNOŚĆ Z NAJNOWSZYMI BIBLIOTEKAMI:
   Używamy 'processing_class' zamiast 'tokenizer' w Trainerze, aby uniknąć 
   ostrzeżeń FutureWarning.
================================================================================
"""

# 1. PRZYKŁADOWY DATASET (Rozszerzony, aby model lepiej zrozumiał wzorzec)
data = {
    "review_body": [
        "Uwielbiam tę książkę! Przeczytałem ją w jeden wieczór. Fabuła o detektywie z Krakowa jest niesamowicie wciągająca.",
        "I found this book to be quite boring. The characters were flat and the plot was predictable.",
        "Książka kucharska z przepisami na domowy chleb. Bardzo jasne instrukcje i piękne zdjęcia.",
        "A comprehensive guide to Python programming. It covers everything from basics to advanced decorators.",
        "Aparat fotograficzny robi świetne zdjęcia w nocy, ale bateria trzyma bardzo krótko. Jakość obrazu ok.",
        "Ta powieść historyczna przenosi nas w czasy XVII-wiecznej Polski. Realizm i dbałość o szczegóły są niesamowite."
    ],
    "review_title": [
        "Wciągający kryminał",
        "Predictable and boring",
        "Świetne przepisy na chleb",
        "Essential Python guide",
        "Słaba bateria, dobre zdjęcia",
        "Świetna powieść historyczna"
    ]
}
raw_dataset = Dataset.from_dict(data)
# Dzielimy na train i test
dataset_dict = raw_dataset.train_test_split(test_size=0.2, seed=42)

# 2. KONFIGURACJA MODELU
# Ustawiamy legacy=False, aby korzystać z nowego zachowania tokenizera T5
model_checkpoint = "google/mt5-small"
print(f"Pobieranie modelu i tokenizera: {model_checkpoint}...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 3. PREPROCESSING (Zgodnie z Rozdziałem 7.5)
max_input_length = 128
max_target_length = 32


def preprocess_function(examples):
    # mT5 nie wymaga prefiksu, ale musimy zmapować wejście i wyjście
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True
    )

    # Przygotowanie etykiet (labels)
    labels = tokenizer(
        text_target=examples["review_title"],
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# 4. DATA COLLATOR (Dynamiczne dopełnianie)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5. ARGUMENTY TRENINGOWE
training_args = Seq2SeqTrainingArguments(
    output_dir="./streszczanie_model",
    eval_strategy="no",
    learning_rate=5e-4,  # Zwiększony LR dla bardzo małych zbiorów danych
    per_device_train_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=50,  # Więcej epok, aby model wyszedł z trybu "sentinel tokens"
    predict_with_generate=True,  # Włącza generowanie podczas ewaluacji
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# 6. TRAINER (Używamy processing_class zamiast tokenizer)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,  # Nowoczesny zamiennik dla 'tokenizer'
    data_collator=data_collator,
)

# 7. TRENING
print("\n--- Rozpoczynam trening (Summarization / Chapter 7.5) ---")
trainer.train()


# 8. TESTOWANIE MODELU (Inference)
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generowanie z użyciem beam search
    outputs = model.generate(
        **inputs,
        max_length=max_target_length,
        num_beams=5,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Weryfikacja na Twoim przykładzie
print("\n--- TEST GENEROWANIA STRESZCZENIA ---")
test_review = "Kupiłem ten aparat fotograficzny tydzień temu. Robi świetne zdjęcia w nocy, ale bateria trzyma bardzo krótko. Ogólnie jestem zadowolony z jakości obrazu."
print(f"Recenzja: {test_review}")
print(f"Wygenerowany tytuł: {summarize(test_review)}")

"""
PODSUMOWANIE ZMIAN:
1. legacy=False: Pozbywa się ostrzeżenia o starym zachowaniu T5.
2. num_train_epochs=50: Przy tak małej ilości danych model potrzebuje czasu, by zacząć generować tekst zamiast <extra_id>.
3. processing_class: Zgodność z Transformers v5.
4. preprocess_function: Usunięto sztywne dopełnianie (padding) wewnątrz funkcji map, zostawiając to kolatorowi.
"""