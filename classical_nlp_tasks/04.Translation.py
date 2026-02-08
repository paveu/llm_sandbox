import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset

"""
================================================================================
CO ROBI TEN SKRYPT? (ZAKTUALIZOWANA WERSJA DLA TRANSFORMERS v5+)
================================================================================
1. ARCHITEKTURA ENCODER-DECODER (Seq2Seq):
   Model BART-base przyjmuje tekst wrażliwy i generuje bezpieczną parafrazę.
   Jest to proces "tłumaczenia" stylu i poziomu szczegółowości danych.

2. NOWA SKŁADNIA TOKENIZACJI:
   Zgodnie z ostrzeżeniem, zamiast 'as_target_tokenizer' używamy 'text_target'.
   Jest to cleaner (czystszy) sposób przekazywania etykiet do modelu.

3. NAPRAWA BŁĘDU EVAL_STRATEGY:
   Zmieniliśmy 'evaluation_strategy' na 'eval_strategy', co rozwiązuje błąd TypeError.
================================================================================
"""

# Wyłączenie logowania online dla stabilności i uniknięcia błędów 401
os.environ["HF_HUB_OFFLINE"] = "0"

# 1. PRZYKŁADOWY DATASET (Dane do lokalnego treningu - fundament RODO)
# Każda para to przykład "jak model ma myśleć". Uczymy go relacji
# między konkretnymi danymi a ich bezpiecznym uogólnieniem.
data = {
    "input_text": [
        "Pacjent Jan Kowalski, lat 45, zgłosił ból w klatce piersiowej.",
        "Mieszkaniec Warszawy, Adam Nowak, posiada zaległość w kwocie 500zł.",
        "Pani Maria Woźniak z ul. Polnej cierpi na nadciśnienie.",
        "Krzysztof Iksiński z Gdańska ma termin operacji na jutro."
    ],
    "target_text": [
        "Mężczyzna w średnim wieku zgłosił ból w klatce piersiowej.",
        "Osoba prywatna z dużego miasta posiada zaległość finansową.",
        "Pacjentka z grupy ryzyka cierpi na nadciśnienie.",
        "Pacjent z północy kraju ma zaplanowany zabieg medyczny."
    ]
}
raw_dataset = Dataset.from_dict(data)

# 2. KONFIGURACJA MODELU
# Wybieramy BART-base, ponieważ jest to model typu "Denoising Autoencoder",
# idealny do zadań, gdzie tekst wyjściowy jest modyfikacją tekstu wejściowego.
model_checkpoint = "facebook/bart-base"
print(f"Pobieranie modelu i tokenizera: {model_checkpoint}...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, token=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, token=False)

# 3. NOWOCZESNY PREPROCESSING (Bez deprecated as_target_tokenizer)
# Tutaj dzieje się przygotowanie danych zgodnie z Rozdziałem 7.4 kursu.
def preprocess_function(examples):
    # Tokenizacja wejścia (Input)
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    # Tokenizacja celu (Target) za pomocą argumentu text_target.
    # To przygotowuje labels w sposób zrozumiały dla Dekodera modelu Seq2Seq.
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Przetwarzanie datasetu (mapowanie funkcji na wszystkie przykłady)
tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# 4. DATA COLLATOR (Obsługa paddingu i 'decoder_input_ids')
# Kluczowy element: przygotowuje przesunięte etykiety (shift-right),
# co pozwala na naukę autoregresyjną (przewidywanie słowo po słowie).
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5. ARGUMENTY TRENINGOWE (Poprawione eval_strategy pod najnowsze biblioteki)
training_args = Seq2SeqTrainingArguments(
    output_dir="./anonimizacja_model",
    eval_strategy="no",             # Wyłączamy ewaluację w trakcie, bo mamy za mały zbiór
    learning_rate=3e-5,             # Subtelny krok uczenia dla precyzyjnego dostrajania
    per_device_train_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=20,            # Zwiększona liczba epok, by model "zauważył" wzorce RODO
    predict_with_generate=True,     # Umożliwia wywołanie metody generate()
    push_to_hub=False,              # Gwarancja lokalności danych (Zgodność z RODO)
    fp16=torch.cuda.is_available(), # Optymalizacja pod karty graficzne NVIDIA
    report_to="none"                # Całkowite odcięcie logowania zewnętrznego
)

# 6. TRAINER (Obiekt zarządzający pętlą treningową)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. TRENING LOKALNY
print("\n--- Rozpoczynam trening (Seq2Seq / Chapter 7.4) ---")
trainer.train()

# 8. TESTOWANIE MODELU (Inference / Przewidywanie)
def anonymize(text):
    # Przygotowanie danych i przesłanie na to samo urządzenie co model (CPU lub GPU)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # Metoda .generate() uruchamia proces tworzenia tekstu przez Dekoder
    # num_beams=5 sprawia, że model szuka najlepszego (niekoniecznie pierwszego) słowa
    outputs = model.generate(**inputs, max_length=64, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Weryfikacja na nowym przykładzie
print("\n--- TEST DZIAŁANIA PO TRENINGU ---")
test_input = "Tomasz Kot z Krakowa zgłosił zgubienie portfela."
print(f"Wejście: {test_input}")
print(f"Wyjście: {anonymize(test_input)}")

"""
PODSUMOWANIE TREŚCI KODU:
Skrypt przeszedł przez pełną ścieżkę opisaną w rozdziale o Tłumaczeniu:
1. Przygotowanie par danych.
2. Tokenizacja z uwzględnieniem targetu (text_target).
3. Konfiguracja Seq2SeqTrainer z parametrem predict_with_generate.
4. Generowanie tekstu metodą autoregresyjną (.generate()).
"""