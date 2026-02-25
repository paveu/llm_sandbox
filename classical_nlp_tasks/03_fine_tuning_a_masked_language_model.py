import math
import collections
import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

"""
================================================================================
CO ROBI TEN SKRYPT? (PODSUMOWANIE ROZDZIAŁU 7.3)
================================================================================
1. ADAPTACJA DOMENOWA (MLM): 
   Zamiast uczyć model nowej czynności, uczymy go "nowego języka" (slangu filmowego). 
   Model DistilBERT, pierwotnie trenowany na Wikipedii, teraz poznaje specyfikę 
   opinii o kinie na zbiorze IMDb.

2. MECHANIZM MASKOWANIA (FILL-MASK):
   Model uczy się poprzez zgadywanie brakujących słów. Ukrywamy 15% tokenów, 
   a model musi wykorzystać kontekst otaczających słów, aby odgadnąć ukryte słowo.

3. PRZETWARZANIE STRUMIENIOWE (CHUNKING):
   W przeciwieństwie do klasyfikacji, tutaj łączymy wszystkie teksty w jeden 
   gigantyczny ciąg i tniemy go na równe bloki. Dzięki temu model widzi 
   maksymalną ilość kontekstu w każdym kroku.

================================================================================
CZEGO MOŻEMY SIĘ SPODZIEWAĆ?
================================================================================
- PERPLEKSJA (METRYKA): To główny wskaźnik. Jeśli perpleksja spada, oznacza to, 
  że model staje się "mniej zdziwiony" tekstami o filmach.
- ZMIANA KONTEKSTU: Przed treningiem model na maskę [MASK] w zdaniu o filmie 
  odpowie ogólnikami. Po treningu zacznie sugerować słowa typu "plot", "cinema", "acting".
- LOKALNOŚĆ I BEZPIECZEŃSTWO: Cały proces odbywa się lokalnie, co jest kluczowe 
  przy pracy z danymi wrażliwymi (choć IMDb jest publiczne).
================================================================================
"""

# Wyłączenie raportowania online (WandB itp.) dla czystości logów i prywatności
os.environ["HF_HUB_OFFLINE"] = "0"

# ==========================================
# 1. WYBÓR "MÓZGU" (MODELU) I TOKENIZERA
# ==========================================
# DistilBERT jest idealny: 40% mniejszy i 60% szybszy niż BERT,
# przy zachowaniu 97% jego możliwości.
model_checkpoint = "distilbert-base-uncased"
print(f"Ładowanie modelu MLM: {model_checkpoint}...")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# ==========================================
# 2. ŁADOWANIE DANYCH (IMDb)
# ==========================================
# Wybieramy zbiór recenzji filmowych, aby przystosować model do języka potocznego.
raw_datasets = load_dataset("imdb")

# ==========================================
# 3. PRZYGOTOWANIE TEKSTU (TOKENIZACJA)
# ==========================================
def tokenize_function(examples):
    # Zamieniamy tekst na format numeryczny (input_ids).
    result = tokenizer(examples["text"])
    # word_ids są kluczowe dla zaawansowanego maskowania (całych słów).
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Batched=True przyspiesza proces dzięki wielowątkowości.
print("Tokenizacja zbioru danych...")
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)

# ==========================================
# 4. ŁĄCZENIE I CIĘCIE (CHUNKING)
# ==========================================
# Modele Transformer najlepiej pracują na blokach o stałej długości (np. 128 tokenów).
chunk_size = 128

def group_texts(examples):
    # Łączymy wszystkie recenzje w jeden długi wektor tokenów.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Obcinamy końcówkę, która nie mieści się w pełnym bloku 128.
    total_length = (total_length // chunk_size) * chunk_size

    # Rozcinamy długi ciąg na paczki po 128 tokenów każda.
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # W MLM 'labels' to kopia wejścia - model uczy się odtwarzać to, co było pod maską.
    result["labels"] = result["input_ids"].copy()
    return result

print("Grupowanie tokenów w bloki (chunking)...")
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# ==========================================
# 5. MECHANIZM MASKOWANIA (DATA COLLATOR)
# ==========================================
# To serce MLM. Podczas każdej epoki collator losowo wybiera 15% tokenów
# i zastępuje je tokenem [MASK] lub innym słowem, wymuszając naukę kontekstu.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Ograniczamy dane, aby trening był błyskawiczny (do celów edukacyjnych).
train_size = 5000
test_size = 500

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# ==========================================
# 6. KONFIGURACJA TRENINGU
# ==========================================
training_args = TrainingArguments(
    output_dir="moj-model-mlm-filmowy",
    eval_strategy="epoch",      # Sprawdzamy postępy po każdej pełnej epoce
    learning_rate=2e-5,         # Klasyczne tempo uczenia dla fine-tuningu
    weight_decay=0.01,          # Regularyzacja zapobiegająca przeuczeniu
    num_train_epochs=1,         # W MLM często wystarczy jedna epoka do adaptacji domenowej
    logging_steps=100,          # Raportowanie postępu co 100 kroków
    save_strategy="no",         # Nie zaśmiecamy dysku zapisami pośrednimi
    report_to="none",           # Pełna prywatność logowania
    push_to_hub=False           # Model zostaje u Ciebie na dysku (RODO)
)

# Inicjalizacja Trenera MLM
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ==========================================
# 7. URUCHOMIENIE I ANALIZA WYNIKÓW
# ==========================================
# Perpleksja to matematyczna miara pewności modelu.
# Exp(loss) pozwala nam zrozumieć, jak "pogubiony" jest model.

print("\n--- KROK 1: Obliczam perpleksję bazową (przed nauką) ---")
eval_before = trainer.evaluate()
print(f"Perpleksja PRZED treningiem: {math.exp(eval_before['eval_loss']):.2f}")

print("\n--- KROK 2: Adaptacja domenowa (Trening na recenzjach IMDb) ---")
trainer.train()

print("\n--- KROK 3: Obliczam perpleksję końcową (po nauce) ---")
eval_after = trainer.evaluate()
print(f"Perpleksja PO treningu: {math.exp(eval_after['eval_loss']):.2f}")

"""
PODSUMOWANIE DZIAŁANIA:
Jeśli perpleksja spadła (np. z 20 na 15), oznacza to, że model lepiej "rozumie" 
świat recenzji filmowych. Teraz możesz użyć tego modelu jako fundamentu 
pod klasyfikator sentymentu, który będzie znacznie skuteczniejszy niż standardowy BERT.
"""