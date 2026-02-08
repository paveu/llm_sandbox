import math
import collections
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

# ==========================================
# 1. WYBÓR "MÓZGU" (MODELU) I TOKENIZERA
# ==========================================
# DistilBERT to lżejsza wersja BERTa.
# AutoModelForMaskedLM jest przygotowany do zadania uzupełniania luk ([MASK]).
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# ==========================================
# 2. ŁADOWANIE DANYCH (IMDb)
# ==========================================
# Ładujemy zbiór recenzji filmowych IMDb.
raw_datasets = load_dataset("imdb")


# ==========================================
# 3. PRZYGOTOWANIE TEKSTU (TOKENIZACJA)
# ==========================================
def tokenize_function(examples):
    # Zamieniamy tekst na liczby.
    result = tokenizer(examples["text"])
    # word_ids pomagają nam wiedzieć, które tokeny należą do tego samego słowa.
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Uruchamiamy tokenizację na całym zbiorze. Usuwamy kolumny tekstowe, bo mamy już liczby.
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)

# ==========================================
# 4. ŁĄCZENIE I CIĘCIE (CHUNKING)
# ==========================================
# Transformer lubi stałe rozmiary wejścia. Łączymy recenzje i tniemy na bloki po 128 tokenów.
chunk_size = 128


def group_texts(examples):
    # Sumujemy listy tokenów ze wszystkich przykładów w jeden długi ciąg.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Odcinamy resztę danych, która nie wypełni całego bloku 128-elementowego.
    total_length = (total_length // chunk_size) * chunk_size

    # Tworzymy listę bloków.
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # W zadaniu MLM uczymy model przewidywania oryginału, więc label = input.
    result["labels"] = result["input_ids"].copy()
    return result


# Mapujemy funkcję grupowania na nasze dane.
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# ==========================================
# 5. MECHANIZM MASKOWANIA (DATA COLLATOR)
# ==========================================
# To narzędzie w locie podczas treningu zamaskuje 15% słów.
# Dzięki temu model nie uczy się na pamięć, tylko próbuje zrozumieć sens.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Ograniczamy dane do testów, żebyś nie musiał czekać godzinami.
# Możesz zwiększyć te liczby (np. do 25000), jeśli masz mocną kartę graficzną (GPU).
train_size = 5000
test_size = 500

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# ==========================================
# 6. KONFIGURACJA TRENINGU (TU BYŁ FIX!)
# ==========================================
training_args = TrainingArguments(
    output_dir="moj-model-filmowy",
    eval_strategy="epoch",  # Poprawione: w nowszych wersjach 'eval_strategy' zastąpiło 'evaluation_strategy'
    learning_rate=2e-5,  # Prędkość nauki. 2e-5 to standard dla fine-tuningu.
    weight_decay=0.01,  # Pomaga uniknąć "przeuczenia" modelu.
    num_train_epochs=1,  # Ile razy model ma "przeczytać" cały zbiór treningowy.
    logging_steps=100,  # Co ile kroków wypisywać informację o stracie (loss).
    save_strategy="no",  # Pomijamy zapisywanie na dysku dla oszczędności miejsca.
    report_to="none"  # Wyłącza raportowanie do zewnętrznych narzędzi typu WandB.
)

# Inicjalizujemy Trenera.
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
print("\n--- KROK 1: Obliczam perpleksję bazową (przed nauką) ---")
eval_before = trainer.evaluate()
print(f"Perpleksja PRZED treningiem: {math.exp(eval_before['eval_loss']):.2f}")

print("\n--- KROK 2: Trenowanie modelu na recenzjach IMDb ---")
trainer.train()

print("\n--- KROK 3: Obliczam perpleksję końcową (po nauce) ---")
eval_after = trainer.evaluate()
print(f"Perpleksja PO treningu: {math.exp(eval_after['eval_loss']):.2f}")

# Zauważ: im niższa perpleksja, tym lepiej model radzi sobie z Twoją dziedziną (filmami)!