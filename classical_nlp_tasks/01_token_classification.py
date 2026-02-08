"""
================================================================================
PROJEKT: AUTONOMICZNY WYKRYWACZ INFORMACJI (NER)
================================================================================
"""

import numpy as np  # Narzędzia do obliczeń (np. znajdowanie największej liczby w wynikach)
import evaluate     # Biblioteka do oceny jakości modelu (F1-score, Accuracy)
import torch        # Silnik obliczeniowy dla sieci neuronowych (PyTorch)
import os           # Zarządzanie systemem operacyjnym
from datasets import load_dataset  # Narzędzie do pobierania baz danych z Hugging Face
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

"""
================================================================================
CO ROBI TEN SKRYPT? (KLASYFIKACJA TOKENÓW)
================================================================================
1. IDENTYFIKACJA JEDNOSTEK (NER): 
   Skrypt uczy model BERT rozpoznawania nazw własnych. Każdy "token" (kawałek słowa) 
   otrzymuje własną etykietę (np. "B-PER" dla początku nazwiska).

2. WYRÓWNYWANIE ETYKIET (ALIGNMENT): 
   BERT często tnie słowa na mniejsze kawałki (np. "Kowalski" -> "Kowal", "##ski"). 
   Skrypt zawiera logikę, która przypisuje etykietę do każdego z tych fragmentów, 
   aby model wiedział, że całe słowo tworzy jedną informację.

3. OCENA SEKWENCYJNA: 
   W przeciwieństwie do klasyfikacji całych zdań, tutaj oceniamy model za pomocą 
   metryki 'seqeval', która sprawdza, czy całe grupy słów (np. "New York") zostały 
   poprawnie wykryte jako jedna lokalizacja.

================================================================================
CZEGO MOŻEMY SIĘ SPODZIEWAĆ?
================================================================================
- PROCES UCZENIA: Model będzie analizował tysiące zdań ze zbioru CoNLL-2003. 
  W konsoli zobaczysz postęp (Train Loss) oraz jakość (F1 i Accuracy).
- INTELIGENCJA KONTEKSTOWA: Model po treningu nie będzie tylko "pamiętał" nazwisk, 
  ale nauczy się, że słowo po "Mr." lub przed "said" to prawdopodobnie osoba.
- PRECYZJA (F1-SCORE): Spodziewamy się wysokich wyników powyżej 90% dla standardowych 
  kategorii takich jak PER (osoby) czy LOC (lokalizacje).
================================================================================
"""

# Wyłączenie logowania online dla prywatności i przejrzystości konsoli
os.environ["HF_HUB_OFFLINE"] = "0"

# ==============================================================================
# ETAP 1: DOSTARCZENIE DANYCH (SUROWY MATERIAŁ)
# ==============================================================================
print("Krok 1: Wczytywanie bazy wiedzy CoNLL-2003...")
# Ładujemy zbiór. DatasetDict zawiera gotowe podziały na trening, walidację i testy.
raw_datasets = load_dataset("lhoestq/conll2003")

# Lista etykiet zgodna ze standardem IOB (Inside, Outside, Beginning).
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# ==============================================================================
# ETAP 2: PRZYGOTOWANIE NARZĘDZI (TOKENIZER)
# ==============================================================================
model_checkpoint = "bert-base-cased"
# BERT-base-cased rozróżnia wielkość liter, co jest kluczowe w NER (imiona piszemy z wielkiej litery).
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ==============================================================================
# ETAP 3: LOGIKA DOPASOWANIA (SERCE SKRYPTU)
# ==============================================================================
def align_labels_with_tokens(labels, word_ids):
    """
    Kluczowa funkcja: przypisuje etykiety do sub-tokenów.
    Jeśli słowo "Hugging" to B-ORG, a zostanie pocięte na "Hugg", "##ing",
    oba fragmenty muszą dostać odpowiednie ID kategorii.
    """
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id != current_word: # Nowe słowo
            current_word = word_id
            # Tokeny specjalne ([CLS], [SEP]) dostają -100 (ignorowane przez Loss)
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else: # Kontynuacja tego samego słowa
            label = labels[word_id]
            # Logika IOB: jeśli słowo to początek (B-), subword staje się kontynuacją (I-)
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

# ==============================================================================
# ETAP 4: MASOWA OBRÓBKA DANYCH
# ==============================================================================
def tokenize_and_align_labels(examples):
    # Tokenizacja tekstów pociętych już na słowa.
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    all_labels = examples["ner_tags"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i) # Pobranie mapy tokenów do słów
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

print("Krok 2: Przygotowanie danych (Tokenizacja i wyrównywanie etykiet)...")
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# ==============================================================================
# ETAP 5: KONFIGURACJA MODELU I METRYK
# ==============================================================================
# Mapowania cyfr na nazwy i odwrotnie (niezbędne do poprawnego wyświetlania wyników).
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

# Ładowanie modelu BERT z głowicą do klasyfikacji tokenów (9 klas wyjściowych).
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# Ładowanie seqeval – standardowej metryki dla zadań sekwencyjnych.
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Przygotowanie list do porównania (usunięcie tokenów specjalnych -100)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ==============================================================================
# ETAP 6: TRENING (NAUKA MODELU)
# ==============================================================================
args = TrainingArguments(
    "bert-finetuned-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    push_to_hub=False # Model zostaje lokalnie na dysku
)

# Collator wyrównuje długość zdań w batchach, dodając padding tam, gdzie to konieczne.
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Krok 3: Rozpoczynanie procesu uczenia...")
trainer.train()

# ==============================================================================
# ETAP 7: SPRAWDZIAN (INFERENCJA)
# ==============================================================================
print("\nKrok 4: Testowanie modelu na żywo...")
text = "Hugging Face is a company based in New York and Paris."

# Automatyczne wykrywanie dostępności karty graficznej (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenizacja i przesłanie na odpowiednie urządzenie
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    # Uzyskanie surowych wyników (logity)
    outputs = model(**inputs).logits

# Wybór najbardziej prawdopodobnych kategorii
predictions = outputs.argmax(-1).squeeze().tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

print("-" * 50)
print(f"{'TOKEN':<15} | {'KATEGORIA'}")
print("-" * 50)
for token, pred in zip(tokens, predictions):
    label = id2label[pred]
    if label != "O": # Pokazujemy tylko wykryte "ciekawe" informacje
        print(f"{token:<15} | {label}")

"""
PODSUMOWANIE DZIAŁANIA:
Model nie tylko nauczył się etykiet, ale rozumie strukturę języka. 
Nawet jeśli słowo "Hugging" zostanie pocięte, logika wyrównywania 
pozwala mu złożyć te kawałki w spójną informację o organizacji (ORG).
"""