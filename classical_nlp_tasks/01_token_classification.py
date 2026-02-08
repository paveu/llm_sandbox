"""
================================================================================
PROJEKT: AUTONOMICZNY WYKRYWACZ INFORMACJI (NER)
================================================================================
"""

import numpy as np  # Narzędzia do obliczeń (np. znajdowanie największej liczby)
import evaluate     # Biblioteka do oceny jakości modelu
import torch        # Silnik obliczeniowy dla sieci neuronowych
from datasets import load_dataset  # Narzędzie do pobierania baz danych
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

# ==============================================================================
# ETAP 1: DOSTARCZENIE DANYCH (SUROWY MATERIAŁ)
# ==============================================================================
print("Krok 1: Wczytywanie bazy wiedzy...")
# Ładujemy zbiór.
# ZWRACA: DatasetDict {train: 14041 rows, validation: 3250 rows, test: 3453 rows}
raw_datasets = load_dataset("lhoestq/conll2003")

# Lista etykiet.
# ZWRACA: ['O', 'B-PER', 'I-PER', ...] (Lista stringów, gdzie indeks to ID kategorii)
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# ==============================================================================
# ETAP 2: PRZYGOTOWANIE NARZĘDZI (TOKENIZER)
# ==============================================================================
model_checkpoint = "bert-base-cased"
# Ładujemy tokenizer.
# ZWRACA: Obiekt BertTokenizerFast (narzędzie zamieniające tekst na liczby)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ==============================================================================
# ETAP 3: LOGIKA DOPASOWANIA (SERCE SKRYPTU)
# ==============================================================================
def align_labels_with_tokens(labels, word_ids):
    """
    labels: [3, 0] (Lista ID kategorii dla słów)
    word_ids: [None, 0, 1, 1, None] (Mapa tokenów do słów)
    ZWRACA: [-100, 3, 0, 0, -100] (Lista etykiet dopasowana do długości tokenów)
    """
    new_labels = []  # Inicjalizacja pustej listy na przetworzone etykiety
    current_word = None  # Zmienna pomocnicza do śledzenia ID słowa

    for word_id in word_ids:  # Iteracja po każdym ID tokena
        if word_id != current_word:  # Sprawdzamy, czy zaczęliśmy nowe słowo
            current_word = word_id  # Aktualizujemy ID obecnego słowa
            # ZWRACA: Liczbę (oryginalne ID etykiety) lub -100 dla tokenów [CLS]/[SEP]
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:  # Obsługa ewentualnych pustych tokenów
            new_labels.append(-100)
        else:  # To jest subword tego samego słowa (np. "##ing" w "Hugging")
            label = labels[word_id]
            # Logika IOB: Jeśli słowo to B-ORG (3), subword staje się I-ORG (4)
            # ZWRACA: Liczbę (parzyste ID dla kontynuacji jednostki)
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

# ==============================================================================
# ETAP 4: MASOWA OBRÓBKA DANYCH
# ==============================================================================
def tokenize_and_align_labels(examples):
    # examples["tokens"]: [['EU', 'rejects', 'German', ...]]
    # ZWRACA: Słownik z listami: {'input_ids': [...], 'attention_mask': [...]}
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True, # Skraca tekst do max 512 tokenów
        is_split_into_words=True # Informacja, że tekst to już lista słów
    )

    all_labels = examples["ner_tags"] # Pobiera listę etykiet z surowych danych
    new_labels = [] # Lista na nowe etykiety dla całej paczki (batch)

    for i, labels in enumerate(all_labels): # Iteracja po każdym zdaniu w paczce
        # ZWRACA: Listę word_ids dla konkretnego zdania, np. [None, 0, 1, 2, None]
        word_ids = tokenized_inputs.word_ids(i)
        # ZWRACA: Listę etykiet wyrównaną do tokenów (wywołanie funkcji z Etapu 3)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    # ZWRACA: Zmodyfikowany słownik tokenized_inputs z dodanym kluczem 'labels'
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

print("Krok 2: Przygotowanie danych do nauki...")
# map() stosuje funkcję do całego zbioru.
# ZWRACA: Dataset (Przetworzona tabela z liczbami zamiast tekstu)
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True, # Szybsze przetwarzanie w grupach
    remove_columns=raw_datasets["train"].column_names, # Usuwa tekst 'tokens', 'ner_tags'
)

# ==============================================================================
# ETAP 5: KONFIGURACJA MODELU I METRYK
# ==============================================================================
# Tworzenie mapowań.
# ZWRACA: Słownik {0: 'O', 1: 'B-PER', ...} i odwrotny
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

# Ładowanie modelu z odpowiednią liczbą wyjść (9 klas).
# ZWRACA: Obiekt BertForTokenClassification (Sieć neuronowa gotowa do nauki)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# Ładowanie metryki seqeval.
# ZWRACA: Obiekt Metric (narzędzie do liczenia F1-score)
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds # Logits: surowe wyniki modelu, Labels: prawda
    # ZWRACA: Macierz Numpy z wybranymi najpewniejszymi indeksami klas
    predictions = np.argmax(logits, axis=-1)

    # Filtracja -100: zamieniamy ID na nazwy ('B-ORG') tylko tam, gdzie l != -100
    # ZWRACA: Listę list stringów [['O', 'B-ORG'], ...]
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Obliczanie wyników.
    # ZWRACA: Słownik, np. {'overall_f1': 0.85, 'overall_accuracy': 0.98, ...}
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ==============================================================================
# ETAP 6: TRENING (NAUKA MODELU)
# ==============================================================================
# Ustawienia techniczne.
# ZWRACA: Obiekt TrainingArguments (zbiór parametrów sterujących GPU/pamięcią)
args = TrainingArguments(
    "bert-finetuned-ner", # Nazwa folderu na dysku
    eval_strategy="epoch", # Licz błędy po każdej epoce
    save_strategy="epoch", # Zapisuj postęp po każdej epoce
    learning_rate=2e-5, # Prędkość nauki (bardzo powolna dla stabilności)
    num_train_epochs=3, # Ile razy model zobaczy wszystkie dane
    weight_decay=0.01, # Ochrona przed "wykuciem danych na pamięć"
    report_to="none" # Nie wysyłaj logów do zewnętrznych serwisów
)

# Collator dba o równe długości w paczkach.
# ZWRACA: Funkcję, która paduje (uzupełnia zerami) dane w locie
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Zarządca treningu.
# ZWRACA: Obiekt Trainer (Silnik łączący dane, model i optymalizator)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Krok 3: Rozpoczynanie nauki...")
# Uruchomienie pętli uczenia.
# ZWRACA: Obiekt TrainOutput (statystyki z całego procesu treningu)
trainer.train()

# ==============================================================================
# ETAP 7: SPRAWDZIAN (INFERENCJA)
# ==============================================================================
print("\nKrok 4: Testowanie modelu na żywo...")
text = "Hugging Face is a company based in New York and Paris."

# ZWRACA: Słowo 'cuda' (jeśli masz kartę NVIDIA) lub 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Przeniesienie wag modelu do pamięci GPU/CPU

# Tokenizacja tekstu testowego.
# ZWRACA: PyTorch Tensor {'input_ids': tensor([[101, 2341, ...]]), ...}
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad(): # Wyłączenie śledzenia gradientów (oszczędność pamięci)
    # ZWRACA: TokenClassifierOutput (obiekt zawierający macierz logitów)
    outputs = model(**inputs).logits

# ZWRACA: Listę liczb (wybrane ID klas dla każdego tokena w zdaniu)
predictions = outputs.argmax(-1).squeeze().tolist()
# ZWRACA: Listę stringów (tokeny, np. ['[CLS]', 'Hugg', '##ing', ...])
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

# Wyświetlanie wyników
for token, pred in zip(tokens, predictions):
    label = id2label[pred]
    if label != "O": # Pomiń tło, pokaż tylko wykryte jednostki
        print(f"Znalazłem: {token:12} -> Kategoria: {label}")