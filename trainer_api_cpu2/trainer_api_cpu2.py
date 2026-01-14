import numpy as np
import evaluate
import torch
import torch.nn.functional as F  # Potrzebne do funkcji Softmax (zamiana surowych wyników na %)
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ==============================================================================
# 1. PRZYGOTOWANIE DANYCH I POJĘĆ (FILOZOFIA PRE-TRAININGU)
# ==============================================================================
checkpoint = "bert-base-uncased"
print(f"\n[1/6] Pobieranie modelu i danych: {checkpoint}...")

# DATASET: Zbiór par zdań. LABEL (Etykieta) to wynik: 1 (parafraza), 0 (różne).
# Oryginalny BERT od Google uczył się na 3.3 mld słów (Wikipedia + Książki).
# On już "rozumie" język, ale nie wie jeszcze, co to jest zadanie "MRPC".
raw_datasets = load_dataset("glue", "mrpc")

# TOKENIZER: Pobiera "Słownik" (Vocab) przypisany do modelu.
# Zamienia słowa na numery ID. Skąd wie jakie numery? Każdy model ma swój
# unikalny plik .txt ze słownikiem. Tutaj tekst staje się matematyką.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    # ATTENTION HEADS (Głowy Uwagi): Wewnątrz modelu jest 12 warstw, a każda ma 12 głów.
    # Razem 144 "mikro-mózgi", które analizują tekst pod różnymi kątami.
    # Łączymy dwa zdania. Tokenizer doda [CLS] na początku i [SEP] między zdaniami.
    # Truncation=True obcina zbyt długie zdania do limitu modelu (np. 512 tokenów).
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


print("\n[2/6] Tokenizacja (zamiana słów na numery ID)...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# EPOKA (Epoch): Przeczytanie całej "książki" (zbioru danych) jeden raz.
# Epoch 1.0 oznacza, że model przejrzał wszystkie przykłady dokładnie raz.

# Zbiór TRENINGOWY (train): To nasze "zadania domowe". Tu model się uczy.
# Wybieramy 200 przykładów do nauki. shuffle(seed=42) miesza dane tak samo za każdym razem.
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(200))

# Zbiór WALIDACYJNY (validation): To nasza "próbna matura".
# Model nie uczy się na tych danych – sprawdzamy tu, czy model faktycznie rozumie,
# czy tylko wykuł przykłady na pamięć (tzw. overfitting).
tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(50))

# DATA COLLATOR: Wyrównuje długość zdań w paczce (batchu) dodając zera (padding).
# Modele wymagają, aby dane w jednej paczce (batch) miały identyczny wymiar.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==============================================================================
# 2. MODEL I WAGI (DODAWANIE NOWEJ GŁOWICY)
# ==============================================================================
print("\n[3/6] Ładowanie modelu i instalacja nowej 'Głowicy' klasyfikatora...")
# KLUCZOWY MOMENT: Odcinamy oryginalną głowę BERT-a (do przewidywania słów)
# i "przyszywamy" nową, klasyfikacyjną głowę z 2 wyjściami (TAK/NIE).
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# WAGI (Weights): To miliony "pokręteł" (liczb) wewnątrz modelu. Trening to kręcenie nimi.
# Każda waga decyduje, jak mocno dany sygnał wpływa na wynik końcowy.
# Wagi w "mózgu" są ustawione przez Google, ale w nowej głowie są na razie LOSOWE.
weights_before = model.classifier.weight.data[0][:5].clone()
print(f"👉 Wagi nowej głowy PRZED treningiem (losowe): {weights_before}")

# ==============================================================================
# 3. TEST MODELU PRZED TRENINGIEM (Zgadujemy!)
# ==============================================================================
print("\n[3.5] TEST PRZED TRENINGIEM (Logity i Softmax):")
# Wyzwanie dla synonimów: czy model "poczuje" podobieństwo bez nauki?
z1 = "Pawel is here"
z2 = "Pawel is present"

# Przygotowanie danych do ręcznego testu matematycznego
inputs = tokenizer(z1, z2, return_tensors="pt")

# torch.inference_mode() – Jeszcze szybszy tryb "tylko do odczytu" niż no_grad.
# Całkowicie izoluje model od mechanizmów treningowych dla maksymalnej wydajności CPU.

with torch.inference_mode():
    outputs_pre = model(**inputs)
    # LOGITY: Surowe punkty z modelu (np. [-1.2, 0.5]). Nie sumują się do 100%.
    logits_pre = outputs_pre.logits

# SOFTMAX: Funkcja, która zamienia surowe punkty (logity) na procenty (0-100%).
probs_pre = F.softmax(logits_pre, dim=-1)
print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Surowe logity PRZED nauką: {logits_pre}")
print(f"👉 Pewność PRZED nauką (Softmax): {probs_pre[0][1].item():.2%}")


# ==============================================================================
# 4. METRYKI, GRADIENTY I LOSS (Zasady oceniania)
# ==============================================================================
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds

    # LOSS (Strata): Matematyczna miara błędu. Jeśli spada, model lepiej rozumie dane.
    # GRADIENT: Instrukcja, w którą stronę kręcić wagą, aby LOSS malał.
    # GRAD_NORM: Siła tej instrukcji (im większy, tym gwałtowniejsza zmiana wag).

    # PREDICTIONS: To ostateczny "strzał" modelu (odpowiedź na egzaminie).
    # Wybieramy indeks (0 lub 1), który otrzymał najwięcej punktów w logitach.
    predictions = np.argmax(logits, axis=-1)

    # LABELS: To "klucz odpowiedzi" (prawdziwe etykiety ze zbioru danych).
    # Nauczyciel (metric) porównuje predictions z labels.
    return metric.compute(predictions=predictions, references=labels)


# ==============================================================================
# 5. KONFIGURACJA TRENINGU (Zoptymalizowana pod Intel Ultra 7)
# ==============================================================================
training_args = TrainingArguments(
    output_dir="./test-trainer-cpu",
    # Używamy CPU, bo GPU zawiesza laptopa przy obliczeniach AI.
    use_cpu=True,
    eval_strategy="epoch",  # Sprawdzian (eval) po każdej pełnej epoce.
    num_train_epochs=3,  # Model przeczyta 200 zdań 3 razy (lepsza stabilność).
    learning_rate=2e-5,  # "Długość kroku" (jak mocno gradient zmienia wagi).
    per_device_train_batch_size=4,  # Wykorzystujemy 14 rdzeni Twojego procesora.
    weight_decay=0.01,  # "Hamulec": zapobiega przypisywaniu ogromnych wag słowom.
    logging_steps=5,  # Co 5 paczek wypisz stan w konsoli.
)

# ==============================================================================
# 6. TWORZENIE TRAINERA (DYRYGENT PROCESU)
# ==============================================================================
# Trainer łączy model, dane, parametry i metryki w jedną maszynę treningową.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ==============================================================================
# 7. TRENING I ANALIZA ZMIAN W "MÓZGU"
# ==============================================================================
print("\n[4/6] Start Fine-tuningu (Trening nowej głowy na Intel Ultra 7)...")
# LOSS (Strata) powinna spadać z każdym krokiem.
trainer.train()

print("\n[5/6] Sprawdzanie zmian w 'mózgu' modelu...")
weights_after = model.classifier.weight.data[0][:5].clone()
print(f"👉 Wagi przed: {weights_before}")
print(f"👉 Wagi po:    {weights_after}")

# RÓŻNICA: Pokazuje o ile fizycznie przesunęły się wagi pod wpływem uczenia.
diff = weights_after - weights_before
print(f"👉 Różnica (fizyczny efekt nauki): {diff}")

# ==============================================================================
# 8. TEST PRAKTYCZNY PO TRENINGU (SYNONYM TEST)
# ==============================================================================
print("\n[6/6] TEST PO TRENINGU (Analiza synonimów):")
# Ponownie używamy inference_mode dla najszybszego sprawdzenia wyniku.
with torch.inference_mode():
    outputs_post = model(**inputs)
    # Ponownie zamieniamy logity na % po treningu
    probs_post = F.softmax(outputs_post.logits, dim=-1)
    confidence = probs_post[0][1].item()

print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Wynik PRZED nauką: {probs_pre[0][1].item():.2%}")
print(f"👉 Wynik PO nauce:    {confidence:.2%}")

# --- ANALIZA TECHNICZNA DLA KOLEGI (WHY THIS MATTERS) ---
# 1. SEMANTIC SIMILARITY: Model musiał wykazać się wiedzą z pre-trainingu,
#    żeby wiedzieć, że "here" i "present" to w tym kontekście synonimy.
#
# 2. ROLE OF ATTENTION: Twoje 144 głowy uwagi analizują kontekst słowa "Pawel".
#    W obu zdaniach Pawel pełni tę samą rolę (podmiot), co pomaga modelowi.
#
# 3. CPU EFFICIENCY: Trening trwał krótko, bo Twój Ultra 7 świetnie
#    radzi sobie z matematyką macierzową dzięki instrukcjom AVX/AMX.

# ==============================================================================
# 9. ZAPISYWANIE MODELU NA DYSKU
# ==============================================================================
# Zapisujemy wagi modelu i słownik tokenizera do folderu.
trainer.save_model("./moj_model_synonimy")
tokenizer.save_pretrained("./moj_model_synonimy")
print("\nModel zapisany w './moj_model_synonimy'!")