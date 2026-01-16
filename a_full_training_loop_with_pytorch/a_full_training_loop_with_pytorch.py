import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
    DataCollatorWithPadding
)
from accelerate import Accelerator
from tqdm.auto import tqdm
import evaluate

# ==============================================================================
# 1. INICJALIZACJA I PRZYGOTOWANIE DANYCH
# ==============================================================================
checkpoint = "bert-base-uncased"
print(f"\n[1/7] Inicjalizacja komponentów dla: {checkpoint}...")

# Accelerator: Automatycznie zarządza sprzętem (CPU/GPU/TPU).
# Na Twoim Intel Ultra 7 przypisze obliczenia do procesora.
accelerator = Accelerator()
device = accelerator.device

# Ładowanie danych MRPC (czy zdania są parafrazami)
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# CZYSZCZENIE DANYCH: PyTorch akceptuje tylko liczby. Usuwamy tekst, zostawiamy tensory.
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Wybieramy małe próbki do testu na CPU
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
eval_dataset = tokenized_datasets["validation"].select(range(50))

# ==============================================================================
# 2. DATALOADERY (POMPY DANYCH - SZCZEGÓŁOWE WYJAŚNIENIE)
# ==============================================================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train_dataloader: To "taśmociąg", który dostarcza dane do modelu podczas nauki.
# - shuffle=True: Bardzo ważne! Miesza kolejność przykładów w każdej epoce.
#   Dzięki temu model nie uczy się kolejności pytań, tylko zasad języka.
# - batch_size=4: Model nie czyta 200 zdań naraz. Czyta je "kęsami" po 4 sztuki.
#   To pozwala oszczędzić pamięć RAM Twojego komputera.
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=4,
    collate_fn=data_collator
)

# eval_dataloader: Taśmociąg dla danych testowych (egzaminacyjnych).
# - Tutaj NIE mieszamy danych (shuffle=False domyślnie), bo kolejność przy
#   sprawdzaniu wyników nie wpływa na proces nauki, a ułatwia analizę błędów.
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=4,
    collate_fn=data_collator
)

# ==============================================================================
# 3. MODEL I TEST PRZED NAUKĄ (ANALIZA MATEMATYCZNA)
# ==============================================================================
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

z1, z2 = "Pawel is here", "Pawel is present"
inputs = tokenizer(z1, z2, return_tensors="pt").to(device)

# --- WYJAŚNIENIE BLOKU LOGITÓW I SOFTMAXU ---
# with torch.no_grad(): - Wyłączamy "tryb nagrywania" gradientów.
# Przy samej predykcji (zgadywaniu) model nie musi pamiętać ścieżki obliczeń.
# To drastycznie przyspiesza działanie i zużywa mniej pamięci.
with torch.no_grad():
    # model(**inputs) - Przekazujemy dane przez sieć neuronową.
    # .logits - Model zwraca surowe wyniki (punkty) dla każdej klasy (0 i 1).
    # Te liczby mogą być dowolne, np. [-2.1, 1.5]. Trudno je zrozumieć człowiekowi.
    logits_pre = model(**inputs).logits

    # F.softmax(logits_pre, dim=-1) - Magiczna funkcja matematyczna.
    # Bierze surowe logity (np. -2.1 i 1.5) i zamienia je na prawdopodobieństwo (0% - 100%).
    # Po Softmaxie suma wyników dla obu klas zawsze wynosi dokładnie 1 (czyli 100%).
    # dim=-1 oznacza, że liczymy to dla ostatniego wymiaru (czyli dla naszych klas).
    probs_pre = F.softmax(logits_pre, dim=-1)

# ==============================================================================
# 4. KONFIGURACJA ACCELERATE I SCHEDULERA (HARMONOGRAMU)
# ==============================================================================
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# --- WYJAŚNIENIE LR_SCHEDULER (HARMONOGRAMU UCZENIA) ---
# lr_scheduler: Kontroluje "współczynnik uczenia" (Learning Rate).
# - "linear": Oznacza, że zaczynamy od pełnej prędkości (5e-5), a z każdym krokiem
#   treningu model uczy się coraz wolniej i ostrożniej (aż do zera).
# - optimizer: Musi wiedzieć, czyim "tempem" steruje.
# - num_warmup_steps=0: Okres rozgrzewki. Gdyby wynosił np. 100, model zacząłby
#   bardzo powoli i przyspieszał przez pierwsze 100 kroków. Tu startujemy od razu.
# - num_training_steps: Harmonogram musi wiedzieć, jak długo trwa cały trening,
#   aby móc idealnie rozłożyć spadek prędkości w czasie.
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# ==============================================================================
# 5. PĘTLA TRENINGOWA
# ==============================================================================

print(f"\n[2/7] Start treningu PyTorch na {device}...")
progress_bar = tqdm(range(num_training_steps))

model.train()  # Aktywujemy tryb treningowy (ważne dla warstw takich jak Dropout)
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)  # Obliczamy gradienty (pochodne błędu)

        optimizer.step()  # Aktualizujemy wagi modelu na podstawie gradientów
        lr_scheduler.step()  # Informujemy harmonogram, że wykonaliśmy krok (zmniejsz LR)
        optimizer.zero_grad()  # Czyścimy "pamięć błędu" przed kolejną paczką danych
        progress_bar.update(1)

# ==============================================================================
# 6. EWALUACJA (SPRAWDZIAN KOŃCOWY)
# ==============================================================================
metric = evaluate.load("glue", "mrpc")
model.eval()  # Wyłączamy funkcje treningowe. Model ma teraz tylko stabilnie odpowiadać.

for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(f"👉 Wyniki końcowe: {metric.compute()}")

# ==============================================================================
# 7. TEST PO NAUCE (PORÓWNANIE I PRZENOSZENIE DANYCH)
# ==============================================================================
with torch.no_grad():
    # --- WYJAŚNIENIE PRZENOSZENIA DANYCH (TO DEVICE) ---
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # To jest krytyczne! W PyTorch model i dane MUSZĄ być na tym samym "urządzeniu".
    # Jeśli model jest na GPU, a dane na CPU (lub odwrotnie) - program się zawiesi.
    # Ta linia bierze nasz słownik 'inputs' (tekst zamieniony na liczby) i upewnia się,
    # że każda jego część (input_ids, attention_mask) jest tam, gdzie nasz model.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ponownie przepuszczamy te same zdania o Pawle przez model, który już się czegoś nauczył.
    logits_post = model(**inputs).logits

    # Ponownie zamieniamy logity (surowe punkty) na procenty (Softmax).
    # Teraz sprawdzimy, czy model jest bardziej pewny, że "here" i "present" to to samo.
    probs_post = F.softmax(logits_post, dim=-1)

print("\n--- PORÓWNANIE SYNONYM TEST ---")
print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Pewność podobieństwa PRZED nauką: {probs_pre[0][1].item():.2%}")
print(f"👉 Pewność podobieństwa PO nauce:    {probs_post[0][1].item():.2%}")

# Zapisujemy efekt naszej pracy
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./a_full_training_loop_with_pytorch/pytorch_model_custom")
tokenizer.save_pretrained("./a_full_training_loop_with_pytorch/pytorch_model_custom")
print("\n[7/7] Trening zakończony pomyślnie!")