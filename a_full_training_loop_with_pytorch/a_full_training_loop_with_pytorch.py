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
print(f"\n[1/7] KROK 1: Inicjalizacja komponentów...")
print(f"👉 Ładowanie modelu bazowego: {checkpoint}")

# Accelerator: Automatycznie zarządza sprzętem (CPU/GPU/TPU).
# Na Twoim Intel Ultra 7 przypisze obliczenia do procesora.
# To serce biblioteki 🤗 Accelerate, które dba o wydajność na Twoim sprzęcie.
accelerator = Accelerator()
device = accelerator.device
print(f"👉 Aktywne urządzenie (Device): {device}")

# Ładowanie danych MRPC (czy zdania są parafrazami)
# Dataset: Zbiór par zdań. LABEL (Etykieta) to wynik: 1 (parafraza), 0 (różne).
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    # Funkcja mapująca: zamieniamy tekst na liczby zrozumiałe dla BERT-a.
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

print("👉 Rozpoczynam tokenizację (zamiana tekstu na wektory liczbowe)...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# CZYSZCZENIE DANYCH: PyTorch akceptuje tylko liczby. Usuwamy tekst, zostawiamy tensory.
# W czystym PyTorch (w przeciwieństwie do Trainera) musimy to zrobić ręcznie,
# inaczej model "pogubi się" próbując przetwarzać napisy.
print("👉 Czyszczenie kolumn i ustawianie formatu tensora...")
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Wybieramy małe próbki do testu na CPU (dla szybkości treningu na laptopie)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
eval_dataset = tokenized_datasets["validation"].select(range(50))
print(f"👉 Gotowe! Rozmiar zbioru treningowego: {len(train_dataset)}, walidacyjnego: {len(eval_dataset)}")

# ==============================================================================
# 2. DATALOADERY (POMPY DANYCH - SZCZEGÓŁOWE WYJAŚNIENIE)
# ==============================================================================
# DataCollator: Odpowiada za dynamiczne wyrównywanie długości zdań w paczkach.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train_dataloader: To "taśmociąg", który dostarcza dane do modelu podczas nauki.
# - shuffle=True: Bardzo ważne! Miesza kolejność przykładów w każdej epoce.
#   Dzięki temu model nie uczy się kolejności pytań, tylko zasad języka.
# - batch_size=4: Model nie czyta 200 zdań naraz. Czyta je "kęsami" po 4 sztuki.
#   To pozwala oszczędzić pamięć RAM Twojego komputera.
# WYJAŚNIENIE: DataLoader zamienia Twój zestaw danych w iterator, który
# automatycznie tworzy paczki (batches) i nakłada na nie padding.
print("\n[2/7] KROK 2: Przygotowanie Dataloaderów (taśmociągów danych)...")
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
print("\n[3/7] KROK 3: Ładowanie modelu i analiza przed treningiem...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# AdamW: Optymalizator z poprawką na zanikanie wag (weight decay).
optimizer = AdamW(model.parameters(), lr=5e-5)

z1, z2 = "Pawel is here", "Pawel is present"
# Zamieniamy nasze zdania testowe na format modelu i wysyłamy na CPU/GPU.
inputs = tokenizer(z1, z2, return_tensors="pt").to(device)

# --- WYJAŚNIENIE BLOKU LOGITÓW I SOFTMAXU ---
# with torch.no_grad(): - Wyłączamy "tryb nagrywania" gradientów.
# Przy samej predykcji (zgadywaniu) model nie musi pamiętać ścieżki obliczeń.
# To drastycznie przyspiesza działanie i zużywa mniej pamięci.
with torch.no_grad():
    # model(**inputs) - Przekazujemy dane przez sieć neuronową.
    # .logits - Model zwraca surowe wyniki (punkty) dla każdej klasy (0 i 1).
    # Te liczby mogą być dowolne, np. [-2.1, 1.5]. Trudno je zrozumieć człowiekowi.
    # LOGITY to surowy output ostatniej warstwy liniowej przed jakąkolwiek normalizacją.
    logits_pre = model(**inputs).logits

    # F.softmax(logits_pre, dim=-1) - Magiczna funkcja matematyczna.
    # Bierze surowe logity (np. -2.1 i 1.5) i zamienia je na prawdopodobieństwo (0% - 100%).
    # Po Softmaxie suma wyników dla obu klas zawsze wynosi dokładnie 1 (czyli 100%).
    # dim=-1 oznacza, że liczymy to dla ostatniego wymiaru (czyli dla naszych klas).
    # SOFTMAX pozwala nam zinterpretować wynik jako "pewność modelu".
    probs_pre = F.softmax(logits_pre, dim=-1)

print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Pewność przed nauką (Softmax): {probs_pre[0][1].item():.2%}")

# ==============================================================================
# 4. KONFIGURACJA ACCELERATE I HARMONOGRAMU (SCHEDULER)
# ==============================================================================
print("\n[4/7] KROK 4: Konfiguracja Accelerate i Schedulera...")

# prepare(): To tutaj Accelerate przejmuje kontrolę nad obiektami.
# Dataloadery zostaną zoptymalizowane pod kątem Twojego procesora.
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# --- WYJAŚNIENIE LR_SCHEDULER (HARMONOGRAMU UCZENIA) ---
# lr_scheduler: Kontroluje "współczynnik uczenia" (Learning Rate).
# - "linear": Oznacza, że zaczynamy od pełnej prędkości (5e-5), a z każdym krokiem
#   treningu model uczy się coraz wolniej i ostrożniej (aż do zera).
# - optimizer: Musi wiedzieć, czyim "tempem" steruje (optymalizator trzyma wagi).
# - num_warmup_steps=0: Okres rozgrzewki. Gdyby wynosił np. 100, model zacząłby
#   bardzo powoli i przyspieszał przez pierwsze 100 kroków. Tu startujemy od razu.
# - num_training_steps: Harmonogram musi wiedzieć, jak długo trwa cały trening,
#   aby móc idealnie rozłożyć spadek prędkości (tzw. decay) w czasie.
# WYJAŚNIENIE: Scheduler zapobiega "przestrzeleniu" celu (overshooting) pod koniec treningu.
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# ==============================================================================
# 5. PĘTLA TRENINGOWA
# ==============================================================================

print(f"\n[5/7] KROK 5: Rozpoczynam pętlę treningową (Manual Training Loop)...")
progress_bar = tqdm(range(num_training_steps))

model.train()  # Aktywujemy tryb treningowy (ważne dla warstw takich jak Dropout i BatchNorm)
for epoch in range(num_epochs):
    print(f"\n--- Epoka {epoch + 1} / {num_epochs} ---")
    for step, batch in enumerate(train_dataloader):
        # Forward pass: Model przewiduje wyniki dla aktualnej paczki (batch).
        outputs = model(**batch)

        # Loss: Obliczamy matematyczną karę za błędy modelu.
        loss = outputs.loss

        # Backward pass: Obliczamy gradienty (pochodne błędu).
        # accelerator.backward() zastępuje standardowe loss.backward() w PyTorch.
        accelerator.backward(loss)

        # Aktualizacja wag: Poprawiamy "pokrętła" modelu na podstawie obliczonych gradientów.
        optimizer.step()

        # Aktualizacja tempa nauki: Scheduler obniża learning rate zgodnie z planem liniowym.
        lr_scheduler.step()

        # Wyzerowanie gradientów: Czyścimy pamięć błędu przed kolejną paczką.
        # W PyTorch gradienty się sumują (akumulują), więc musimy je ręcznie czyścić!
        optimizer.zero_grad()

        progress_bar.update(1)

# ==============================================================================
# 6. EWALUACJA (SPRAWDZIAN KOŃCOWY)
# ==============================================================================
print("\n[6/7] KROK 6: Rozpoczynam sprawdzian modelu (Ewaluacja)...")
metric = evaluate.load("glue", "mrpc")
model.eval()  # Wyłączamy funkcje treningowe. Model ma teraz tylko stabilnie odpowiadać.

for batch in eval_dataloader:
    # Podczas ewaluacji nigdy nie liczymy gradientów (oszczędność czasu i energii CPU).
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    # argmax: Wybieramy indeks (0 lub 1), który dostał najwięcej punktów (najwyższy logit).
    predictions = torch.argmax(logits, dim=-1)

    # Przekazujemy wyniki paczki do globalnego licznika metryk.
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(f"👉 WYNIKI KOŃCOWE METRYKI: {metric.compute()}")

# ==============================================================================
# 7. TEST PO NAUCE (PORÓWNANIE I PRZENOSZENIE DANYCH)
# ==============================================================================
print("\n[7/7] KROK 7: Końcowy test praktyczny i zapisywanie modelu...")

with torch.no_grad():
    # --- WYJAŚNIENIE PRZENOSZENIA DANYCH (TO DEVICE) ---
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # To jest krytyczne! W PyTorch model i dane MUSZĄ być na tym samym "urządzeniu".
    # Jeśli model jest na GPU, a dane na CPU (lub odwrotnie) - program się zawiesi.
    # Ta linia bierze nasz słownik 'inputs' (tekst zamieniony na liczby) i upewnia się,
    # że każda jego część (input_ids, attention_mask) jest tam, gdzie nasz model.
    # WYJAŚNIENIE: Ponieważ nasz model przeszedł przez accelerator.prepare(),
    # może znajdować się na specyficznym urządzeniu. Dane testowe muszą tam "dołączyć".
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ponownie przepuszczamy te same zdania o Pawle przez model, który już się czegoś nauczył.
    # .logits pobiera surowe wyniki z modelu po procesie fine-tuningu.
    logits_post = model(**inputs).logits

    # Ponownie zamieniamy logity (surowe punkty) na procenty (Softmax).
    # Teraz sprawdzimy, czy model jest bardziej pewny, że "here" i "present" to to samo.
    # F.softmax wykonuje operację: e^xi / suma(e^xj).
    probs_post = F.softmax(logits_post, dim=-1)

print("\n--- ANALIZA PORÓWNAWCZA (TEST SYNONIMÓW) ---")
print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Pewność podobieństwa PRZED nauką: {probs_pre[0][1].item():.2%}")
print(f"👉 Pewność podobieństwa PO nauce:    {probs_post[0][1].item():.2%}")

# Zapisujemy efekt naszej pracy:
# unwrap_model: Wyciąga czysty model PyTorch z "opakowania" Accelerate.
# Jest to niezbędne, aby zapisać pliki w standardowym formacie Transformers.
unwrapped_model = accelerator.unwrap_model(model)
path = "./pytorch_model_custom"
unwrapped_model.save_pretrained(path)
tokenizer.save_pretrained(path)
print(f"\n✅ Trening zakończony! Model zapisany w folderze: {path}")