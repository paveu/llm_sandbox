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
# Głównym zadaniem Accelerate jest umożliwienie treningu rozproszonego na wielu GPU/TPU przy minimalnych zmianach w kodzie.
accelerator = Accelerator()
device = accelerator.device
print(f"👉 Aktywne urządzenie (Device): {device}")

# Ładowanie danych MRPC (czy zdania są parafrazami)
# Dataset: Zbiór par zdań. LABEL (Etykieta) to wynik: 1 (parafraza), 0 (różne).
# Drugi argument w load_dataset (np. 'mrpc') określa konkretne zadanie lub podzbiór (subset) w ramach danego benchmarku GLUE.
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    # Funkcja mapująca: zamieniamy tekst na liczby zrozumiałe dla BERT-a.
    # Użycie batched=True w metodzie .map() przetwarza wiele przykładów naraz, co radykalnie przyspiesza proces tokenizacji.
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

print("👉 Rozpoczynam tokenizację (zamiana tekstu na wektory liczbowe)...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# CZYSZCZENIE DANYCH: PyTorch akceptuje tylko liczby. Usuwamy tekst, zostawiamy tensory.
# W czystym PyTorch (w przeciwieństwie do Trainera) musimy to zrobić ręcznie,
# inaczej model "pogubi się" próbując przetwarzać napisy.
# Usuwamy kolumny surowego tekstu, bo model oczekuje tensorów liczbowych; próba ich zachowania mogłaby spowodować błędy.
print("👉 Czyszczenie kolumn i ustawianie formatu tensora...")
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# --- KLUCZOWY MOMENT KONWERSJI ---
# Domyślnie 'datasets' zwraca listy Pythona. Model BERT (PyTorch) wymaga jednak
# obiektów typu torch.Tensor do obliczeń macierzowych. Poniższa linia
# automatycznie "opakowuje" dane w tensory, co pozwala na ich bezpośrednie
# przesyłanie do modelu i na kartę graficzną/procesor.
tokenized_datasets.set_format("torch")

# Wybieramy małe próbki do testu na CPU (dla szybkości treningu na laptopie)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
eval_dataset = tokenized_datasets["validation"].select(range(50))
print(f"👉 Gotowe! Rozmiar zbioru treningowego: {len(train_dataset)}, walidacyjnego: {len(eval_dataset)}")

# ==============================================================================
# 2. DATALOADERY (POMPY DANYCH - SZCZEGÓŁOWE WYJAŚNIENIE)
# ==============================================================================
# DataCollator: Odpowiada za dynamiczne wyrównywanie długości zdań w paczkach.
# Dynamiczne dopełnianie (Dynamic Padding) jest wydajniejsze niż stałe, bo ogranicza rozmiar do najdłuższego zdania TYLKO w danej partii (batch).
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
# --- ŁADOWANIE ARCHITEKTURY I WAG ---
# AutoModelForSequenceClassification: Pobiera architekturę BERT-a i automatycznie
# dodaje na jej szczycie "głowicę klasyfikacyjną" (warstwę Linear).
# - checkpoint: Wczytuje wyuczone już wagi języka (wiedza o gramatyce i znaczeniu słów).
# - num_labels=2: Mówi modelowi, że na końcu ma mieć 2 wyjścia (w tym przypadku:
#   0 - to nie parafraza, 1 - to parafraza).
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# --- SILNIK UCZENIA (OPTYMALIZATOR) ---
# AdamW: Wyrafinowana wersja algorytmu spadku gradientu. To on decyduje,
# jak mocno zmienić "pokrętła" (parametry) modelu, aby zmniejszyć błąd (loss).
# - model.parameters(): Dajemy optymalizatorowi dostęp do wszystkich wag modelu,
#   które ma prawo modyfikować.
# - lr=5e-5: (Learning Rate) "Prędkość nauki". Bardzo mała wartość (0.00005),
#   aby model nie "zapomniał" tego, co już wie, a jedynie delikatnie dostosował się
#   do nowego zadania (fine-tuning).
# - 'W' w AdamW (Weight Decay): Technika zapobiegająca przeuczeniu – model
#   stara się trzymać wagi na niskim poziomie, co promuje prostsze rozwiązania.
# Kluczową różnicą między Adam a AdamW jest to, że AdamW stosuje odizolowaną (decoupled) regularyzację spadku wag.
optimizer = AdamW(model.parameters(), lr=5e-5)

z1, z2 = "Pawel is here", "Pawel is present"
# Zamieniamy nasze zdania testowe na format modelu i wysyłamy na CPU/GPU.
# Przenosimy dane na device (np. GPU), ponieważ model i dane muszą znajdować się na tym samym urządzeniu dla obliczeń.
inputs = tokenizer(z1, z2, return_tensors="pt").to(device)

# --- WYJAŚNIENIE TRYBU INFERENCJI I INTERPRETACJI WYNIKÓW ---
# with torch.inference_mode(): - Nowsza, bezpieczniejsza i szybsza wersja no_grad().
# Całkowicie izoluje model od mechanizmu gradientów. W świecie LLM "inferencja"
# to moment, w którym model nie uczy się, a jedynie wykorzystuje zdobytą wiedzę.
with torch.inference_mode():
    # model(**inputs) - Przekazujemy dane przez sieć neuronową.
    # .logits - Model zwraca surowe wyniki (punkty) dla każdej klasy (0 i 1).
    # KONTEKST LLM: Sieci neuronowe na ostatniej warstwie nie "myślą" kategoriami
    # prawdy czy fałszu, ale "napięciem" na neuronach wyjściowych.
    # LOGITY to właśnie te surowe wartości – im wyższy logit, tym bardziej model
    # "wierzy" w daną klasę, ale te liczby są nienormalizowane (np. mogą wynosić 5.4 i -1.2).
    # Pole 'token_type_ids' w BERT informuje model, który token należy do której sekwencji w parze zdań.
    logits_pre = model(**inputs).logits

    # F.softmax(logits_pre, dim=-1) - Magiczna funkcja matematyczna.
    # KONTEKST LLM: Ponieważ trudno operować na logitach, używamy Softmaxu, aby:
    # 1. Sprowadzić wszystkie wyniki do przedziału (0, 1) – czyli prawdopodobieństwa.
    # 2. Sprawić, by suma wszystkich wyników wynosiła 1.0 (100%).
    # To kluczowy moment: dzięki temu wiemy, czy model jest "pewny na 99%", czy "waha się 51/49".
    # dim=-1 oznacza, że liczymy to dla ostatniego wymiaru (czyli dla naszych klas).
    probs_pre = F.softmax(logits_pre, dim=-1)

print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Pewność przed nauką (Softmax): {probs_pre[0][1].item():.2%}")

# ==============================================================================
# 4. KONFIGURACJA ACCELERATE I HARMONOGRAMU (SCHEDULER)
# ==============================================================================
print("\n[4/7] KROK 4: Konfiguracja Accelerate i Schedulera...")
# ===========================================================================

# --- WYJAŚNIENIE FUNKCJI PREPARE() ---
# accelerator.prepare(): To najważniejszy moment w pracy z biblioteką Accelerate.
# Ta linia "owija" (wrapuje) Twoje obiekty w inteligentne opakowania, które:
# 1. MODEL I OPTYMALIZATOR: Przenosi je na odpowiednie urządzenie (CPU, GPU lub wiele GPU).
# 2. DATALOADERY: Zmienia je w wersje, które potrafią dostarczać dane do modelu
#    w sposób zsynchronizowany ze sprzętem.
# 3. AUTOMATYZACJA: Dzięki temu ten sam kod uruchomisz na swoim laptopie z procesorem
#    Intel Ultra 7, jak i na potężnym klastrze obliczeniowym, bez zmiany ani jednej linii kodu.
# WYJAŚNIENIE: Zamiast ręcznie pisać .to(device) dla każdego elementu,
# powierzasz to zadanie Acceleratorowi, który dba o maksymalną wydajność.
# Przy użyciu Accelerate, włączenie 'fp16=True' w argumentach umożliwiłoby trening z 16-bitową precyzją, oszczędzając pamięć i przyspieszając naukę.
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
# Parametr 'eval_strategy' (w klasie Trainer) określałby, czy ewaluacja odbywa się co określoną liczbę kroków ('steps'), czy co epokę.
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# ==============================================================================
# 5. PĘTLA TRENINGOWA (PROCES NAUKI)
# ==============================================================================

print(f"\n[5/7] KROK 5: Rozpoczynam pętlę treningową (Manual Training Loop)...")
progress_bar = tqdm(range(num_training_steps))

# model.train(): Przełącza model w tryb uczenia. Niektóre warstwy (jak Dropout)
# zachowują się inaczej podczas treningu niż podczas testów. To sygnał dla modelu:
# "Będziemy aktualizować Twoją wiedzę, bądź w gotowości".
model.train()

for epoch in range(num_epochs):
    print(f"\n--- Epoka {epoch + 1} / {num_epochs} ---")
    for step, batch in enumerate(train_dataloader):
        # --- KROK 1: FORWARD PASS (PRZEJŚCIE DO PRZODU) ---
        # Logika kolejności: Najpierw Forward, ponieważ model musi najpierw "zgadnąć" wynik, żebyśmy mogli sprawdzić, jak bardzo się pomylił względem prawdy.
        # Przepuszczamy dane przez wszystkie warstwy sieci. Model generuje
        # przewidywania (logits) i automatycznie porównuje je z poprawnymi
        # odpowiedziami (labels) zawartymi w 'batch'.
        outputs = model(**batch)

        # --- KROK 2: LOSS CALCULATION (OBLICZANIE STRATY) ---
        # 'loss' to pojedyncza liczba mówiąca o tym, jak bardzo model się pomylił.
        # Im większy błąd, tym wyższa strata. Naszym celem jest zminimalizowanie tej liczby.
        loss = outputs.loss

        # --- KROK 3: BACKWARD PASS (PROPAGACJA WSTECZNA) ---
        # Logika kolejności: Na podstawie błędu (loss) obliczamy gradienty (mapę poprawek). Bez błędu nie wiedzielibyśmy, co poprawiać.
        # accelerator.backward(loss): Obliczamy tzw. gradienty dla każdego parametru.
        # Gradient to informacja: "O ile i w którą stronę muszę przesunąć to konkretne
        # pokrętło w modelu, żeby strata (loss) była mniejsza?".
        # W bibliotece Accelerate zastępujemy standardowe loss.backward() metodą accelerator.backward(loss).
        accelerator.backward(loss)

        # --- KROK 4: OPTIMIZER STEP (AKTUALIZACJA WAG) ---
        # Logika kolejności: Dopiero teraz mamy "mapę" zmian (gradienty) i możemy faktycznie fizycznie zmienić wagi modelu (przekręcić pokrętła).
        # Teraz, gdy wiemy już, w którą stronę kręcić pokrętłami (mamy gradienty),
        # optymalizator AdamW faktycznie wykonuje ten ruch, zmieniając wagi modelu.
        optimizer.step()

        # --- KROK 5: SCHEDULER STEP (KOREKTA PRĘDKOŚCI) ---
        # Logika kolejności: Skoro model właśnie zrobił krok i się czegoś nauczył, aktualizujemy LR, żeby stawał się coraz precyzyjniejszy w kolejnych krokach.
        # Zgodnie z planem liniowym, po każdej aktualizacji wag nieco zmniejszamy
        # współczynnik uczenia (Learning Rate). Model z czasem staje się coraz
        # bardziej "ostrożny" w swoich zmianach.
        lr_scheduler.step()

        # --- KROK 6: ZERO GRAD (CZYSZCZENIE PAMIĘCI) ---
        # Logika kolejności: Na końcu usuwamy stare gradienty, bo już zostały zużyte do poprawy wag. Musimy mieć "czystą kartę" dla następnej paczki danych.
        # KLUCZOWE: PyTorch domyślnie dodaje nowe gradienty do starych.
        # Jeśli ich nie wyzerujemy, model "pogubi się", sumując poprawki z poprzednich
        # paczek danych. Czyścimy tablicę przed kolejnym krokiem.
        optimizer.zero_grad()

        progress_bar.update(1)

# 'Gradient Accumulation' pozwala symulować większy batch size poprzez akumulację gradientów z kilku mniejszych kroków przed wykonaniem optimizer.step().

# ==============================================================================
# 6. EWALUACJA (EGZAMIN GENERALNY MODELU)
# ==============================================================================
print("\n[6/7] KROK 6: Rozpoczynam sprawdzian modelu (Ewaluacja)...")

# evaluate.load: Pobieramy gotowy "arkusz ocen" dla zadania MRPC.
# Metryki (takie jak Accuracy czy F1-score) pozwalają nam obiektywnie ocenić,
# czy model faktycznie rozumie język, czy tylko zgaduje.
# Zadaniem funkcji compute_metrics (lub biblioteki evaluate) jest przekonwertowanie logitów na przewidywania i obliczenie miar jakości.
metric = evaluate.load("glue", "mrpc")

# model.eval(): Przełącza model w tryb "Egzaminu".
# Jest to absolutnie kluczowe! Wyłącza mechanizmy takie jak Dropout, które
# podczas treningu celowo wprowadzają szum, by model był odporniejszy.
# W trybie eval() model staje się stabilny i deterministyczny.
# model.eval() zmienia tryb pracy warstw (np. Dropout, Batchnorm) na odpowiedni dla fazy inferencji.
"""
WYJAŚNIENIE FAZY INFERENCJI (WNIOSKOWANIA):
To moment, w którym model wykorzystuje zamrożoną wiedzę do przewidywania wyników, nie zmieniając już swoich wag.
1. Brak nauki: Nie wykonujemy kroków backward() ani optimizer.step() - oszczędzamy czas i pamięć.
2. Stabilność: Warstwy takie jak Dropout są wyłączone, aby każda predykcja była stała i oparta na wszystkich neuronach.
3. Kierunek: Dane płyną wyłącznie "do przodu" (Forward Pass) - od tekstu wejściowego do logitów na wyjściu.
To odpowiednik wykorzystania wiedzy w praktyce (np. przez użytkownika aplikacji) po zakończeniu etapu nauki.
"""

for batch in eval_dataloader:
    # --- TRYB BEZ GRADIENTÓW (ZAMIAST no_grad MOŻNA UŻYĆ inference_mode) ---
    # Podczas sprawdzianu nie chcemy zmieniać wag modelu ani tracić pamięci
    # na zapamiętywanie ścieżki obliczeń do Backpropagation.
    # To sprawia, że proces jest dużo szybszy i zużywa ułamek pamięci RAM.
    # Użycie torch.no_grad() lub inference_mode podczas ewaluacji oszczędza pamięć i przyspiesza obliczenia poprzez wyłączenie śledzenia gradientów.
    with torch.inference_mode():
        outputs = model(**batch)

    # Logits: Pobieramy "pewność siebie" modelu dla każdej z dwóch klas.
    logits = outputs.logits

    # --- KROK 1: ARGMAX (DECYZJA MODELU) ---
    # Model wyrzuca logity (np. [-1.2, 3.5]). Funkcja argmax patrzy, która liczba
    # jest większa i zwraca jej indeks (w tym przypadku '1').
    # To jest moment, w którym model finalnie mówi nam: "Uważam, że to parafraza".
    predictions = torch.argmax(logits, dim=-1)

    # --- KROK 2: AKUMULACJA WYNIKÓW ---
    # metric.add_batch: Nie oceniamy modelu po jednej paczce.
    # Zbieramy wszystkie przewidywania (predictions) i porównujemy je
    # z prawdziwymi odpowiedziami (references/labels).
    # Metryka gromadzi te dane w pamięci, by na końcu obliczyć średnią.
    metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute(): Finalne obliczenie wyników (np. % poprawnych odpowiedzi).
# Jeśli do obiektu Trainer nie podano by 'eval_dataset', trening by trwał, ale nie otrzymalibyśmy raportów o metrykach podczas nauki.
print(f"👉 WYNIKI KOŃCOWE METRYKI: {metric.compute()}")

# ==============================================================================
# 7. TEST PRAKTYCZNY I ZAPISYWANIE (WERYFIKACJA EFEKTÓW)
# ==============================================================================
print("\n[7/7] KROK 7: Końcowy test praktyczny i zapisywanie modelu...")

# Zmieniamy na inference_mode() dla lepszej wydajności i bezpieczeństwa.
with torch.inference_mode():
    # --- WYJAŚNIENIE PRZENOSZENIA DANYCH (TO DEVICE) ---
    # Ta linia to "odprawa celna" dla danych. W PyTorch model i dane MUSZĄ przebywać
    # w tej samej pamięci (np. oba na CPU lub oba na GPU).
    # Ponieważ accelerator.prepare() mógł przenieść model na konkretne urządzenie,
    # musimy upewnić się, że nasze nowe, testowe zdania też tam trafią.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- PROCES PREDYKCJI (INFERENCJA) ---
    # Przepuszczamy zdania "Pawel is here" i "Pawel is present" przez sieć.
    # Model używa teraz swoich zaktualizowanych wag (tych "pokręteł", które
    # ustawiliśmy w Kroku 5), aby ocenić podobieństwo.
    logits_post = model(**inputs).logits

    # --- SOFTMAX (INTERPRETACJA DLA CZŁOWIEKA) ---
    # Softmax zamienia logity na prawdopodobieństwo.
    # Interesuje nas wartość pod indeksem [0][1], czyli "Prawdopodobieństwo, że to parafraza".
    # Wartość 1.0 = 100% pewności, 0.5 = model nie wie, 0.0 = na pewno nie parafraza.
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