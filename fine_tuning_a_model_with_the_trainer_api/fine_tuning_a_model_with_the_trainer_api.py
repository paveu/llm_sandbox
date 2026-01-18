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
    EarlyStoppingCallback,
)
import wandb

import os

os.environ['WANDB_API_KEY'] = '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'

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
    # --- ETAP PRZYGOTOWANIA PALIWA DLA MODELU ---
    # Tokenizer nie posiada jeszcze "Głów Uwagi" (Heads), ale przygotowuje dane
    # w taki sposób, aby 144 głowy wewnątrz BERT-a wiedziały, co robić:

    # 1. INPUT_IDS: Zamienia słowa na liczby.
    #    Każdy numer to klucz do wielowymiarowego wektora znaczeniowego (Embedding).

    # 2. [CLS] (Classification Token): Dodaje specjalny token na samym początku.
    #    To "stacja zbiorcza" – model po przejściu przez wszystkie 144 głowy uwagi
    #    skupi całą wiedzę o relacji między zdaniami właśnie w tym jednym miejscu.
    #    Nasza "Głowica Klasyfikacyjna" (num_labels=2) patrzy TYLKO na ten token.

    # 3. [SEP] (Separator Token): Wstawia znacznik między zdanie A i B.
    #    Dzięki temu mechanizm Attention wie, gdzie kończy się kontekst jednego zdania.

    # 4. TOKEN_TYPE_IDS (Segment Embeddings): Tworzy maskę (0 dla zdania A, 1 dla B).
    #    To "podpowiedź" dla modelu, która pozwala mu fizycznie odróżnić od siebie dwa teksty.

    # 5. ATTENTION MASK: Tworzy mapę (1 dla tekstu, 0 dla paddingu).
    #    Mówi głowom uwagi: "Skup się na 1, ignoruj 0 (puste miejsca)".

    # 6. TRUNCATION: Bezpiecznik. Jeśli suma tokenów zdania A i B > 512,
    #    obetnie końcówkę, by nie przekroczyć fizycznej pamięci warstw Attention.
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

print("\n[2/6] Tokenizacja (zamiana słów na numery ID)...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# EPOKA (Epoch): Przeczytanie całej "książki" (zbioru danych) jeden raz.
# Epoch 1.0 oznacza, że model przejrzał wszystkie przykłady dokładnie raz.

# --- KLUCZOWE ROZRÓŻNIENIE: TRAIN VS VALIDATION ---
# Dlaczego dzielimy dane? Wyobraź sobie, że uczysz się do egzaminu z matematyki.

# 1. Zbiór TRENINGOWY (train): To Twoje ZADANIA DOMOWE.
# Tu model widzi zarówno pytania, jak i poprawne odpowiedzi. Na ich podstawie model
# kręci swoimi "pokrętłami" (wagami), żeby zminimalizować błąd.
# KIEDY UŻYWAMY: Zawsze podczas fazy właściwego uczenia (trainer.train()).
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(200))

# 2. Zbiór WALIDACYJNY (validation): To Twój EGZAMIN PRÓBNY.
# Model dostaje pytania, ale NIE widzi odpowiedzi podczas "rozwiązywania". My sprawdzamy
# jego odpowiedzi dopiero po fakcie. Model NIGDY nie poprawia swoich wag na podstawie
# tego zbioru – on służy tylko nam, żeby sprawdzić, czy model się uczy zasad, czy tylko
# kuje przykłady na pamięć.
# KIEDY UŻYWAMY: Podczas treningu, zazwyczaj po każdej epoce, żeby monitorować postępy.
# DLACZEGO TO POTRZEBNE? Bez walidacji nie wiedzielibyśmy, czy model nie wpada w
# OVERFITTING (przeuczenie). To sytuacja, w której model na zadaniach domowych ma 100%
# skuteczności, ale na nowym pytaniu, którego wcześniej nie widział, całkowicie polega.
tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(50))

# DATA COLLATOR: Wyrównuje długość zdań w paczce (batchu) dodając zera (padding).
# Modele wymagają, aby dane w jednej paczce (batch) miały identyczny wymiar.
# DYNAMIC PADDING: Zmniejsza obciążenie obliczeniowe poprzez dopełnianie tylko do
# maksymalnej długości w obrębie każdej partii (batch), a nie całego zbioru (np. 512).
# Kluczowe dla szybkości na procesorze Ultra 7 - nie marnujemy cykli na przetwarzanie zer.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==============================================================================
# 2. ARCHITEKTURA: WARSTWY (PIĘTRA), HEADY (OCZY) I GŁOWICA KLASYFIKATORA
# ==============================================================================
print("\n[3/6] Ładowanie modelu i instalacja nowej 'Głowicy' klasyfikatora...")

# KLUCZOWY MOMENT: Odcinamy oryginalną głowę BERT-a (tę do przewidywania słów)
# i "przyszywamy" nową, klasyfikacyjną głowę z 2 wyjściami (TAK/NIE).
#
# TRANSFER LEARNING: Nie uczysz modelu angielskiego od zera. Wykorzystujesz "wiedzę ogólną"
# bert-base-uncased i dodajesz do niej nową warstwę (Linear Layer),
# która uczy się wyłącznie specyfiki zadania MRPC (rozpoznawanie parafraz).
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# --- CZYM JEST 12 WARSTW (LAYERS) W BERT-BASE? (PIONOWA HIERARCHIA) ---
# Wyobraź sobie model jako 12-piętrowy wieżowiec. Dane wchodzą na parterze i jadą windą do góry.
# Każda warstwa (piętro) przetwarza tekst na coraz wyższym poziomie abstrakcji:
#
# 1. WARSTWY DOLNE (1-4): "Lingwiści Lokalni"
#    Analizują proste, fizyczne relacje między literami i słowami. Skupiają się na tym,
#    jak sąsiadujące słowa wpływają na siebie (np. czy "is" pasuje do "Pawel").
#    Budują fundament gramatyczny i rozpoznają proste struktury składniowe.
#
# 2. WARSTWY ŚRODKOWE (5-8): "Łącznicy Kontekstu"
#    Zaczynają rozumieć szerszy sens. Tu model zauważa zależności na poziomie całych
#    fraz. Rozpoznaje części mowy i rozumie, że zaimek "on" odnosi się do osoby
#    wymienionej trzy słowa wcześniej. To etap budowania "mapy powiązań".
#
# 3. WARSTWY GÓRNE (9-12): "Filozofowie Znaczenia"
#    Tutaj dzieje się magia czystej semantyki. Te warstwy nie analizują już liter,
#    ale czyste koncepcje (idee). Rozumieją, że "here" i "present" w tej konkretnej
#    parze zdań oznaczają to samo. To te warstwy wysyłają raport do tokena [CLS].
#
# DLACZEGO JEST ICH AŻ 12?
# Bo zrozumienie języka jest hierarchiczne. Nie da się zrozumieć ironii (poziom 12)
# bez zrozumienia znaczenia słowa (poziom 1). Każda warstwa korzysta z wyników
# pracy warstwy poprzedniej, coraz bardziej "wyżymając" esencję ze zdań.

# --- CZYM SĄ HEADY (GŁOWY UWAGI)? (POZIOMA SPECJALIZACJA) ---
# UWAGA: Warstwy to NIE to samo co Heady! Heady pracują WEWNĄTRZ każdej warstwy.
# Na każdym z 12 pięter (warstw) pracuje 12 wyspecjalizowanych pracowników (Głów).
# Łącznie masz 144 "mikro-mózgi" (12 warstw x 12 głów).
#
# DLACZEGO JEST ICH 12 NA KAŻDYM PIĘTRZE?
# Zamiast jednego pracownika, który patrzy na wszystko, masz 12 detektywów:
# Jeden pilnuje gramatyki, drugi szuka synonimów, trzeci patrzy na interpunkcję,
# a jeszcze inny sprawdza emocje. Pracują równolegle, dając modelowi 12 różnych
# perspektyw na to samo słowo w tym samym czasie.

# --- JAK DZIAŁAJĄ HEADY? (MECHANIZM Q, K, V) ---
# Każda głowa dla każdego słowa tworzy trzy wektory (matematyczne reprezentacje):
# 1. QUERY (Q) - "Zapytanie": Słowo 'here' wysyła zapytanie: "Szukam słów o miejscu".
# 2. KEY (K) - "Klucz": Słowo 'present' ma klucz, który pasuje: "Ja opisuję obecność".
# 3. VALUE (V) - "Wartość": Skoro Q i K do siebie pasują, głowa pobiera 'wartość'
#    (znaczenie) ze słowa 'present' i aktualizuje nim wektor słowa 'here'.
# Wynik tej "rozmowy" między słowami płynie w górę do kolejnej warstwy.

# --- WAGI (WEIGHTS) I TRENING ---
# WAGI: To miliony "pokręteł" (liczb) wewnątrz modelu. Trening to kręcenie nimi.
# Każda waga decyduje, jak mocno dany sygnał (np. informacja z konkretnej głowy
# w 10. warstwie) wpływa na wynik końcowy.
#
# Wagi w "mózgu" (warstwy Attention) są już ustawione przez Google na podstawie
# miliardów zdań, ale w Twojej nowej "Głowicy Klasyfikacyjnej" są na razie
# całkowicie LOSOWE – to dlatego przed treningiem model zgaduje wynik na 50%.
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
    # Są to surowe wyniki zwracane przez ostatnią warstwę modelu przed Softmaxem.
    logits_pre = outputs_pre.logits

# SOFTMAX: Funkcja, która zamienia surowe punkty (logity) na procenty (0-100%).
# W matematyce: przekształca wektor liczb na wektor prawdopodobieństw, które sumują się do 1.
probs_pre = F.softmax(logits_pre, dim=-1)
print(f"👉 Zdanie A: {z1} | Zdanie B: {z2}")
print(f"👉 Surowe logity PRZED nauką: {logits_pre}")
print(f"👉 Pewność PRZED nauką (Softmax): {probs_pre[0][1].item():.2%}")

# ==============================================================================
# 4. METRYKI, GRADIENTY I LOSS (Zasady oceniania)
# ==============================================================================

# Pobieramy gotowy "arkusz ocen" poza funkcją, aby uniknąć przeładowywania go.
metric = evaluate.load("glue", "mrpc")


# TA FUNKCJA TO "EGZAMINATOR". Określa, jak model będzie oceniany podczas nauki.
def compute_metrics(eval_preds):
    # EWALUACJA (Evaluation): Aby ocenić wydajność modelu w sposób zrozumiały dla człowieka,
    # potrzebujemy metryk, a nie tylko samej straty (loss).

    # eval_preds to paczka zawierająca:
    # 1. Logity (co model "myśli" - surowe liczby)
    # 2. Labels (jaka jest prawda - etykiety 0/1)
    logits, labels = eval_preds

    # LOSS (Strata): Matematyczna miara błędu. Jeśli spada, model lepiej rozumie dane.
    # Wyobraź sobie Loss jako odległość od celu – im mniejszy Loss, tym bliżej jesteśmy prawdy.
    # Na początku treningu Loss może być wysoki (np. ok. 0.7-0.9).
    # Powinieneś zobaczyć jego spadek z każdym krokiem (logging step).

    # GRADIENT: Instrukcja, w którą stronę kręcić wagą, aby LOSS malał.
    # Gradient to matematyczna "strzałka" mówiąca: "Zmniejsz tę wagę o 0.01, aby być bliżej wyniku".
    # GRAD_NORM: Siła tej instrukcji (im większy, tym gwałtowniejsza zmiana wag).

    # PREDICTIONS: To ostateczny "strzał" modelu (odpowiedź na egzaminie).
    # Wybieramy indeks (0 lub 1), który otrzymał najwięcej punktów w logitach.
    # ARGMAX: Wybieramy indeks o najwyższej wartości dla każdej próbki,
    # aby przekształcić logity w konkretne klasy (0 lub 1).
    # Przykład: jeśli logity to [-2.5, 3.1], argmax wybierze indeks "1" (klasa pozytywna).
    predictions = np.argmax(logits, axis=-1)

    # LABELS: To "klucz odpowiedzi" (prawdziwe etykiety ze zbioru danych).
    # Nauczyciel (metric) porównuje predictions z labels.
    # Wynik to słownik zawierający Accuracy (dokładność) oraz F1 Score (średnia precyzji i pełności).
    # Accuracy mówi: "Ile razy trafiłeś?". F1 mówi: "Jak dobrze radzisz sobie z obiema klasami?".
    # W zadaniu MRPC metryka F1 jest ważniejsza niż samo Accuracy, ponieważ zbiory te bywają niezbalansowane.

    # --- INTERPRETACJA METRYK W CZASIE RZECZYWISTYM ---
    # Jeśli Accuracy rośnie wolniej niż spada Loss, to znaczy, że model staje się
    # pewniejszy swoich decyzji, ale jeszcze nie na tyle, by zmienić klasyfikację błędnych przykładów.
    return metric.compute(predictions=predictions, references=labels)


# ==============================================================================
# 5. KONFIGURACJA TRENINGU (Zoptymalizowana pod Intel Ultra 7 + Zaawansowane funkcje)
# ==============================================================================
# TrainingArguments to "centrum sterowania" procesem nauki. To tutaj decydujemy o strategii.

# Inicjalizacja Weights & Biases do śledzenia eksperymentów
wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis-huggingface-trainer-api")

training_args = TrainingArguments(
    output_dir="./test-trainer-cpu",
    # Używamy CPU, bo GPU zawiesza laptopa przy obliczeniach AI.
    use_cpu=True,

    # --- ZAAWANSOWANE FUNKCJE TRENINGOWE (ADVANCED FEATURES) ---

    # EVALUATION STRATEGY: Pozwala kontrolować częstotliwość przeprowadzania testów.
    # "epoch" oznacza sprawdzian (eval) po każdej pełnej epoce (przeczytaniu całych danych).
    # Dzięki temu po każdej epoce zobaczymy, czy model staje się mądrzejszy.
    # ANALIZA: Jeśli Validation Loss zacznie rosnąć po 2. epoce, mimo że Train Loss spada,
    # mamy do czynienia z przeuczeniem (Overfitting).
    eval_strategy="epoch",
    # SAVE STRATEGY: Musi być identyczna jak eval_strategy, gdy używamy load_best_model_at_end.
    # Dzięki temu Trainer po każdym sprawdzianie (epoch) zapisze wagi modelu na dysku,
    # co pozwoli mu na samym końcu wrócić do tej wersji, która miała najlepsze wyniki.
    save_strategy="epoch",

    # LEARNING RATE SCHEDULER: Model domyślnie zmniejsza "długość kroku" (LR) wraz z treningiem.
    # "cosine" (kosinusoidalny) to zaawansowany sposób: najpierw model uczy się szybko,
    # a potem coraz delikatniej "cyzeluje" wagi, co zapobiega psuciu wyników na koniec.
    # Metafora: Na początku biegniesz w stronę celu, a na końcu robisz małe, precyzyjne kroczki.
    lr_scheduler_type="cosine",

    # MIXED PRECISION (fp16): Pozwala na obliczenia na liczbach 16-bitowych zamiast 32-bitowych.
    # UWAGA: Na CPU zazwyczaj zostawiamy False, ale na nowszych GPU ustawienie fp16=True
    # dramatycznie przyspiesza trening i oszczędza połowę pamięci VRAM.
    # Mniejsza precyzja (mniej cyfr po przecinku) pozwala "upchnąć" więcej obliczeń naraz.
    fp16=False,

    # GRADIENT ACCUMULATION: Technika dla osób z małą ilością pamięci.
    # Jeśli ustawisz gradient_accumulation_steps=4 i batch_size=4, model zachowa się tak,
    # jakby trenował na paczce o rozmiarze 16, ale "pogryzie" ją na mniejsze kawałki po 4.
    # Dzięki temu oszczędzamy pamięć RAM, symulując pracę na potężnym sprzęcie.
    gradient_accumulation_steps=1,

    num_train_epochs=3,  # Model przeczyta 200 zdań 3 razy (lepsza stabilność).
    # LEARNING_RATE: To "pewność siebie" modelu. 2e-5 to bardzo mała wartość (0.00002).
    # Małe kroki zapobiegają "przeskoczeniu" idealnego ustawienia wag (tzw. overshooting).
    # Jeśli krzywa straty na W&B jest bardzo "poszarpana", warto zmniejszyć tę wartość.
    learning_rate=2e-5,  # "Długość kroku" (jak mocno gradient zmienia wagi).

    per_device_train_batch_size=4,  # Wykorzystujemy 14 rdzeni Twojego procesora.
    # ANALIZA BATCHA: Większy batch size (np. 16, 32) daje gładsze krzywe uczenia,
    # bo kierunek zmian wag jest uśredniany z większej liczby przykładów.

    weight_decay=0.01,  # "Hamulec": zapobiega przypisywaniu ogromnych wag słowom.
    # WEIGHT DECAY to kara za zbyt duże wagi. Zapobiega sytuacji, w której model skupia się
    # obsesyjnie na jednym słowie (np. "the") ignorując resztę kontekstu.
    logging_steps=5,  # Co 5 paczek wypisz stan w konsoli.
    report_to="wandb",  # Wysyłanie logów do Weights & Biases
    load_best_model_at_end=True,  # Załaduj najlepszy model na końcu (ten z najniższym Validation Loss).
)

# ==============================================================================
# 6. TWORZENIE TRAINERA (DYRYGENT PROCESU)
# ==============================================================================
# Trainer łączy model, dane, parametry i metryki w jedną maszynę treningową.
# Wyobraź sobie Trainera jako dyrygenta orkiestry – pilnuje, aby dane płynęły do modelu,
# metryki były liczone, a wagi aktualizowane w odpowiednim momencie.
trainer = Trainer(
    model=model,  # Nasz BERT z nową głowicą.
    args=training_args,  # Wszystkie ustawienia z punktu 5.
    train_dataset=tokenized_datasets["train"],  # Materiały do nauki.
    eval_dataset=tokenized_datasets["validation"],  # Materiały do sprawdzianu.
    data_collator=data_collator,  # Maszyna do wyrównywania długości zdań (padding).
    processing_class=tokenizer,  # Nasz tłumacz tekstu na liczby.
    compute_metrics=compute_metrics,  # Nasz egzaminator z punktu 4.

    # EARLY STOPPING CALLBACK: Mechanizm bezpieczeństwa.
    # Jeśli przez 3 sprawdziany (patience=3) model nie poprawi wyniku na zbiorze walidacyjnym,
    # Trainer przerwie naukę, chroniąc model przed "wykuciem danych na blachę" (Overfitting).
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ==============================================================================
# 7. TRENING I ANALIZA ZMIAN W "MÓZGU"
# ==============================================================================
print("\n[4/6] Start Fine-tuningu (Trening nowej głowy na Intel Ultra 7)...")
# LOSS (Strata) powinna spadać z każdym krokiem.
# Podczas trainer.train() model wykonuje pętlę treningową:
# 1. Przekazuje dane przez model (Forward pass) - "Czytanie pytania"
# 2. Oblicza błąd (Loss) - "Sprawdzanie błędu"
# 3. Oblicza gradienty (Backward pass) - "Szukanie przyczyny błędu"
# 4. Aktualizuje pokrętła (Optimizer step) - "Poprawa wiedzy (zmiana wag)"

# --- JAK INTERPRETOWAĆ LOGI W TRAKCIE? ---
# Jeśli "Training Loss" spada, a "Validation Loss" stoi w miejscu lub rośnie:
# Oznacza to, że model traci zdolność generalizacji. Ciesz się wtedy, że masz EarlyStopping!
trainer.train()

print("\n[5/6] Sprawdzanie zmian w 'mózgu' modelu...")
weights_after = model.classifier.weight.data[0][:5].clone()
print(f"👉 Wagi przed: {weights_before}")
print(f"👉 Wagi po:    {weights_after}")

# --- DLACZEGO LICZYMY 'diff' I CO TO MA DO ZNACZENIA? ---
# Wyobraź sobie, że waga (weight) to "siła zaufania" do danej cechy:
# 1. Model przed nauką ma losowe zaufanie (np. ufa literze 'X' w szukaniu synonimów).
# 2. Podczas treningu model zauważa: "Zaraz, litera 'X' nic mi nie mówi o parafrazie,
#    ale wektor z 10. warstwy (ten od synonimów) jest mega ważny!".
# 3. Model "przesuwa" wagę (liczbę) z cechy nieistotnej na istotną.
#
# ZWIĄZEK Z TRENINGIEM:
# Te liczby w 'diff' to ślad po tym, co model widział w tych 200 zdaniach.
# Jeśli 'diff' jest wyraźne, to znaczy, że te 200 zdań dało modelowi "lekcję",
# która kazała mu zmienić zdanie o tym, co jest ważne.
#
# TO NIE JEST PORÓWNANIE ZDAŃ - TO BILANS ZYSKÓW I STRAT WIEDZY.
# Wynik 'diff' mówi nam: "O tyle model stał się inny po przeczytaniu książki".

diff = weights_after - weights_before

# DLACZEGO TO JEST WAŻNE DLA POCZĄTKUJĄCEGO?
# Jeśli 'diff' wynosiłoby same zera, oznaczałoby to, że model niczego się nie nauczył
# (np. Learning Rate był za mały lub dane były błędne).
# Każda liczba różna od zera w 'diff' to dowód na to, że model "żyje" i reaguje na dane.
# Duże wartości w 'diff' mogą sugerować, że model gwałtownie zmieniał zdanie (niestabilny trening).
print(f"👉 Różnica (fizyczny efekt nauki): {diff}")

# ==============================================================================
# 8. TEST PRAKTYCZNY PO TRENINGU (WERYFIKACJA "NOWYCH NAWYKÓW" MODELU)
# ==============================================================================
print("\n[6/6] TEST PO TRENINGU (Analiza synonimów):")

# torch.inference_mode() – Wyłączamy "tryb nauki".
# Mówimy modelowi: "Teraz nie masz nic zmieniać w wagach, po prostu użyj tego,
# czego się przed chwilą nauczyłeś". To oszczędza RAM i przyspiesza działanie.
with torch.inference_mode():
    # FORWARD PASS: Przepuszczamy nasze testowe zdania ("Pawel is here/present")
    # przez odświeżoną architekturę. Teraz każda ze 144 głów uwagi (Heads)
    # wysyła sygnał do nowo ustawionej Głowicy Klasyfikacyjnej.
    outputs_post = model(**inputs)

    # LOGITY: To surowe punkty (np. [-2.5, 4.1]).
    # To jest moment, w którym model "krzyczy" wynik na podstawie swoich nowych wag.
    # Wyższa liczba na drugim miejscu (indeks 1) oznacza: "Tak, to parafraza!".
    logits_post = outputs_post.logits

    # SOFTMAX: Zamiana surowej siły głosu na cywilizowane procenty.
    # Ta funkcja bierze logity i rozdziela je tak, by suma obu wynosiła 100%.
    # Przykładowo: logity [-2, 4] zmienią się w [0.2%, 99.8%].
    probs_post = F.softmax(logits_post, dim=-1)

    # CONFIDENCE: Wyciągamy konkretną liczbę dla klasy "Parafraza" (indeks 1).
    # .item() zamienia obiekt PyTorch (tensor) na zwykłą liczbę typu float w Pythonie.
    confidence = probs_post[0][1].item()

# --- DLACZEGO TO JEST MOMENT "PRAWDY"? ---
# 1. Przed nauką: Głowica miała losowe wagi, więc wynik Softmax był bliski 50% (rzut monetą).
# 2. Po nauce: Głowica "wie", że sygnały o synonimach z Heads są ważne.
#    Dlatego logity dla klasy 1 powinny być teraz znacznie wyższe.
#
# ANALIZA: Jeśli pewność (Confidence) wzrosła, np. z 52% na 88%, Twój fine-tuning
# odniósł sukces – model fizycznie "zrozumiał" intencję Twojego zadania.

# ZMIANA PEWNOŚCI: Twoje testowe zdania ("Pawel is here" vs "Pawel is present")
# powinny po treningu uzyskać znacznie wyższy wynik procentowy w klasie 1 (parafraza),
# o ile 200 przykładów wystarczy, by model "zrozumiał" intencję zadania.
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
trainer.save_model("./fine_tuning_a_model_with_the_trainer_api/moj_model_synonimy")
tokenizer.save_pretrained("./fine_tuning_a_model_with_the_trainer_api/moj_model_synonimy")
print("\nModel zapisany w './fine_tuning_a_model_with_the_trainer_api/moj_model_synonimy'!")