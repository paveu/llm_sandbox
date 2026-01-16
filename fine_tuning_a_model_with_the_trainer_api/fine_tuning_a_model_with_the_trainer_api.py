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
# DYNAMIC PREDDING: Zmniejsza obciążenie obliczeniowe poprzez dopełnianie tylko do
# maksymalnej długości w obrębie każdej partii (batch), a nie całego zbioru.
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
# TA FUNKCJA TO "EGZAMINATOR". Określa, jak model będzie oceniany podczas nauki.
def compute_metrics(eval_preds):
    # EWALUACJA (Evaluation): Aby ocenić wydajność modelu w sposób zrozumiały dla człowieka,
    # potrzebujemy metryk, a nie tylko samej straty (loss).
    # Biblioteka 'evaluate' dostarcza prosty sposób na ładowanie metryk (np. GLUE MRPC).
    metric = evaluate.load("glue", "mrpc")

    # eval_preds to paczka zawierająca:
    # 1. Logity (co model "myśli" - surowe liczby)
    # 2. Labels (jaka jest prawda - etykiety 0/1)
    logits, labels = eval_preds

    # LOSS (Strata): Matematyczna miara błędu. Jeśli spada, model lepiej rozumie dane.
    # Wyobraź sobie Loss jako odległość od celu – im mniejszy Loss, tym bliżej jesteśmy prawdy.
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
    return metric.compute(predictions=predictions, references=labels)

# ==============================================================================
# 5. KONFIGURACJA TRENINGU (Zoptymalizowana pod Intel Ultra 7 + Zaawansowane funkcje)
# ==============================================================================
# TrainingArguments to "centrum sterowania" procesem nauki. To tutaj decydujemy o strategii.
training_args = TrainingArguments(
    output_dir="./test-trainer-cpu",
    # Używamy CPU, bo GPU zawiesza laptopa przy obliczeniach AI.
    use_cpu=True,

    # --- ZAAWANSOWANE FUNKCJE TRENINGOWE (ADVANCED FEATURES) ---

    # EVALUATION STRATEGY: Pozwala kontrolować częstotliwość przeprowadzania testów.
    # "epoch" oznacza sprawdzian (eval) po każdej pełnej epoce (przeczytaniu całych danych).
    # Dzięki temu po każdej epoce zobaczymy, czy model staje się mądrzejszy.
    eval_strategy="epoch",

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
    learning_rate=2e-5,  # "Długość kroku" (jak mocno gradient zmienia wagi).
    per_device_train_batch_size=4,  # Wykorzystujemy 14 rdzeni Twojego procesora.
    weight_decay=0.01,  # "Hamulec": zapobiega przypisywaniu ogromnych wag słowom.
    # WEIGHT DECAY to kara za zbyt duże wagi. Zapobiega sytuacji, w której model skupia się
    # obsesyjnie na jednym słowie (np. "the") ignorując resztę kontekstu.
    logging_steps=5,  # Co 5 paczek wypisz stan w konsoli.
)

# ==============================================================================
# 6. TWORZENIE TRAINERA (DYRYGENT PROCESU)
# ==============================================================================
# Trainer łączy model, dane, parametry i metryki w jedną maszynę treningową.
# Wyobraź sobie Trainera jako dyrygenta orkiestry – pilnuje, aby dane płynęły do modelu,
# metryki były liczone, a wagi aktualizowane w odpowiednim momencie.
trainer = Trainer(
    model=model,                 # Nasz BERT z nową głowicą.
    args=training_args,          # Wszystkie ustawienia z punktu 5.
    train_dataset=tokenized_datasets["train"],      # Materiały do nauki.
    eval_dataset=tokenized_datasets["validation"], # Materiały do sprawdzianu.
    data_collator=data_collator, # Maszyna do wyrównywania długości zdań (padding).
    processing_class=tokenizer,  # Nasz tłumacz tekstu na liczby.
    compute_metrics=compute_metrics, # Nasz egzaminator z punktu 4.
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
    # Ponownie zamieniamy logity na % po treningu za pomocą Softmaxu
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
trainer.save_model("./fine_tuning_a_model_with_the_trainer_api/moj_model_synonimy")
tokenizer.save_pretrained("./fine_tuning_a_model_with_the_trainer_api/moj_model_synonimy")
print("\nModel zapisany w './fine_tuning_a_model_with_the_trainer_api/moj_model_synonimy'!")