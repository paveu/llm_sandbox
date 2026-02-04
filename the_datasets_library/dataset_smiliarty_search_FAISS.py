# --- INICJALIZACJA I NAPRAWA ŚRODOWISKA ---
import numpy as np

# CO ROBI: Przypisuje standardowy typ 'object' do aliasu 'np.object'.
# DLACZEGO: Nowy NumPy 2.x usunął ten alias, co psuje starsze biblioteki.
# WYNIK: (Cicha poprawka w pamięci Pythona)
np.object = object

import torch  # CO ROBI: Importuje silnik obliczeniowy PyTorch.
import pandas as pd  # CO ROBI: Importuje bibliotekę do pracy na tabelach danych.
import faiss  # CO ROBI: Importuje bibliotekę do błyskawicznego przeszukiwania wektorów.
from datasets import load_dataset, Dataset  # CO ROBI: Pobiera narzędzia do zarządzania dużymi zbiorami danych.
from transformers import AutoTokenizer, AutoModel  # CO ROBI: Pobiera narzędzia do obsługi modeli Transformer.

# --- KROK 1: PRZYGOTOWANIE DANYCH ---
# CO ROBI: Informuje o rozpoczęciu przetwarzania.
# WYNIK: Krok 1: Ładowanie i przygotowanie danych...
print("Krok 1: Ładowanie i przygotowanie danych...")

# CO ROBI: Ściąga z chmury HF 1000 pierwszych wierszy zbioru wiadomości 'ag_news'.
# WYNIK: Dataset({ features: ['text', 'label'], num_rows: 1000 })
raw_dataset = load_dataset("ag_news", split="train[:1000]")

# CO ROBI: Definiuje sposób łączenia danych w jeden czytelny dla modelu blok.
# WYNIK: (Funkcja gotowa do użycia)
def concatenate_text(examples):
    # Tworzymy f-string, który dodaje nagłówek "News Article" przed właściwą treścią.
    return {"text": f"News Article \n {examples['text']}"}

# CO ROBI: Przetwarza cały zbiór danych funkcją concatenate_text w sposób zrównoleglony.
# WYNIK: Map: 100%|██████████| 1000/1000 [00:00<00:00, 19132.33ex/s]
comments_dataset = raw_dataset.map(concatenate_text)

# --- KROK 2: KONFIGURACJA MODELU JĘZYKOWEGO ---
# CO ROBI: Informuje o przejściu do etapu sztucznej inteligencji.
# WYNIK: Krok 2: Inicjalizacja modelu i generowanie wektorów...
print("Krok 2: Inicjalizacja modelu i generowanie wektorów...")

# CO ROBI: Określa nazwę modelu zoptymalizowanego pod kątem wyszukiwania odpowiedzi (QA).
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# CO ROBI: Pobiera tokenizer, który tnie słowa na kawałki (tokeny) zrozumiałe dla modelu.
# WYNIK: (Pobieranie plików tokenizer_config.json, vocab.txt itp.)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# CO ROBI: Pobiera właściwy "mózg" modelu (sieć neuronową) z wyuczonymi wagami.
# WYNIK: (Pobieranie modelu o rozmiarze ok. 400-500 MB)
model = AutoModel.from_pretrained(model_ckpt)

# CO ROBI: Sprawdza, czy w systemie dostępna jest karta NVIDIA (CUDA).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CO ROBI: Wysyła model do pamięci karty graficznej (GPU) dla przyspieszenia obliczeń.
# WYNIK: Używane urządzenie: cuda
model.to(device)
print(f"Używane urządzenie: {device}")

# CO ROBI: Definiuje metodę "wyciągania" sensu zdania z warstw ukrytych modelu.
def cls_pooling(model_output):
    # Pobieramy tylko pierwszy wektor (token [CLS]) z ostatniej warstwy.
    return model_output.last_hidden_state[:, 0]

# CO ROBI: Zamienia listę tekstów na macierz liczb (embeddingi).
def get_embeddings(text_list):
    # Zamiana tekstu na tensory (macierze) z dopełnieniem (padding) do równej długości.
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    # Przeniesienie wejścia na to samo urządzenie co model (GPU).
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    # Wyłączenie śledzenia gradientów (blokada uczenia), co oszczędza RAM i przyspiesza skrypt.
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Wywołanie poolingu i zwrócenie "skondensowanego" sensu zdania.
    return cls_pooling(model_output)

# CO ROBI: Wykonuje najcięższą pracę – zamienia każdy z 1000 artykułów na wektor 768 liczb.
# WYNIK: Map: 100%|██████████| 1000/1000 [00:10<00:00, 81.08ex/s]
embeddings_dataset = comments_dataset.map(
    # Wynik z GPU przenosimy z powrotem na CPU i zamieniamy na format NumPy dla FAISS.
    lambda x: {"embeddings": get_embeddings(x["text"]).cpu().numpy()[0]}
)

# --- KROK 3: BUDOWANIE INDEKSU PRZESTRZENNEGO ---
# CO ROBI: Informuje o budowaniu bazy wektorowej.
# WYNIK: Krok 3: Budowanie indeksu FAISS...
print("Krok 3: Budowanie indeksu FAISS...")

# CO ROBI: Tworzy indeks FAISS na kolumnie 'embeddings'. Organizuje wektory w drzewo wyszukiwań.
# WYNIK: 100%|██████████| 1/1 [00:00<00:00, 75.91it/s]
embeddings_dataset.add_faiss_index(column="embeddings")

# --- KROK 4: ZADAWANIE PYTANIA ---
# CO ROBI: Definiuje zapytanie, na które chcemy znaleźć odpowiedź.
question = "Space exploration and technology news"
# WYNIK: Zadawanie pytania: Space exploration and technology news
print(f"\nZadawanie pytania: {question}")

# CO ROBI: Zamienia Twoje pytanie na wektor w tej samej przestrzeni 768-wymiarowej.
q_embedding = get_embeddings([question]).cpu().numpy()

# CO ROBI: Szuka 3 dokumentów, których wektory mają największy iloczyn skalarny z zapytaniem.
# WYNIK: (Zwraca zmienne scores (podobieństwo) i samples (dane dokumentów))
scores, samples = embeddings_dataset.get_nearest_examples("embeddings", q_embedding, k=3)

# --- KROK 5: WYŚWIETLANIE WYNIKÓW ---
# CO ROBI: Wyświetla finalny nagłówek.
# WYNIK: NAJLEPSZE WYNIKI:
print("\nNAJLEPSZE WYNIKI:")

# CO ROBI: Pętla przechodząca przez 3 znalezione wyniki.
for i in range(len(samples["text"])):
    # CO ROBI: Wyświetla matematyczną ocenę dopasowania.
    # WYNIK: POZIOM DOPASOWANIA: 29.71
    print(f"POZIOM DOPASOWANIA: {scores[i]:.2f}")

    # CO ROBI: Wyświetla początek tekstu artykułu, który AI uznała za pasujący.
    # WYNIK: TREŚĆ: News Article \n Redesigning Rockets...
    print(f"TREŚĆ: {samples['text'][i][:150]}...")

    # CO ROBI: Wyświetla separator graficzny dla czytelności.
    # WYNIK: ------------------------------------------------------------
    print("-" * 60)