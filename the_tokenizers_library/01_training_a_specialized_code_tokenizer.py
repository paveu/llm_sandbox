"""
=============================================================================
WSTĘP: TRENOWANIE NOWEGO TOKENIZERA (ROZDZIAŁ 6.2)
=============================================================================
Kiedy pracujemy z domenami takimi jak medycyna, prawo czy programowanie, 
standardowe tokenizatory (jak GPT-2) stają się nieefektywne.

WYZWANIE: Standardowy tokenizer GPT-2 widzi kod Python jako ciąg dziwnych 
znaków, przez co zużywa bardzo dużo tokenów na proste komendy.

ROZWIĄZANIE: Wykorzystujemy 'BPE' (Byte-Pair Encoding) do wytrenowania 
nowego słownika na 600 tysiącach przykładów kodu źródłowego.
=============================================================================
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer


def get_training_corpus(dataset):
    """
    GENERATOR DANYCH:
    Zamiast ładować wszystko do RAM, podajemy dane partiami po 1000 tekstów.
    To pozwala na trenowanie na zbiorach danych idących w gigabajty.
    """
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        # 'content' to kolumna zawierająca surowy kod Python
        yield samples["content"]


def main():
    # --- KROK 1: ŁADOWANIE DANYCH ---
    # Używamy zbioru CodeParrot - potężnego zbioru skryptów w Pythonie.
    print("--- Krok 1: Pobieranie kodu z Hugging Face Hub ---")
    raw_datasets = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    print(f"Załadowano {len(raw_datasets)} skryptów do analizy.")

    # --- KROK 2: BAZOWY TOKENIZER ---
    # Chcemy zachować strukturę GPT-2 (Byte-level BPE), ale zmienić SŁOWNIK.
    print("\n--- Krok 2: Pobieranie wzorca (GPT-2) ---")
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # --- KROK 3: TRENING ---
    # Cel: Tokenizer analizuje cały korpus i znajduje najczęstsze pary znaków.
    print("\n--- Krok 3: Trening (Budowanie nowego słownika subwordów) ---")
    # 52000 to rozmiar słownika (vocab_size) - tyle unikalnych tokenów chcemy mieć.
    training_corpus = get_training_corpus(raw_datasets)
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    print("Sukces! Tokenizer nauczył się statystyk kodu.")

    # --- KROK 4: PORÓWNANIE I TEST ---
    # Sprawdzamy, czy nowy tokenizer jest bardziej 'kompaktowy' dla kodu.
    example_code = """def add_numbers(a, b):
    \"\"\"Dodaj dwie liczby.\"\"\"
    return a + b"""

    old_tokens = old_tokenizer.tokenize(example_code)
    new_tokens = tokenizer.tokenize(example_code)

    print("\n--- TEST WYDAJNOŚCI ---")
    print(f"Liczba tokenów (Oryginalny GPT-2): {len(old_tokens)}")
    print(f"Liczba tokenów (Twój Tokenizer):  {len(new_tokens)}")

    # Zauważysz, że Twój tokenizer lepiej radzi sobie ze spacjami i słowami 'def' czy 'return'
    print(f"\nReprezentacja Twojego tokenizera:\n{new_tokens}")

    # --- KROK 5: ZAPISYWANIE ---
    # Zapisujemy pliki 'vocab.json' oraz 'merges.txt', aby użyć ich w modelu.
    save_path = "code-tokenizer-python"
    tokenizer.save_pretrained(save_path)
    print(f"\n--- ZAPISANO W: {os.path.abspath(save_path)} ---")


if __name__ == "__main__":
    main()

# PRO-TIP:
# Trenowanie własnego tokenizera to pierwszy krok do stworzenia własnego 
# modelu typu 'Copilot'. Zmniejszenie liczby tokenów na wejściu pozwala 
# modelowi przetwarzać dłuższe fragmenty kodu w tym samym oknie kontekstowym!