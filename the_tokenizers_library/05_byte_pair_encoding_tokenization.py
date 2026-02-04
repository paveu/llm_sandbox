"""
=============================================================================
WSTĘP: ARCHITEKTURA TOKENIZERÓW (ROZDZIAŁ 6.4 - 6.5)
=============================================================================
Zanim algorytm (BPE, WordPiece, Unigram) zacznie działać, tekst musi przejść
przez dwa kluczowe etapy:
1. NORMALIZACJA: Ujednolicenie tekstu (małe litery, usuwanie ogonków itp.).
2. PRE-TOKENIZACJA: Wstępne pocięcie tekstu na słowa lub fragmenty.

CEL: Zrozumienie, jak surowy tekst wejściowy przygotowywany jest dla modelu.
=============================================================================
"""

import os

# Wyłączenie ostrzeżeń o równoległości dla czystości logów
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Metaspace


def sekcja_normalizacja():
    """
    KROK 1: NORMALIZACJA
    Zadaniem jest sprawienie, by tekst był 'czysty' i przewidywalny.
    """
    print("--- 1. NORMALIZACJA (PRZYKŁADY Z KURSU) ---")

    # Przykład 1: Pobranie normalizatora z BERT
    # BERT używa m.in. czyszczenia znaków sterujących i małych liter.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"Normalizator BERT: {tokenizer.backend_tokenizer.normalizer}")

    # Przykład 2: Ręczne tworzenie sekwencji normalizującej
    # NFD: Rozkład znaków Unicode (np. 'é' na 'e' + '´')
    # StripAccents: Usunięcie '´', zostaje samo 'e'
    custom_normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tekst_kursowy = "Héllò hôw are ü?"

    print(f"Oryginał: {tekst_kursowy}")
    print(f"Po normalizacji (NFD + Lower + Strip): {custom_normalizer.normalize_str(tekst_kursowy)}\n")


def sekcja_pre_tokenizacja():
    """
    KROK 2: PRE-TOKENIZACJA
    Dzielenie ciągu znaków na mniejsze jednostki (np. słowa).
    Każdy model robi to inaczej!
    """
    print("--- 2. PRE-TOKENIZACJA (PRZYKŁADY Z KURSU) ---")
    tekst = "Hello, how are  you?"  # Zauważ podwójną spację przed 'you'

    # Przykład 3: GPT-2 (ByteLevel)
    # ByteLevel traktuje spacje jako specjalne znaki (często widoczne jako 'Ġ').
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    print(f"GPT-2 (ByteLevel):")
    print(tokenizer_gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tekst))

    # Przykład 4: BERT (Whitespace + Punctuation)
    # BERT dzieli po spacjach ORAZ wokół znaków interpunkcyjnych.
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"\nBERT (Whitespace + Punctuation):")
    print(tokenizer_bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tekst))

    # Przykład 5: T5 (Metaspace)
    # T5 zastępuje spacje znakiem '_' i dzieli tekst zachowując te informacje.
    tokenizer_t5 = AutoTokenizer.from_pretrained("t5-small")
    print(f"\nT5 (Metaspace):")
    print(tokenizer_t5.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tekst))
    print("\n")


def sekcja_budowanie_potoku():
    """
    KROK 3: RĘCZNE BUDOWANIE ELEMENTÓW
    Możemy sami decydować, jak chcemy ciąć tekst.
    """
    print("--- 3. BUDOWANIE WŁASNEGO POTOKU (PIPELINE) ---")

    # Whitespace: Najprostszy podział (tylko spacje)
    pre_tok_ws = Whitespace()
    print(f"Ręczny Whitespace: {pre_tok_ws.pre_tokenize_str('Hello, how are you?')}")

    # ByteLevel: Często używany w modelach typu RoBERTa/GPT
    # Parametr add_prefix_space decyduje, czy dodawać spację na początku tekstu
    pre_tok_byte = ByteLevel(add_prefix_space=False)
    print(f"Ręczny ByteLevel:  {pre_tok_byte.pre_tokenize_str('Hello, how are you?')}")

    print("\n--- WYNIK KOŃCOWY ---")
    print("Zauważ, jak różnie modele traktują interpunkcję i spacje.")
    print("Te różnice mają ogromny wpływ na to, jak model później 'rozumie' tekst.")


if __name__ == "__main__":
    print("ZASADA BPE: Zaczyna od małego słownika i łączy NAJCZĘSTSZE pary tokenów.")
    sekcja_normalizacja()
    sekcja_pre_tokenizacja()
    sekcja_budowanie_potoku()