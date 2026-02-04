import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents, BertNormalizer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Metaspace, Sequence


def sekcja_normalizacja():
    print("--- 1. NORMALIZACJA (PRZYKŁADY Z KURSU) ---")

    # Przykład 1: Pobranie normalizatora bezpośrednio z modelu BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"Normalizator BERT: {tokenizer.backend_tokenizer.normalizer}")

    # Przykład 2: Ręczne tworzenie normalizatora (NFD, Lowercase, StripAccents)
    # W kursie użyto normalizers.Sequence do połączenia tych kroków
    custom_normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tekst_kursowy = "Héllò hôw are ü?"
    print(f"Oryginał: {tekst_kursowy}")
    print(f"Po normalizacji: {custom_normalizer.normalize_str(tekst_kursowy)}\n")


def sekcja_pre_tokenizacja():
    print("--- 2. PRE-TOKENIZACJA (PRZYKŁADY Z KURSU) ---")
    tekst = "Hello, how are  you?"  # tekst z kursu (podwójna spacja)

    # Przykład 3: Pre-tokenizator GPT-2 (ByteLevel)
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    print(f"GPT-2 pre-tokenization:")
    print(tokenizer_gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tekst))

    # Przykład 4: Pre-tokenizator BERT
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"\nBERT pre-tokenization:")
    print(tokenizer_bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tekst))

    # Przykład 5: Pre-tokenizator T5 (Metaspace)
    tokenizer_t5 = AutoTokenizer.from_pretrained("t5-small")
    print(f"\nT5 pre-tokenization:")
    print(tokenizer_t5.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tekst))
    print("\n")


def sekcja_budowanie_potoku():
    print("--- 3. BUDOWANIE WŁASNEGO POTOKU (PIPELINE) ---")

    # Dokumentacja pokazuje, jak łączyć komponenty ręcznie
    # Używamy Whitespace, aby podzielić na słowa
    pre_tok_ws = Whitespace()
    print(f"Ręczny Whitespace: {pre_tok_ws.pre_tokenize_str('Hello, how are you?')}")

    # ByteLevel w wersji z kursu
    pre_tok_byte = ByteLevel(add_prefix_space=False)
    print(f"Ręczny ByteLevel:  {pre_tok_byte.pre_tokenize_str('Hello, how are you?')}")


if __name__ == "__main__":
    sekcja_normalizacja()
    sekcja_pre_tokenizacja()
    sekcja_budowanie_potoku()