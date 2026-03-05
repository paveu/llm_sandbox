"""
=============================================================================
WSTĘP: NORMALIZACJA I PRE-TOKENIZACJA (ROZDZIAŁ 6.4 - 6.5)
=============================================================================
Zanim algorytmy takie jak BPE czy WordPiece zaczną łączyć znaki, tekst musi
zostać wstępnie przygotowany. Ten proces nazywamy Pipeline'em Tokenizatora.

CO CHCEMY OSIĄGNĄĆ?
Chcemy zobaczyć, jak różne modele 'widzą' ten sam tekst wejściowy. Różnice
w pre-tokenizacji decydują o tym, czy model zachowa informację o spacjach,
czy poprawnie odczyta interpunkcję i jak poradzi sobie z błędami typograficznymi.
=============================================================================
"""

from transformers import AutoTokenizer

def section_normalization():
    """
    SEKCJA 1: NORMALIZACJA
    Cel: Ujednolicenie tekstu, aby zredukować rozmiar słownika (np. 'H' i 'h' to to samo).
    """
    print("--- 1. NORMALIZACJA (BERT UNCASED) ---")
    # BERT uncased wykonuje m.in. Lowercase i StripAccents (usuwanie ogonków)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Dostęp do backendu napisanego w Rust (bardzo wydajny)
    backend = tokenizer.backend_tokenizer

    raw_text = "Héllò hôw are ü?"
    normalized_text = backend.normalizer.normalize_str(raw_text)

    print(f"Tekst wejściowy: {raw_text}")
    print(f"Po normalizacji: {normalized_text}")
    print("-" * 30)


def section_pretokenization():
    """
    SEKCJA 2: PRE-TOKENIZACJA
    Cel: Wstępny podział tekstu na mniejsze kawałki przed właściwym kodowaniem.
    """
    print("--- 2. PRE-TOKENIZACJA (RÓŻNE MODELE) ---")
    text = "Hello, how are you?"

    # --- PRZYKŁAD BERT ---
    # BERT stosuje Whitespace oraz Punctuation. Interpunkcja staje się osobnym tokenem.
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    pre_bert = tokenizer_bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(f"BERT (Podział na słowa i znaki):\n{pre_bert}\n")

    # --- PRZYKŁAD GPT-2 ---
    # GPT-2 używa ByteLevel. Spacje są zamieniane na znak 'Ġ', co pozwala
    # na bezstratne odtworzenie tekstu (tzw. lossless reconstruction).
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    pre_gpt2 = tokenizer_gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(f"GPT-2 (Spacje jako Ġ):\n{pre_gpt2}\n")

    # --- PRZYKŁAD T5 (SentencePiece) ---
    # T5 (używając Metaspace) traktuje wszystko jako ciąg znaków, a spację
    # zastępuje podkreślnikiem '_'. Nie dzieli agresywnie na interpunkcji.
    tokenizer_t5 = AutoTokenizer.from_pretrained("t5-small")
    pre_t5 = tokenizer_t5.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(f"T5 (SentencePiece / Metaspace):\n{pre_t5}\n")


def section_offsets_and_spaces():
    """
    SEKCJA 3: ANALIZA PRECYZJI (OFFSETY)
    Cel: Sprawdzenie, jak model zachowuje się w przypadku nietypowego formatowania.
    """
    print("--- 3. OBSŁUGA PODWÓJNYCH SPACJI ---")
    text_with_spaces = "Hello, how  are you?"  # Podwójna spacja między 'how' a 'are'

    # BERT: Często upraszcza białe znaki podczas normalizacji/pre-tokenizacji.
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("BERT (Jak reaguje na podwójną spację?):")
    print(tokenizer_bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text_with_spaces))

    # GPT-2: Jest ekstremalnie precyzyjny. Każda spacja jest dla niego istotna
    # i otrzyma własną reprezentację bajtową.
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    print("\nGPT-2 (Każda spacja jest osobnym tokenem):")
    print(tokenizer_gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text_with_spaces))


if __name__ == "__main__":
    # Startujemy symulację potoku tokenizacji
    section_normalization()
    section_pretokenization()
    section_offsets_and_spaces()

    print("\n--- PODSUMOWANIE ---")
    print("Zwróć uwagę na 'offsets' w wynikach - to one mówią modelowi,")
    print("gdzie w oryginalnym tekście znajdował się dany fragment.")