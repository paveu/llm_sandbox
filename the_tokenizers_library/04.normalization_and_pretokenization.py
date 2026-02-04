from transformers import AutoTokenizer


def section_normalization():
    print("--- 1. NORMALIZACJA ---")
    # Ładowanie tokenizatora BERT (uncased - zamienia na małe litery i usuwa akcenty)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Dostęp do backendu napisanego w Rust
    backend = tokenizer.backend_tokenizer

    raw_text = "Héllò hôw are ü?"
    normalized_text = backend.normalizer.normalize_str(raw_text)

    print(f"Tekst wejściowy: {raw_text}")
    print(f"Po normalizacji (BERT uncased): {normalized_text}")
    print("-" * 30)


def section_pretokenization():
    print("--- 2. PRETOKENIZACJA (RÓŻNE MODELE) ---")
    text = "Hello, how are you?"

    # --- PRZYKŁAD BERT ---
    # Dzieli na spacji i interpunkcji
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    pre_bert = tokenizer_bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(f"BERT Pre-tokenizacja:\n{pre_bert}\n")

    # --- PRZYKŁAD GPT-2 ---
    # Zachowuje spacje jako znak 'Ġ'
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    pre_gpt2 = tokenizer_gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(f"GPT-2 Pre-tokenizacja (zauważ symbol Ġ):\n{pre_gpt2}\n")

    # --- PRZYKŁAD T5 (SentencePiece) ---
    # Używa '_' dla spacji, dodaje spację na początku, ignoruje interpunkcję przy podziale
    tokenizer_t5 = AutoTokenizer.from_pretrained("t5-small")
    pre_t5 = tokenizer_t5.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(f"T5 Pre-tokenizacja (SentencePiece):\n{pre_t5}\n")


def section_offsets_and_spaces():
    print("--- 3. OBSŁUGA PODWÓJNYCH SPACJI ---")
    text_with_spaces = "Hello, how  are you?"  # podwójna spacja między how a are

    # BERT często "gubi" nadmiarowe spacje w offsetach
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("BERT (podwójna spacja):")
    print(tokenizer_bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text_with_spaces))

    # GPT-2 traktuje każdą spację jako osobny fragment
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    print("\nGPT-2 (podwójna spacja):")
    print(tokenizer_gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text_with_spaces))


if __name__ == "__main__":
    # Uruchomienie poszczególnych sekcji
    section_normalization()
    section_pretokenization()
    section_offsets_and_spaces()