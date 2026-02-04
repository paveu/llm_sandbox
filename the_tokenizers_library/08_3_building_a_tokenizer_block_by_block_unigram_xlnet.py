from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders

# TEORIA (Quiz): Etapy Potoku
# 1. NORMALIZACJA: Wstępne czyszczenie (np. Lowercase, NFKD).
# 2. PRE-TOKENIZACJA: Podział na słowa (np. na spacjach), zanim zadziała model subwordowy.

# ==============================================================================
# KROK 1: WYBÓR MODELU (UNIGRAM)
# Unigram to model probabilistyczny, który ocenia "stratę" przy usuwaniu tokenów.
# ==============================================================================
tokenizer = Tokenizer(models.Unigram())

# ==============================================================================
# KROK 2: NORMALIZACJA (W STYLU SENTENCEPIECE)
# XLNet dba o ujednolicenie spacji i normalizację wizualną znaków.
# ==============================================================================
tokenizer.normalizer = normalizers.Sequence([
    normalizers.Replace(" {2,}", " "), # Redukcja wielu spacji do jednej
    normalizers.NFKD(),                # Normalizacja Unicode
    normalizers.StripAccents()         # Usunięcie akcentów
])

# ==============================================================================
# KROK 3: PRE-TOKENIZACJA (METASPACE)
# Zamiast dzielić na spacjach, zamienia spację na znak ' ' (U+2581).
# ==============================================================================
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# ==============================================================================
# KROK 4: TRENING
# UnigramTrainer wymaga jawnego wskazania tokena nieznanego (unk_token).
# ==============================================================================
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000,
    special_tokens=special_tokens,
    unk_token="<unk>"
)
corpus = ["Unigram is a probabilistic algorithm.", "XLNet uses Metaspace."]
tokenizer.train_from_iterator(corpus, trainer=trainer)

# ==============================================================================
# KROK 5: POST-PROCESSING
# XLNet umieszcza <cls> na końcu sekwencji (tzw. padding z lewej strony).
# ==============================================================================
cls_id = tokenizer.token_to_id("<cls>")
sep_id = tokenizer.token_to_id("<sep>")

tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_id), ("<cls>", cls_id)],
)

# ==============================================================================
# KROK 6: DEKODOWANIE
# Usuwa znak ' ' i przywraca standardowe spacje.
# ==============================================================================
tokenizer.decoder = decoders.Metaspace()

print("\n--- TEST XLNet (Unigram) ---")
encoding = tokenizer.encode("Unigram test.")
print(f"Tokeny: {encoding.tokens}")
print(f"Etykiety typu (Type IDs): {encoding.type_ids}")