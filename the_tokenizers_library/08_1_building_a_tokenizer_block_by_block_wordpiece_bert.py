from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders

# ==============================================================================
# KROK 1: WYBÓR MODELU (ALBORYTMU RDZENNEGO)
# WordPiece szuka subwordów i oznacza kontynuację słowa prefiksem "##".
# ==============================================================================
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# ==============================================================================
# KROK 2: NORMALIZACJA (CZYSZCZENIE TEKSTU)
# BERT-uncased zamienia wszystko na małe litery i usuwa akcenty (np. 'ó' -> 'o').
# ==============================================================================
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),           # Rozkład znaków Unicode
    normalizers.Lowercase(),     # Zamiana na małe litery
    normalizers.StripAccents()   # Usunięcie znaczników akcentów
])

# ==============================================================================
# KROK 3: PRE-TOKENIZACJA (WSTĘPNY PODZIAŁ)
# Dzielimy tekst na słowa, używając spacji oraz znaków interpunkcyjnych.
# ==============================================================================
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# ==============================================================================
# KROK 4: TRENING (UCZENIE NA DANYCH)
# Tokenizer analizuje korpus i buduje słownik o zadanym rozmiarze.
# ==============================================================================
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=30000, special_tokens=special_tokens)

# Przykładowy korpus treningowy
corpus = ["Hugging Face is building a great library.", "WordPiece is used by BERT."]
tokenizer.train_from_iterator(corpus, trainer=trainer)

# ==============================================================================
# KROK 5: POST-PROCESSING (TOKENY SPECJALNE)
# BERT wymaga dodania [CLS] na początku i [SEP] na końcu każdej sekwencji.
# ==============================================================================
cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
)

# ==============================================================================
# KROK 6: DEKODOWANIE (TEKST Z TOKENÓW)
# Mówimy dekoderowi, że "##" oznacza brak spacji przed tym fragmentem.
# ==============================================================================
tokenizer.decoder = decoders.WordPiece(prefix="##")

print("--- TEST BERT (WordPiece) ---")
encoding = tokenizer.encode("Hugging Face is building.")
print(f"Tokeny: {encoding.tokens}")