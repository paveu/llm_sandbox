from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders

# TEORIA (Quiz): Etapy Potoku
# 1. NORMALIZACJA: Wstępne czyszczenie (np. Lowercase, NFKD).
# 2. PRE-TOKENIZACJA: Podział na słowa (np. na spacjach), zanim zadziała model subwordowy.

# ==============================================================================
# KROK 1: WYBÓR MODELU (BPE)
# Byte-Pair Encoding łączy najczęstsze pary bajtów w coraz większe jednostki.
# ==============================================================================
tokenizer = Tokenizer(models.BPE())

# ==============================================================================
# KROK 2: NORMALIZACJA
# GPT-2 NIE stosuje normalizacji. Chce zachować tekst dokładnie takim, jaki jest.
# ==============================================================================

# ==============================================================================
# KROK 3: PRE-TOKENIZACJA (BYTE-LEVEL)
# Spacje są zamieniane na specjalny znak 'Ġ'. Każdy bajt ma swoją reprezentację.
# ==============================================================================
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# ==============================================================================
# KROK 4: TRENING
# W GPT-2 głównym tokenem specjalnym jest znacznik końca tekstu.
# ==============================================================================
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
corpus = ["Let's train a Byte-level BPE tokenizer.", "GPT-2 uses this approach."]
tokenizer.train_from_iterator(corpus, trainer=trainer)

# ==============================================================================
# KROK 5: POST-PROCESSING
# ByteLevel post-processor dba o poprawne mapowanie offsetów (pozycji znaków).
# ==============================================================================
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# ==============================================================================
# KROK 6: DEKODOWANIE
# Zamienia 'Ġ' z powrotem na spacje i składa bajty w znaki Unicode.
# ==============================================================================
tokenizer.decoder = decoders.ByteLevel()

print("\n--- TEST GPT-2 (BPE) ---")
encoding = tokenizer.encode("Let's train BPE.")
print(f"Tokeny: {encoding.tokens}")