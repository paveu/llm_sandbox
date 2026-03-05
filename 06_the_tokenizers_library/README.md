
# 🤖 Hugging Face Tokenizers Mastery

Ten projekt to kompletne kompendium wiedzy o potokach tokenizacji (Tokenization Pipelines) w ekosystemie 🤗 Transformers. Zawiera on implementacje różnych algorytmów, techniki trenowania od zera oraz praktyczne zastosowania w zadaniach NLP.

## 📁 Struktura Repozytorium

### 1. Specjalistyczny Trening (Training)
* **`01_training_a_specialized_code_tokenizer.py`**: Trenowanie tokenizera BPE na korpusie Python (CodeParrot). Implementuje wykorzystanie **generatorów (yield)**, co zapobiega przepełnieniu pamięci RAM przy dużych zbiorach danych.

### 2. Zaawansowane Funkcje (Fast Tokenizer Features)
* **`02_fast_tokenizers_special_powers.py`**: Wykorzystanie mapowania przesunięć (**offset_mapping**) w zadaniu NER. Obsługuje logikę grupowania tokenów typu subword przy użyciu etykiet B- i I-.
* **`03_fast_tokenizers_in_the_qa_pipeline.py`**: Implementacja strategii **Sliding Window** (przesuwne okno) z parametrem `stride`. Pozwala na ekstrakcję odpowiedzi z kontekstów przekraczających limit 512 tokenów.

### 3. Komponenty Potoku (Pipeline Components)
* **`04.normalization_and_pretokenization.py`**: Analiza różnic w podejściu do tekstu między modelami BERT (Whitespace + Punctuation) a GPT-2 (Byte-level).
* **`05_byte_pair_encoding_tokenization.py`**: Demonstracja etapów normalizacji (NFD, StripAccents) oraz pre-tokenizacji.

### 4. Głębia Algorytmów (Algorithm Deep Dives)
* **`05_bpe_logic.py`**: Implementacja łączenia najczęstszych par tokenów.
* **`06_wordpiece_tokenization.py`**: Symulacja algorytmu WordPiece z wykorzystaniem wzoru: $score = \frac{freq\_pary}{freq\_el1 \times freq\_el2}$.
* **`07_unigram_tokenization.py`**: Model probabilistyczny wybierający segmentację o najniższej stracie (loss).

### 5. Budowanie Blokowe (Block-by-Block)
Zbiór skryptów pokazujący, jak złożyć kompletny tokenizer z poszczególnych "klocków" biblioteki 🤗 Tokenizers:
* **`08_1_wordpiece_bert.py`**: Odtworzenie potoku BERT (Lowercase -> Whitespace -> WordPiece).
* **`08_2_bpe_gpt2.py`**: Bezzwrotna rekonstrukcja tekstu (lossless) przy użyciu Byte-level BPE.
* **`08_3_building_unigram_xlnet.py`**: Implementacja normalizacji SentencePiece i pre-tokenizacji Metaspace.

## 🚀 Kluczowe Koncepcje Wykorzystane w Kodzie

| Koncepcja | Opis |
| :--- | :--- |
| **Normalization** | Wstępne czyszczenie tekstu (NFD, NFKD, Lowercase) przed podziałem. |
| **Pre-tokenization** | Podział na słowa, zachowujący informację o spacjach (np. znak `Ġ` lub `_`). |
| **Post-processing** | Automatyczne dodawanie tokenów specjalnych jak `[CLS]` i `[SEP]`. |
| **Offset Mapping** | Powiązanie tokenów z ich pozycją w oryginalnym tekście znakowym. |

---
*Projekt zrealizowany w oparciu o Rozdział 6 kursu Hugging Face NLP.*