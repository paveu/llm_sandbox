"""
=============================================================================
WSTĘP: NAMED ENTITY RECOGNITION (NER) I MAPOWANIE (ROZDZIAŁ 6.3)
=============================================================================
W zadaniach rozpoznawania encji (NER) model musi wskazać, które fragmenty tekstu
to imiona, nazwy firm czy lokalizacje.

WYZWANIE: Modele często dzielą słowa na fragmenty (subwordy). 'Sylvain' może
stać się czterema tokenami. Jak wiedzieć, że wszystkie cztery to jedno imię?

ROZWIĄZANIE: Fast Tokenizers dostarczają 'offset_mapping' – mapę, która wiąże
indeks tokena z konkretnymi pozycjami znaków w surowym tekście.
=============================================================================
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- KROK 1: ŁADOWANIE MODELU I TOKENIZATORA ---
print("--- Krok 1: Ładowanie 'mózgu' (Checkpoint) ---")
# Model BERT-large wytrenowany na zbiorze CoNLL-03 (PER, ORG, LOC, MISC).
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"

# Używamy AutoTokenizer, który domyślnie ładuje wersję 'Fast' (napisaną w Rust).
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
print(f"Tekst wejściowy: {example}\n")

# --- KROK 2: WERYFIKACJA MOŻLIWOŚCI ---
# Tylko tokenizatory 'Fast' obsługują offset_mapping, który jest tu kluczowy.
print(f"Czy to 'Fast' Tokenizer? {tokenizer.is_fast}")



# --- KROK 3: KODOWANIE Z MAPOWANIEM PRZESUNIĘĆ ---
print("\n--- Krok 3: Kodowanie i tworzenie 'Mapy Przesunięć' ---")
# return_offsets_mapping=True: To nasza 'Supermoc'. Pamięta pozycje znaków dla każdego tokena.
inputs = tokenizer(example, return_offsets_mapping=True, return_tensors="pt")

offsets = inputs["offset_mapping"][0]
tokens = inputs.tokens()

print(f"Utworzone tokeny: {tokens}")
print(f"Word IDs (które tokeny to to samo słowo): {inputs.word_ids()}")

# --- KROK 4: INFERENCJA (PROGNOZOWANIE) ---
print("\n--- Krok 4: Predykcja modelu (Inference) ---")
# Usuwamy offsety z wejścia do modelu (model ich nie potrzebuje, służą tylko nam).
model_inputs = {k: v for k, v in inputs.items() if k != "offset_mapping"}

with torch.no_grad():
    outputs = model(**model_inputs)

# Wybieramy etykietę z najwyższym wynikiem dla każdego tokena (argmax).
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
# Zamieniamy wyniki na prawdopodobieństwa (pewność modelu).
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

# --- KROK 5: REKONSTRUKCJA I GRUPOWANIE ---
# Cel: Złączyć '##yl', '##va' w 'Sylvain' i nadać im jedną etykietę PER.
print("\n--- Krok 5: Składanie fragmentów w całe encje ---")



results = []
idx = 0

while idx < len(predictions):
    pred_idx = predictions[idx]
    label = model.config.id2label[pred_idx] # Mapujemy ID (np. 4) na nazwę (np. 'B-PER')

    if label != "O": # 'O' oznacza brak encji (zwykłe słowo)
        entity_type = label[2:] # Usuwamy prefix B- lub I-, zostaje 'PER'
        start, _ = offsets[idx]

        print(f"Wykryto fragment: '{tokens[idx]}' -> {label}")

        # Logika grupowania sąsiadujących tokenów tej samej encji
        all_scores = []
        while (idx < len(predictions) and
               model.config.id2label[predictions[idx]].endswith(entity_type)):
            all_scores.append(probabilities[idx][predictions[idx]])
            _, end = offsets[idx]
            idx += 1

        # UŻYCIE OFFSETS: Wycinamy finalne słowo bezpośrednio z oryginału.
        # Dzięki temu nie musimy martwić się o usuwanie '##' czy spacje.
        final_word = example[start:end]

        results.append({
            "entity_group": entity_type,
            "confidence": np.mean(all_scores).item(),
            "word": final_word,
            "start": start.item(),
            "end": end.item(),
        })
    else:
        idx += 1

# --- FINALNE WYNIKI ---
print("\n--- Finalne Wyniki (Zgrupowane Encje) ---")
import pprint
pprint.pprint(results)