"""
=============================================================================
WSTĘP: STRATEGIA SLIDING WINDOW (ROZDZIAŁ 6.3 - 7.1)
=============================================================================
Modele takie jak BERT mają ograniczoną 'szerokość spojrzenia' (Context Window).
Jeśli Twój tekst ma 2000 słów, model 'oślepi' wszystko powyżej limitu.

CO CHCEMY OSIĄGNĄĆ?
1. Podzielić długi tekst na mniejsze, nakładające się fragmenty (Features).
2. Przetworzyć każdy fragment przez sieć neuronową.
3. Przekonwertować surowe wyniki sieci (Logity) na czytelną odpowiedź.

TO JEST KLUCZ DO: Budowania chatbotów przeszukujących całe dokumenty PDF.
=============================================================================
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# --- KROK 1: ŁADOWANIE MODELU I TOKENIZATORA ---
# Cel: Pobranie 'mózgu' wyspecjalizowanego w szukaniu odpowiedzi (SQuAD).
print("--- Krok 1: Ładowanie komponentów ---")
model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

question = "Where does Sylvain work?"
context = """My name is Sylvain and I am a machine learning engineer at Hugging Face. 
Hugging Face is a company based in New York City. The headquarters are in DUMBO, Brooklyn. 
Sylvain has been working there since 2018 and enjoys building open-source tools for the community."""

# --- KROK 2: CIĘCIE TEKSTU (SLIDING WINDOW) ---
# Cel: Przetworzenie kontekstu w małych kawałkach, aby nic nie umknęło.
print("\n--- Krok 2: Tworzenie nakładających się fragmentów (Sliding Window) ---")

inputs = tokenizer(
    question,
    context,
    max_length=32,      # Wymuszamy małe okno, by zobaczyć mechanizm cięcia.
    truncation="only_second", # Tniemy tylko kontekst, pytanie musi zostać w całości.
    stride=10,          # 10 wspólnych tokenów między fragmentami (bezpiecznik kontekstu).
    padding=True,       # Wszystkie Tensory muszą mieć ten sam rozmiar.
    return_overflowing_tokens=True, # Tworzy dodatkowe obiekty dla reszty tekstu.
    return_offsets_mapping=True,    # Pamięta, skąd pochodzi dany token w tekście.
    return_tensors="pt"             # Zwraca format gotowy dla PyTorch.
)

print(f"Ilość wygenerowanych fragmentów (chunks): {len(inputs['input_ids'])}")
print(f"Kształt macierzy wejściowej: {inputs['input_ids'].shape}")



# --- KROK 3: PRZETWARZANIE PRZEZ SIEĆ NEURONOWĄ ---
# Cel: Uzyskanie logitów (surowych wyników) dla startu i końca odpowiedzi.
print("\n--- Krok 3: Praca modelu (Inference) ---")

# Usuwamy metadane, których model nie rozumie (tylko dla nas).
model_inputs = {
    k: v for k, v in inputs.items()
    if k not in ["offset_mapping", "overflow_to_sample_mapping"]
}

with torch.no_grad(): # Wyłączamy liczenie gradientów (oszczędność RAM/GPU).
    outputs = model(**model_inputs)

# Logity to 'pewność' modelu co do każdego tokena.
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# --- KROK 4: DEKODOWANIE ODPOWIEDZI ---
# Cel: Zamiana liczb z powrotem na tekst.
print("\n--- Krok 4: Szukanie odpowiedzi w fragmentach ---")

# Analizujemy fragment nr 0 (tam znajduje się odpowiedź).
sequence_idx = 0
# argmax wyciąga indeks tokena z najwyższym wynikiem.
start_token_pos = torch.argmax(start_logits[sequence_idx])
end_token_pos = torch.argmax(end_logits[sequence_idx])

# Pobieramy mapę przesunięć (offsets) dla tego fragmentu.
offsets = inputs["offset_mapping"][sequence_idx]

# Mapujemy indeks tokena na konkretne znaki w oryginalnym ciągu 'context'.
start_char = offsets[start_token_pos][0]
end_char = offsets[end_token_pos][1]

# Wycinamy odpowiedź bezpośrednio z tekstu źródłowego.
predicted_answer = context[start_char:end_char]

print(f"Pytanie: {question}")
print(f"Znaleziona odpowiedź: '{predicted_answer}'")
print(f"Pozycja w tekście: Znaki od {start_char} do {end_char}")

# --- PODSUMOWANIE I WSKAZÓWKA ---
print("\n--- WYNIK I WSKAZÓWKA ---")
print("W produkcyjnych systemach QA sprawdza się wszystkie fragmenty")
print("i wybiera ten, gdzie suma (start_logit + end_logit) jest najwyższa.")
print("To pozwala modelowi wybrać najbardziej wiarygodną odpowiedź z całego dokumentu.")