"""
Skrypt do testowania (inferencji) lokalnie dostrojonego modelu językowego.

Przeznaczenie:
    - Załadowanie modelu i tokenizera z lokalnego katalogu.
    - Sformatowanie zapytania użytkownika przy użyciu szablonów czatu.
    - Wygenerowanie odpowiedzi przez model (proces autoregresyjny).
    - Dekodowanie wyników do czytelnej postaci tekstowej.

Uwagi:
    - Skrypt wykorzystuje 'add_generation_prompt=True', aby wymusić na modelu start odpowiedzi.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Definicja ścieżki do folderu utworzonego przez skrypt treningowy
model_path = "./moj_model_smollm_final"

print("Ładowanie modelu z dysku...")
# Ładujemy model i tokenizer z lokalnych plików
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Przygotowanie struktury rozmowy
# System prompt ustawia zachowanie modelu, user zawiera Twoje pytanie.
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I'm looking for a beach resort in Poland. Any recommendations?"},
]

# Konwersja listy na format ChatML (dodaje np. <|im_start|>assistant na końcu)
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True # Kluczowe: model wie, że teraz ma pisać jako asystent
)

# 3. Tokenizacja - zamiana tekstu na format zrozumiały dla PyTorch (Tensory)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("Asystent myśli...")
# 4. Generowanie odpowiedzi przez model
outputs = model.generate(
    **inputs,
    max_new_tokens=100, # Maksymalna długość nowej odpowiedzi
    do_sample=True,      # Włącza losowanie słów (odpowiedź będzie bardziej naturalna)
    temperature=0.7,     # Kontroluje "kreatywność" (niższa = bardziej przewidywalna)
    top_p=0.9            # Technika Nucleus Sampling dla lepszej jakości tekstu
)

# 5. Wycinanie promptu wejściowego z wyniku i dekodowanie samej odpowiedzi
# outputs zawiera (prompt + nowa_odpowiedź), dlatego bierzemy tylko nowe tokeny.
generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(f"\nOdpowiedź modelu:\n{response}")