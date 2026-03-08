"""
SKRYPT TESTOWY (INFERENCE): ROZMOWA Z TWOIM MODELEM REASONING
---------------------------------------------------------------------------
CO ROBI TEN SKRYPT?
1. Ładuje Twój nowo wytrenowany model z folderu 'grpo-reasoning-model'.
2. Zadaje mu pytanie, którego model NIE WIDZIAŁ podczas treningu.
3. Pozwala modelowi na "wyprodukowanie" procesu myślowego wewnątrz <think>.

SPODZIEWANY EFEKT:
Model powinien zacząć odpowiedź od <think>, wypisać kroki logiczne,
a na końcu podać wynik w <answer>.
---------------------------------------------------------------------------
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Ścieżka do Twojego wytrenowanego modelu
MODEL_PATH = "grpo-reasoning-model"

print(f"--- ŁADOWANIE MODELU Z: {MODEL_PATH} ---")

# Ładowanie tokenizera i modelu (automatycznie wykrywa urządzenie GPU/CPU)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto", # Użyj precyzji, w której model został zapisany
    device_map="auto"   # Automatycznie umieść model na karcie graficznej (jeśli dostępna)
)

# Tworzymy 'pipeline' - to najwygodniejszy sposób na generowanie tekstu w Hugging Face
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Przygotowujemy nową zagadkę (spoza zbioru GSM8K)
# Chcemy sprawdzić, czy model "pomyśli" przed odpowiedzią.
test_question = "If I have 5 apples and I give 2 to Mary and 1 to John, how many apples do I have left?"

# Formatujemy zapytanie tak samo jak podczas treningu (rola systemowa jest kluczowa!)
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. You must first think about the problem inside <think> tags, and then provide the final answer inside <answer> tags."
    },
    {
        "role": "user",
        "content": test_question
    }
]

print(f"\n--- ZADAJĘ PYTANIE: {test_question} ---")

# Generujemy odpowiedź
# max_new_tokens: limit długości (myślenie zajmuje sporo miejsca!)
# temperature: 0.7 pozwala na odrobinę kreatywności w myśleniu
output = text_generator(
    messages,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    return_full_text=False # Chcemy zobaczyć tylko nową odpowiedź modelu
)

# Wyświetlamy wynik
print("\n--- ODPOWIEDŹ MODELU ---")
print(output[0]['generated_text'])

print("\n--- KONIEC TESTU ---")