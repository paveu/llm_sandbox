from transformers import pipeline
import os

# ==============================================================================
# 1. KONFIGURACJA ŚCIEŻKI
# ==============================================================================
# Kropka na początku oznacza "bieżący folder".
# Skrypt szuka podfolderu, który stworzyłeś poleceniem trainer.save_model().
model_path = "./moj_inteligentny_model"

if not os.path.exists(model_path):
    print(f"❌ BŁĄD: Nie widzę folderu {model_path}!")
    print(f"Obecny katalog roboczy to: {os.getcwd()}")
    print("Upewnij się, że uruchamiasz skrypt z folderu, w którym jest model.")
    exit()

# ==============================================================================
# 2. WCZYTYWANIE MODELU I TOKENIZERA
# ==============================================================================
print(f"🔄 Wczytywanie Twojego douczonego modelu z {model_path}...")

# pipeline to najwyższy poziom abstrakcji.
# Automatycznie ładuje model.safetensors (wagi) oraz tokenizer_config.json (słownik).
# device=-1 wymusza użycie procesora (CPU), co zapobiega błędom na Twoim komputerze.
classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    device=-1
)

print("\n✅ Model gotowy! System ustawiony na CPU.")
print("--- TESTER PARAFRAZY (Zadanie MRPC) ---")
print("Wpisz dwa zdania, by sprawdzić, czy model uzna je za synonimy.")

# ==============================================================================
# 3. PĘTLA INTERAKTYWNA
# ==============================================================================
while True:
    z1 = input("\nZdanie 1 (lub 'q' aby wyjść): ")
    if z1.lower() == 'q':
        break

    z2 = input("Zdanie 2: ")

    # Przekazujemy parę zdań jako słownik.
    # To ważne, bo model BERT był trenowany na parach (Sentence A i Sentence B).
    wynik = classifier({"text": z1, "text_pair": z2})

    # --- POPRAWKA BŁĘDU KEYERROR ---
    # Niektóre wersje pipeline zwracają listę [{...}], a inne sam słownik {...}.
    # Ten kod obsługuje oba przypadki:
    if isinstance(wynik, list):
        wynik_dict = wynik[0]
    else:
        wynik_dict = wynik

    label = wynik_dict['label']
    score = wynik_dict['score']

    # LABEL_1: Zdania znaczą to samo (Parafraza)
    # LABEL_0: Zdania są o czymś innym (Różne)
    if label == "LABEL_1":
        status = "✅ To jest PARAFRAZA (to samo znaczenie)"
    else:
        status = "❌ To są RÓŻNE zdania"

    print(f"WYNIK: {status}")
    print(f"Pewność modelu: {score:.2%}")

print("\nZamykanie testera. Powodzenia w dalszej nauce LLM!")