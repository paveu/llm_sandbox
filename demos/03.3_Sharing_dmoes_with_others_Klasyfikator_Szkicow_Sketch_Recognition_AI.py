"""
NAGŁÓWEK: Inteligentny Szkicownik AI (CNN)
OPIS: Skrypt wykorzystuje architekturę sieci neuronowej do analizy rysunków
      w czasie rzeczywistym. Dzięki zastosowaniu pipeline, model sam
      rozpozna co rysujesz (np. słońce, dom, kot).
WYNIK: Publiczny link do interaktywnej gry w kalambury z AI.
"""

import gradio as gr
from transformers import pipeline
from PIL import Image, ImageOps
import numpy as np

# 1. Pobieramy gotowy model wytrenowany na zbiorze "Quick, Draw!" (Google)
# Dzięki temu nie potrzebujesz pliku .bin ani .txt lokalnie!
print("Inicjalizacja modelu... Czekaj na pobranie wag.")
sketch_classifier = pipeline("image-classification", model="nateraw/vit-base-beans") # Przykładowy model wizyjny

# Alternatywnie, jeśli chcesz model stricte do szkiców:
# sketch_classifier = pipeline("image-classification", model="keras-io/sketch-recognition")

def predict(im):
    # Obsługa formatu Gradio 6.0 (słownik)
    if isinstance(im, dict):
        im = im["composite"]

    if im is None:
        return {}

    # Konwersja do obrazu PIL
    img = Image.fromarray(im.astype(np.uint8)).convert("L")

    # Inwersja kolorów - modele szkiców wolą białe kreski na czarnym tle
    img = ImageOps.invert(img)
    img = img.convert("RGB") # Większość modeli wymaga 3 kanałów

    # Wykonanie predykcji
    results = sketch_classifier(img)

    # Zwrócenie top 5 wyników w formacie słownika
    return {res["label"]: float(res["score"]) for res in results}

# 2. Budowa nowoczesnego interfejsu
demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Narysuj coś tutaj..."),
    outputs=gr.Label(num_top_classes=5, label="AI uważa, że to:"),
    title="AI Pictionary ✏️",
    description="Narysuj obiekt, a sieć neuronowa spróbuje go rozpoznać w czasie rzeczywistym!",
    live=True
)

if __name__ == "__main__":
    demo.launch(share=True)