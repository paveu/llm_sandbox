"""
PROJEKT: Zaawansowany Interfejs Gradio 6.5.1 - Demo Funkcjonalności
==================================================================

OPIS DZIAŁANIA:
---------------
Skrypt uruchamia lokalną aplikację webową opartą na bibliotece Gradio,
która demonstruje wykorzystanie nowoczesnych komponentów 'Blocks'
oraz integrację z modelem uczenia maszynowego (Computer Vision).

GŁÓWNE FUNKCJE:
1. KARTA "🤖 Chatbot":
   - Wykorzystuje komponent ChatInterface do prowadzenia konwersacji.
   - Automatycznie zarządza historią rozmowy (stanem sesji).
   - Zwraca losowe odpowiedzi tekstowe w ramach logiki symulacyjnej.

2. KARTA "🖼️ Klasyfikacja":
   - Wykorzystuje gotowy model MobileNetV2 (TensorFlow/Keras).
   - Przetwarza obrazy wejściowe (resize, preprocess) i klasyfikuje je
     do 1000 kategorii ImageNet.
   - Wyświetla 3 najbardziej prawdopodobne wyniki wraz z poziomem pewności.

OCZEKIWANE ZACHOWANIE:
----------------------
- OPTYMALIZACJA: Skrypt wymusza użycie CPU (os.environ["CUDA_VISIBLE_DEVICES"] = "-1"),
  co zapobiega błędom pamięci na słabszych konfiguracjach GPU.
- POBIERANIE DANYCH: Przy pierwszym uruchomieniu zostaną pobrane wagi modelu
  oraz plik etykiet 'ImageNetLabels.txt'.
- INTERFEJS: Po uruchomieniu w konsoli pojawi się adres (np. http://127.0.0.1:7860),
  pod którym dostępny jest graficzny interfejs użytkownika.

WYMAGANIA:
----------
pip install gradio tensorflow pillow numpy requests
"""
import os

# Wymuszenie CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import random

# --- LOGIKA MODELU OBRAZU ---
model = tf.keras.applications.MobileNetV2(weights="imagenet")


def classify_image(img):
    if img is None:
        return None
    # Preprocessing dla MobileNetV2
    img_pil = Image.fromarray(img).resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_pil)[np.newaxis, ...])

    prediction = model.predict(img_array).flatten()

    # Pobranie etykiet ImageNet
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    with open(labels_path) as f:
        labels = f.read().splitlines()[1:]

    return {labels[i]: float(prediction[i]) for i in range(1000)}


# --- LOGIKA CZATU (Gradio 6) ---
def chat_logic(message, history):
    # W Gradio 6 nie musisz ręcznie modyfikować historii w ChatInterface,
    # wystarczy zwrócić tekst odpowiedzi.
    responses = [
        "To fascynujące!",
        "Opowiedz mi o tym coś więcej.",
        "Rozumiem, kontynuuj.",
        "Ciekawy punkt widzenia."
    ]
    return random.choice(responses)


# --- BUDOWA APLIKACJI ---

with gr.Blocks(title="Gradio 6 Modern Demo") as demo:
    gr.Markdown("# 🚀 Zaawansowane Funkcje (Gradio 6.5.1)")

    with gr.Tab("🤖 Chatbot"):
        gr.Markdown("Dedykowany interfejs czatu z automatycznym zarządzaniem stanem.")
        # Usunięto parametr 'type', który powodował błąd
        gr.ChatInterface(fn=chat_logic)

    with gr.Tab("🖼️ Klasyfikacja"):
        gr.Markdown("Klasyfikator obrazów MobileNetV2 działający na CPU.")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Wgraj obraz")
                btn = gr.Button("Klasyfikuj", variant="primary")
            with gr.Column():
                output_label = gr.Label(num_top_classes=3, label="Wynik")

        # Powiązanie przycisku z funkcją
        btn.click(fn=classify_image, inputs=input_img, outputs=output_label)

if __name__ == "__main__":
    print("🚀 Serwer startuje na Python 3.13 + Gradio 6.5.1...")
    demo.launch()