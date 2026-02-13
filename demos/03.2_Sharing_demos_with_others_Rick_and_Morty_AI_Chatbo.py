"""
NAGŁÓWEK: Rick and Morty AI Chatbot
OPIS: Stylizowane demo Gradio, które symuluje rozmowę z Rickiem Sanchezem.
      Używa HTML do wyświetlania grafik i Markdownu do linkowania źródeł.
WYNIK: Interaktywny interfejs z motywem graficznym, gotowy do udostępnienia.
"""

import gradio as gr
from transformers import pipeline

# Inicjalizujemy prosty model konwersacyjny (np. Blenderbot od Facebooka)
# Możesz też użyć dowolnego modelu tekstowego z Hugging Face
chatbot = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

def predict(input_text):
    # Generujemy odpowiedź
    response = chatbot(input_text)[0]['generated_text']
    # Dodajemy nieco "Rickowego" stylu do odpowiedzi
    return f"{response}... *burp* Morty!"

# Stylizacja interfejsu (Twoje zmienne)
title = "Ask Rick a Question"
description = """
The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
<center><img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px></center>
"""

article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of."

# Definicja i uruchomienie
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Zadaj pytanie Rickowi", placeholder="What's the meaning of life?"),
    outputs=gr.Text(label="Rick mówi:"),
    title=title,
    description=description,
    article=article,
    examples=[["What are you doing?"], ["Where should we time travel to?"]],
)

if __name__ == "__main__":
    # Dodaję share=True, abyś mógł go wysłać znajomym!
    demo.launch(share=True)