"""
NAGŁÓWEK: Polski Transkrypter AI (Whisper)
OPIS: Skrypt wykorzystuje model OpenAI Whisper zoptymalizowany pod kątem
      języka polskiego do precyzyjnej zamiany mowy na tekst.
WYNIK: Publiczny link do polskiego systemu Speech-to-Text.
"""

from transformers import pipeline
import gradio as gr

# 1. Zmiana modelu na Whisper (wersja 'tiny' lub 'base' dla szybkości, 'small' dla jakości)
# 'openai/whisper-small' świetnie radzi sobie z polską gramatyką
model_id = "openai/whisper-small"

print(f"Ładowanie modelu {model_id}... To może chwilę potrwać.")
asr_model = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,  # Pozwala na przetwarzanie długich nagrań
)


def transcribe_audio(audio_path):
    if audio_path is None:
        return "Proszę nagrać dźwięk..."

    # Przetwarzanie z wymuszeniem języka polskiego
    result = asr_model(audio_path, generate_kwargs={"language": "polish"})
    return result["text"]


# 2. Interfejs Gradio
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Mów po polsku..."),
    outputs=gr.Textbox(label="Wynik transkrypcji"),
    title="Polski Transkrypter AI 🇵🇱",
    description="Ten model najlepiej radzi sobie z językiem polskim. Spróbuj powiedzieć coś z polskimi znakami!"
)

if __name__ == "__main__":
    demo.launch(share=True)