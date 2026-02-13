"""
NAGŁÓWEK: Transkrypcja Audio na Tekst (Speech-to-Text AI)
OPIS: Skrypt wykorzystuje model Deep Learning (domyślnie Wav2Vec2) do automatycznego
      rozpoznawania mowy. Zamienia plik dźwiękowy bezpośrednio na tekst pisany.
WYNIK: Interfejs, w którym po wgraniu pliku audio lub nagraniu mowy,
       otrzymasz gotową transkrypcję w formie tekstowej.
"""

from transformers import pipeline
import gradio as gr

# Inicjalizacja modelu ASR (Automatic Speech Recognition)
# Przy pierwszym uruchomieniu skrypt pobierze wagę modelu z Hugging Face
asr_model = pipeline("automatic-speech-recognition")


def transcribe_audio(audio_path):
    if audio_path is None:
        return "Proszę nagrać lub przesłać plik audio."

    # Przetwarzanie mowy przez sieć neuronową
    result = asr_model(audio_path)
    return result["text"]


# Budowa interfejsu
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Wgraj plik lub nagraj mowę"),
    outputs=gr.Textbox(label="Rozpoznany tekst"),
    title="AI Speech Transcriber 🤖",
    description="Powiedz coś, a sztuczna inteligencja zamieni to na tekst."
)

if __name__ == "__main__":
    demo.launch()