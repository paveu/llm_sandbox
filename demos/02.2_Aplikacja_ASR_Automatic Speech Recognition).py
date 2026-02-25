"""
O CZYM JEST TEN SKRYPT:
Aplikacja ASR (Automatic Speech Recognition) zbudowana przy użyciu modelu Facebook Wav2Vec2.
Umożliwia zamianę mowy na tekst w czasie rzeczywistym przy użyciu bibliotek Transformers i Gradio.

CZEGO SIĘ SPODZIEWAMY:
1. Po uruchomieniu, w terminalu pojawi się lokalny adres URL (np. http://127.0.0.1:7860).
2. W przeglądarce zobaczymy interfejs z widżetem audio (mikrofon/upload).
3. Po dostarczeniu dźwięku, model dokona transkrypcji, a wynik pojawi się w polu tekstowym.
"""

import gradio as gr
from transformers import pipeline

# 1. Ładowanie konkretnego modelu (dobra praktyka, unika ostrzeżeń i niepewności)
print("Ładowanie modelu AI (Wav2Vec2)... Proszę czekać.")
asr_pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe(audio_file):
    if audio_file is None:
        return "Brak nagrania. Proszę użyć mikrofonu lub wgrać plik."

    # Przetwarzanie audio - model przyjmuje ścieżkę do pliku i zwraca tekst
    result = asr_pipe(audio_file)
    return result["text"]

# 2. Definicja interfejsu (Opcja 1: usunięto show_copy_button dla kompatybilności)
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Twój głos (mikrofon lub plik)"),
    outputs=gr.Textbox(label="Rozpoznany tekst"),
    title="Lokalny Transkryptor Głosowy",
    description="Kliknij przycisk nagrywania, powiedz coś, a następnie kliknij 'Submit'."
)

if __name__ == "__main__":
    # 3. Inicjalizacja serwera lokalnego
    demo.launch()