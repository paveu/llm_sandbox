"""
NAGŁÓWEK: Audio Reverser - Aplikacja do odwracania dźwięku
OPIS: Skrypt uruchamia interfejs webowy Gradio, który pozwala użytkownikowi 
      nagrać dźwięk przez mikrofon, a następnie odwraca go za pomocą NumPy.
WYNIK: Użytkownik otrzymuje plik audio, który jest lustrzanym odbiciem 
       nagrania w domenie czasu (odtwarzanie od tyłu).
"""

import numpy as np
import gradio as gr


def reverse_audio(audio):
    # audio to krotka: (sample_rate, data)
    if audio is None:
        return None

    sr, data = audio

    # Odwracamy tablicę wzdłuż osi czasu
    # flipud działa dla 1D i 2D (stereo)
    reversed_data = np.flipud(data)

    return (sr, reversed_data)


# Definicja interfejsu
demo = gr.Interface(
    fn=reverse_audio,
    inputs=gr.Audio(sources=["microphone"], type="numpy", label="Nagraj coś..."),
    outputs="audio",
    title="Audio Reverser",
    description="Nagraj dźwięk, a ja go odtworzę od tyłu!"
)

if __name__ == "__main__":
    demo.launch()