"""
NAGŁÓWEK: Generator Tonów Muzycznych (Mathematical Synthesizer)
OPIS: Skrypt wykorzystuje matematyczną zależność między nutami a częstotliwością (skala równomiernie temperowana),
      aby wygenerować falę sinusoidalną odpowiadającą konkretnej nucie i oktawie.
WYNIK: Interfejs Gradio z suwakami, który po wybraniu nuty generuje i pozwala odsłuchać czysty dźwięk (beep)
       o zadanej długości w formacie 16-bit PCM.
"""

import numpy as np
import gradio as gr

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def generate_tone(note_index, octave, duration):
    sr = 48000
    # Obliczanie częstotliwości f = 440 * 2^(n/12)
    a4_freq = 440
    tones_from_a4 = 12 * (octave - 4) + (note_index - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)

    # Generowanie fali dźwiękowej
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 20000 to amplituda (głośność), np.sin tworzy oscylację fali
    audio = (20000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

    return (sr, audio)


# Budowa interfejsu
demo = gr.Interface(
    fn=generate_tone,
    inputs=[
        gr.Dropdown(notes, type="index", label="Wybierz nutę"),
        gr.Slider(minimum=2, maximum=7, step=1, value=4, label="Oktawa"),
        gr.Number(value=1, label="Czas trwania (sekundy)"),
    ],
    outputs="audio",
    title="Python Synthesizer 🎹",
    description="Wybierz nutę i oktawę, aby wygenerować czysty ton sinusowy."
)

if __name__ == "__main__":
    demo.launch()