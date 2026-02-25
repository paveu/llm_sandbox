"""
O CZYM JEST TEN SKRYPT:
Skrypt demonstruje praktyczne zastosowanie klasy `gr.Interface` z rozdziału 9.3.
Tworzy on interaktywną aplikację webową do analizy sentymentu tekstu, wykorzystując
gotowy model z biblioteki Hugging Face Transformers.

CZEGO SIĘ SPODZIEWAMY:
1. Po uruchomieniu skryptu (i pobraniu modelu), w terminalu pojawi się lokalny adres URL (zazwyczaj http://127.0.0.1:7860).
2. W przeglądarce otworzy się interfejs z dwoma panelami wejściowymi: polem tekstowym oraz suwakiem.
3. Po wpisaniu tekstu i kliknięciu "Submit", aplikacja zwróci dwa wyniki: czytelną etykietę (Label)
   z dominującym emocjonalnym wydźwiękiem oraz szczegółowy obiekt JSON z wynikami powyżej progu z suwaka.
"""

import gradio as gr
from transformers import pipeline

# 1. Inicjalizacja modelu (Analiza sentymentu)
classifier = pipeline("sentiment-analysis", return_all_scores=True)


# 2. Definicja funkcji predykcyjnej
# Zwróć uwagę, że funkcja przyjmuje dwa argumenty, bo mamy dwa komponenty wejściowe
def analyze_text(text, threshold):
    if not text:
        return "Brak tekstu", {}

    results = classifier(text)[0]

    # Filtrowanie wyników na podstawie suwaka (threshold)
    filtered_results = {res["label"]: res["score"] for res in results if res["score"] >= threshold}

    # Najlepszy wynik dla komponentu Label
    top_label = max(results, key=lambda x: x["score"])["label"]

    return top_label, filtered_results


# 3. Konfiguracja zaawansowanych komponentów
input_text = gr.Textbox(
    label="Wpisz tekst do analizy",
    placeholder="To jest niesamowity kurs!",
    lines=3
)

input_slider = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=0.5,
    label="Próg pewności (Threshold)",
    info="Pokaż wyniki tylko powyżej tej wartości"
)

output_label = gr.Label(label="Główny Sentyment")
output_json = gr.JSON(label="Pełne dane (JSON)")

# 4. Budowa interfejsu
demo = gr.Interface(
    fn=analyze_text,
    inputs=[input_text, input_slider],  # Lista wielu wejść
    outputs=[output_label, output_json],  # Lista wielu wyjść
    title="Analizator Sentymentu Hugging Face",
    description="Wpisz zdanie, aby sprawdzić, czy ma wydźwięk pozytywny, czy negatywny. Możesz sterować czułością za pomocą suwaka.",
    examples=[
        ["I love learning about AI!", 0.1],
        ["This error is very frustrating.", 0.8],
        ["Gradio makes building apps so easy!", 0.5]
    ],
    cache_examples=True  # Przyspiesza działanie przykładów
)

# 5. Uruchomienie aplikacji
if __name__ == "__main__":
    # share=True wygeneruje publiczny link działający przez 72h
    demo.launch(share=False)