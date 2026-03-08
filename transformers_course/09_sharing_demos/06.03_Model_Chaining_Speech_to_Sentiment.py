"""
Gradio Blocks Demo: Speech-to-Sentiment Pipeline

This script illustrates 'Model Chaining.' It connects two separate
Hugging Face Transformers pipelines:
1. Automatic Speech Recognition (ASR) - wav2vec2
2. Sentiment Analysis - distilbert

The output of the first model (text) serves as the input for the second.

Requirements:
    - gradio
    - transformers
    - torch

Usage:
    1. Record audio or upload a .wav file.
    2. Click 'Recognize Speech' to generate text.
    3. Click 'Analyze Sentiment' to classify the generated text.
"""
import gradio as gr
from transformers import pipeline

print("Initializing AI models... this may take a moment.")
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def speech_to_text(audio):
    if audio is None: return ""
    result = asr(audio)
    return result["text"]


def analyze_sentiment(text):
    if not text: return "No text to analyze"
    result = classifier(text)
    return f"Sentiment: {result[0]['label']} (Score: {result[0]['score']:.2f})"


with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Speech AI Analyzer")

    audio_in = gr.Audio(type="filepath", label="Record Speech (English)")
    transcript_box = gr.Textbox(label="Transcribed Text")
    sentiment_label = gr.Label(label="Sentiment Analysis")

    with gr.Row():
        btn_asr = gr.Button("Step 1: Recognize Speech")
        btn_cls = gr.Button("Step 2: Analyze Sentiment")

    btn_asr.click(speech_to_text, inputs=audio_in, outputs=transcript_box)
    btn_cls.click(analyze_sentiment, inputs=transcript_box, outputs=sentiment_label)

if __name__ == "__main__":
    demo.launch()