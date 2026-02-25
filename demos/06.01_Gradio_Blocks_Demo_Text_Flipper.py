"""
Gradio Blocks Demo: Text and Image Flipper

This script demonstrates how to handle multiple data types (Text and Images)
within the Gradio Blocks API. It uses horizontal layout (Rows) and
demonstrates how buttons trigger specific processing functions.

Requirements:
    - gradio
    - numpy

Usage:
    1. Type in the text box and click 'Flip Text'.
    2. Upload an image and click 'Flip Image' to see it mirrored horizontally.
"""
import gradio as gr
import numpy as np

def flip_text(text):
    return text[::-1]

def flip_image(image):
    # np.fliplr flips a 3D array (image) along the left/right axis
    if image is None:
        return None
    return np.fliplr(image)

with gr.Blocks() as demo:
    gr.Markdown("# 🔄 The Ultimate Flipper")

    # Text Section
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", placeholder="Hello World")
            text_btn = gr.Button("Flip Text")
        with gr.Column():
            text_output = gr.Textbox(label="Reversed Text")

    gr.Markdown("---") # Visual separator

    # Image Section
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image")
            image_btn = gr.Button("Flip Image")
        with gr.Column():
            image_output = gr.Image(label="Mirrored Image")

    # Event Handlers
    text_btn.click(fn=flip_text, inputs=text_input, outputs=text_output)
    image_btn.click(fn=flip_image, inputs=image_input, outputs=image_output)

if __name__ == "__main__":
    demo.launch()