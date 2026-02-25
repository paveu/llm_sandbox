"""
Gradio Blocks Demo: Multi-Tool Tabs & Rows

This script demonstrates advanced layout management in Gradio. It uses
`gr.Tabs` for navigation and `gr.Row` for horizontal positioning. It also
shows how to handle different data types (Text and Images) within a single app.

Requirements:
    - gradio
    - numpy

Usage:
    Navigate between the 'Text Lab' and 'Image Lab' tabs to process data.
"""
import numpy as np
import gradio as gr


def flip_text(x):
    return x[::-1]


def flip_image(image):
    return np.fliplr(image)


with gr.Blocks() as demo:
    gr.Markdown("# 🛠️ Multi-Tool Utility")

    with gr.Tabs():
        with gr.TabItem("Text Lab"):
            with gr.Row():
                t_input = gr.Textbox(label="Input")
                t_output = gr.Textbox(label="Output")
            t_button = gr.Button("Flip Text")

        with gr.TabItem("Image Lab"):
            with gr.Row():
                i_input = gr.Image(label="Upload Image")
                i_output = gr.Image(label="Result")
            i_button = gr.Button("Flip Image")

    t_button.click(flip_text, inputs=t_input, outputs=t_output)
    i_button.click(flip_image, inputs=i_input, outputs=i_output)

if __name__ == "__main__":
    demo.launch()