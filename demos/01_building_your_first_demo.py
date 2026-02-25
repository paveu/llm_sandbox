import gradio as gr
from transformers import pipeline

# 1. Load the text-generation model (GPT-2 by default)
model = pipeline("text-generation")

# 2. Define the prediction function
def predict(prompt):
    # The pipeline returns a list of dictionaries; we extract the generated text
    completion = model(prompt)[0]["generated_text"]
    return completion

# 3. Create a customized Interface
# We use the Textbox class to add a label and placeholder
input_component = gr.Textbox(label="Enter your prompt:", placeholder="Once upon a time...")

demo = gr.Interface(
    fn=predict,
    inputs=input_component,
    outputs="text",
    title="GPT-2 Text Generator",
    description="Type something and let the model complete your sentence!"
)

# 4. Launch the app
if __name__ == "__main__":
    demo.launch()