"""
Gradio Blocks Demo: Dynamic UI Property Updates (Total Compatibility Fix)

This version uses the dictionary return method. This is the most robust
way to handle UI updates in Gradio because it explicitly maps property
names to their new values.

Requirements:
    - gradio

Usage:
    Select an option from the Radio menu. The Textbox should immediately
    change its visibility, label, and height.
"""
import gradio as gr

def update_ui(choice):
    if choice == "Short form":
        # Returning a dictionary is the most stable way to update UI properties
        return {
            dynamic_input: gr.update(lines=2, visible=True, label="Write a short note")
        }
    elif choice == "Long form":
        return {
            dynamic_input: gr.update(lines=10, visible=True, label="Write a full essay")
        }
    else:
        return {
            dynamic_input: gr.update(visible=False)
        }

with gr.Blocks() as demo:
    gr.Markdown("# ⚡ Dynamic Interface")

    selector = gr.Radio(
        ["Short form", "Long form", "Hide everything"],
        label="What would you like to write?"
    )

    dynamic_input = gr.Textbox(visible=False)

    # Note: When returning a dictionary, we don't necessarily need to list
    # the output in the event listener, but it's good practice.
    selector.change(fn=update_ui, inputs=selector, outputs=[dynamic_input])

if __name__ == "__main__":
    demo.launch()