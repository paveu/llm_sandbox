import gradio as gr
from huggingface_hub import InferenceClient, whoami
import io
from PIL import Image

# --- KONFIGURACJA ---
HF_TOKEN = "x"

try:
    user = whoami(token=HF_TOKEN)
    print(f"✅ Zalogowano jako: {user['name']}")
except Exception as e:
    print(f"❌ BŁĄD AUTORYZACJI: {e}")

# Klient do wszystkiego
client = InferenceClient(token=HF_TOKEN)


# --- FUNKCJE ---

def chat_fn(message, history):
    try:
        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[{"role": "user", "content": message}],
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Błąd czatu: {e}"


def remove_background(image_path):
    """Bezpośrednie wywołanie modelu BriaAI przez Inference API."""
    if not image_path:
        return None
    try:
        print(f"🚀 Przesyłanie do BriaAI RMBG-1.4...")

        # Otwieramy obraz i zamieniamy na bajty
        with open(image_path, "rb") as f:
            img_data = f.read()

        # Wywołanie bezpośrednie modelu (zwraca surowe bajty obrazu PNG)
        response_bytes = client.post(
            model="briaai/RMBG-1.4",
            data=img_data
        )

        # Zamiana bajtów z powrotem na obraz PIL, aby Gradio mógł go wyświetlić
        result_img = Image.open(io.BytesIO(response_bytes))
        return result_img

    except Exception as e:
        print(f"❌ Błąd API: {e}")
        return None


def predict_gptj(text):
    if not text: return ""
    try:
        return client.text_generation(model="EleutherAI/gpt-j-6B", prompt=text, max_new_tokens=100)
    except Exception as e:
        return f"Błąd GPT-J: {e}"


# --- INTERFEJS ---

with gr.Blocks(title="AI Toolbox 2026") as demo:
    gr.Markdown("# 🚀 Hugging Face Direct API Hub")

    with gr.Tabs():
        with gr.Tab("💬 Czat"):
            gr.ChatInterface(fn=chat_fn)

        with gr.Tab("🖼️ Usuwanie Tła"):
            gr.Markdown("### 🪄 Model: BriaAI RMBG-1.4 (Direct API)")
            with gr.Row():
                img_in = gr.Image(type="filepath", label="Wejście")
                img_out = gr.Image(label="Wynik (PNG)")
            btn = gr.Button("USUŃ TŁO", variant="primary")
            btn.click(fn=remove_background, inputs=img_in, outputs=img_out)

        with gr.Tab("✍️ Tekst"):
            t_in = gr.Textbox(label="Prompt")
            t_out = gr.Textbox(label="Wynik")
            btn_g = gr.Button("Generuj")
            btn_g.click(fn=predict_gptj, inputs=t_in, outputs=t_out)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), debug=True)