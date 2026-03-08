"""
Skrypt do nadzorowanego dostrajania (Supervised Fine-Tuning - SFT) modelu językowego.

Przeznaczenie:
    - Pobranie danych konwersacyjnych ze zbioru 'smoltalk'.
    - Przekształcenie danych do formatu ChatML zrozumiałego dla modelu.
    - Przeprowadzenie procesu douczania modelu SmolLM2 na specyficznych przykładach.
    - Zapisanie finalnych wag modelu do użytku lokalnego.

Wymagania:
    - Biblioteki: datasets, transformers, trl, torch.
    - Plik tokenizer_config.json modelu musi wspierać szablony czatu.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# === 1. PREPROCESSING (Przygotowanie danych) ===

# Ładujemy 10 przykładowych rozmów z podzbioru 'everyday-conversations'
# Konfiguracja danych jest kluczowa dla jakości końcowych odpowiedzi modelu.
dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:10]")

# Definiujemy ID modelu bazowego, który będziemy dostrajać
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Ładujemy tokenizer, który odpowiada za zamianę tekstu na liczby (tokeny)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tokenizer potrzebuje 'pad_token', aby wyrównać długość różnych zdań w jednej partii (batch)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def convert_to_chatml(example):
    """Konwertuje listę wiadomości na ustrukturyzowany ciąg znaków ChatML."""
    messages = example["messages"]
    # apply_chat_template dodaje znaczniki typu <|im_start|> i <|im_end|>
    formatted_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # False, bo podczas treningu znamy już odpowiedź asystenta
    )
    return {"formatted_text": formatted_chat}

# Mapujemy funkcję na cały zbiór danych (tworzymy nową kolumnę 'formatted_text')
processed_dataset = dataset.map(convert_to_chatml)

# === 2. KONFIGURACJA I TRENING ===

# Ładujemy model (wagi wagowe). device_map="auto" automatycznie wykrywa GPU/CPU.
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Definiujemy parametry techniczne procesu uczenia
sft_config = SFTConfig(
    output_dir="moj_model_smollm",      # Katalog na tymczasowe punkty kontrolne (checkpoints)
    max_length=512,                      # Maksymalna długość kontekstu (chroni pamięć RAM/VRAM)
    learning_rate=2e-5,                  # Współczynnik uczenia - jak duże zmiany wprowadzać w wagach
    dataset_text_field="formatted_text", # Wskazujemy kolumnę z sformatowanym tekstem
    num_train_epochs=1,                  # Ile razy model ma przeczytać cały zbiór danych
    packing=False,                       # Czy łączyć krótkie teksty (oszczędność miejsca, ale trudniejszy trening)
    bf16=False,                          # Wyłączone dla kompatybilności ze starszym sprzętem/CPU
    push_to_hub=False,                   # Nie wysyłamy modelu na serwery Hugging Face
    report_to="none",                    # Wyłączamy raportowanie do zewnętrznych narzędzi (np. WandB)
)

# Inicjalizujemy Trenera, który połączy model, dane i konfigurację
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    args=sft_config,
)

if __name__ == "__main__":
    print("Rozpoczynam proces uczenia (SFT)...")
    # Główna pętla treningowa: obliczanie błędu (loss) i aktualizacja wag
    trainer.train()

    # Zapisujemy finalny model i tokenizer (ważne dla szablonów czatu!)
    trainer.save_model("./moj_model_smollm_final")
    print("Uczenie zakończone. Model zapisany w folderze './moj_model_smollm_final'")