"""
Skrypt do Supervised Fine-Tuning (SFT) na bazie rozdziału o bibliotece TRL.

PRZEZNACZENIE I ZASTOSOWANIE:
    - Transformacja modeli bazowych w modele typu 'assistant' zdolne do prowadzenia dialogu.
    - Stosowane, gdy samo promptowanie (well-crafted prompts) okazuje się niewystarczające.
    - Umożliwia precyzyjną kontrolę nad szablonem (Template Control) oraz adaptację
      do specyficznych dziedzin (Domain Adaptation).

DANE I FORMATOWANIE:
    - Wymaga zestawu danych z parami input-output (prompt-reakcja).
    - SFTTrainer automatycznie aplikuje szablony czatu dla pól 'messages' pobranych z Hub-a.
    - Packing: Optymalizuje wydajność poprzez łączenie krótkich przykładów w jedną sekwencję.

MONITOROWANIE I ZBIEŻNOŚĆ (CONVERGENCE):
    - Healthy Training: Charakteryzuje się małą luką między stratą treningową a walidacyjną.
    - Fazy Loss: Gwałtowny spadek (adaptacja), stopniowa stabilizacja, konwergencja.
    - Ostrzeżenia: Wzrost straty walidacyjnej przy spadku treningowej sugeruje overfitting (przeuczenie).
    - Monitoring: Należy śledzić zarówno metryki ilościowe (loss), jak i jakościowe (treść odpowiedzi).
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 1. Konfiguracja urządzenia i modelu
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 2. Dataset Preparation
# SFT wymaga specyficznych danych zadaniowych.
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# 3. Ładowanie modelu i tokenizera
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Training Configuration
training_args = SFTConfig(
    output_dir="./deepseek_sft_output",

    # Training Duration: num_train_epochs lub max_steps.
    # Więcej epok to lepsza nauka, ale ryzyko overfittingu.
    max_steps=500,

    # Batch Size: per_device_train_batch_size określa stabilność i użycie pamięci.
    per_device_train_batch_size=2,
    # Gradient Accumulation: Zwiększa efektywny batch size bez dodatkowej pamięci VRAM.
    gradient_accumulation_steps=4,

    # Learning Rate: Zbyt wysoki powoduje niestabilność, zbyt niski wolną naukę.
    learning_rate=5e-5,
    # Warmup Ratio: Faza wstępna dla stabilizacji aktualizacji wag.
    warmup_ratio=0.1,

    # Monitoring Parameters
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,

    # Packing: Maksymalizuje utylizację GPU (packing=True).
    packing=True,
    bf16=False,
    report_to="none"
)

# 5. Inicjalizacja SFTTrainer
# SFTTrainer automatycznie obsługuje chat-style conversations.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# 6. Start treningu i monitorowanie progresu
if __name__ == "__main__":
    print("Rozpoczynam proces SFT...")

    # Należy obserwować fazy Loss: sharp drop -> stabilization -> convergence.
    trainer.train()

    # Zapisanie modelu do ewaluacji po SFT.
    trainer.save_model("./deepseek_sft_final")
    print("Trening zakończony. Pamiętaj o ewaluacji jakościowej odpowiedzi!")

"""
EWALUACJA PO SFT (Kolejne kroki wg rozdziału):
    - Sprawdź przestrzeganie szablonu (template adherence).
    - Przetestuj retencję wiedzy dziedzinowej.
    - Dokumentuj parametry treningu i charakterystykę danych dla przyszłych iteracji.
"""