"""
https://huggingface.co/learn/llm-course/chapter12/5
SKRYPT: PRAKTYCZNE DOSTRAJANIE MODELU SMOL-LM PRZY UŻYCIU GRPO
-----------------------------------------------------------------------------------
OPIS DLA POCZĄTKUJĄCYCH:
Ten skrypt realizuje proces uczenia ze wzmocnieniem (Reinforcement Learning) na małym
modelu językowym. W odróżnieniu od standardowego uczenia (SFT), gdzie pokazujemy
modelowi gotowe odpowiedzi, tutaj model uczy się poprzez "eksperymentowanie" i
otrzymywanie punktów (nagród) za zachowania, których od niego oczekujemy.

CO DOKŁADNIE SIĘ DZIEJE:
1. ŁADOWANIE MODELU: Używamy SmolLM2-135M – to bardzo mały model, który zmieści się
   w pamięci domowej karty graficznej.
2. ADAPTACJA LoRA: Zamiast trenować miliardy parametrów, dodajemy do modelu
   inteligentne "nakładki" (adaptery). Dzięki temu oszczędzamy ok. 90% pamięci GPU.
3. FUNKCJA NAGRODY (Reward Function): Definiujemy prostą regułę: "Twoja odpowiedź
   powinna mieć około 50 tokenów". Model dostaje karę za każde odstępstwo od tej liczby.
4. TRENING GRPO: Model generuje grupy 8 odpowiedzi na każdy prompt, porównuje je
   między sobą i aktualizuje swoje wagi tak, aby częściej trafiać w oczekiwaną długość.

SPODZIEWANY EFEKT:
Po zakończeniu treningu model, który wcześniej pisał odpowiedzi o różnej długości,
zacznie generować teksty wyraźnie oscylujące wokół 50 tokenów. Zobaczysz to w logach
jako parametr 'reward' zbliżający się do zera.
-----------------------------------------------------------------------------------
"""

import torch
import re
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import GRPOConfig, GRPOTrainer

# ==========================================================
# 1. KONFIGURACJA MODELU I OPTYMALIZACJI UWAGI (ATTENTION)
# ==========================================================
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Ponieważ instalacja flash-attn wymaga specyficznych sterowników (CUDA_HOME/nvcc),
# stosujemy mechanizm zabezpieczający. Jeśli flash-attn nie zadziała,
# skrypt użyje SDPA – wbudowanego, szybkiego mechanizmu w PyTorch.
try:
    import flash_attn
    attn_impl = "flash_attention_2"
except ImportError:
    # SDPA (Scaled Dot Product Attention) jest domyślnym wyborem dla nowoczesnych kart GPU
    attn_impl = "sdpa"

# Wczytujemy model z automatycznym dopasowaniem precyzji (dtype)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    attn_implementation=attn_impl,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ustawiamy token dopełnienia (pad_token), aby model wiedział, gdzie kończy się tekst
tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# 2. KONFIGURACJA LoRA (EFEKTYWNE WYKORZYSTANIE PAMIĘCI)
# ==========================================================
# LoRA pozwala na trenowanie modeli przy drastycznie mniejszym zużyciu VRAM.
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,               # Ranga macierzy – im wyższa, tym więcej model może zapamiętać, ale zużywa więcej RAM
    lora_alpha=32,
    target_modules="all-linear", # Celujemy we wszystkie warstwy liniowe dla lepszych efektów
)
model = get_peft_model(model, lora_config)
# Drukuje informację o tym, ile parametrów faktycznie będziemy trenować (zazwyczaj < 2%)
model.print_trainable_parameters()

# ==========================================================
# 3. DANE TRENINGOWE I DEFINICJA "NAUCZYCIELA" (NAGRODY)
# ==========================================================
# Ładujemy zbiór krótkich opowiadań smoltldr.
dataset = load_dataset("mlabonne/smoltldr", split="train")

ideal_length = 50

def reward_len(completions, **kwargs):
    """
    Funkcja nagrody oceniająca długość odpowiedzi.
    Im bliżej 50 tokenów jest model, tym wyższą (bliższą 0) dostaje nagrodę.
    """
    # Zwracamy ujemną różnicę od ideału (kara za błąd)
    return [-abs(ideal_length - len(completion)) for completion in completions]

# ==========================================================
# 4. USTAWIENIA TRENINGU GRPO (KLUCZOWE PARAMETRY)
# ==========================================================
training_args = GRPOConfig(
    output_dir="SmolGRPO-Result",
    learning_rate=2e-5,              # Prędkość nauki – mała wartość zapewnia stabilność
    per_device_train_batch_size=4,   # Ilość przykładów przetwarzanych jednocześnie
    gradient_accumulation_steps=4,   # Kumulacja wyników przed aktualizacją wag (oszczędność VRAM)
    num_generations=8,               # Rozmiar grupy dla GRPO – model generuje 8 wersji na raz
    generation_batch_size=8,         # Musi być wielokrotnością num_generations
    num_train_epochs=1,              # Jedno pełne przejście przez dane
    # Korzystamy z precyzji bf16, jeśli karta graficzna na to pozwala
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    logging_steps=1,                 # Chcemy widzieć wyniki po każdym kroku
    report_to="none",                # Wyłączamy zewnętrzne systemy logowania jak WandB
)

# ==========================================================
# 5. INICJALIZACJA I URUCHOMIENIE TRENERA
# ==========================================================
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_len],       # Przekazujemy naszą listę funkcji nagród
)



print(f"--- ROZPOCZYNAM TRENING GRPO ---")
print(f"Technologia uwagi: {attn_impl}")
trainer.train()

# ==========================================================
# 6. TESTOWANIE WYTRENOWANEGO MODELU (WERYFIKACJA)
# ==========================================================
# Łączymy wagi LoRA z modelem bazowym, aby uzyskać gotowy plik
model = model.merge_and_unload()
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Zadajemy testowe pytanie, aby sprawdzić "styl" po treningu
prompt = "Tell me a very short story about a brave astronaut."
messages = [{"role": "user", "content": prompt}]

output = generator(messages, max_new_tokens=100, do_sample=True, temperature=0.7)
print("\n--- TESTOWA ODPOWIEDŹ MODELU PO TRENINGU ---")
print(output[0]['generated_text'])