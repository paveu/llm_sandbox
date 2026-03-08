"""
https://huggingface.co/learn/llm-course/chapter12/4
SKRYPT TRENINGOWY: IMPLEMENTACJA GRPO (GROUP RELATIVE POLICY OPTIMIZATION) W TRL
---------------------------------------------------------------------------
CO ROBI TEN SKRYPT? (WYJAŚNIENIE DLA POCZĄTKUJĄCYCH)
Ten program realizuje proces "Reinforcement Learning from Human Feedback" (RLHF)
w nowoczesnym wydaniu zwanym GRPO.

1. WCZYTYWANIE MODELU: Pobieramy mały model Qwen2-0.5B, który uczy się bardzo szybko.
2. PRZYGOTOWANIE DANYCH: Bierzemy zadania matematyczne (GSM8K). Model dostaje instrukcję:
   "Najpierw myśl w tagach <think>, potem odpowiedz w <answer>".
3. SYSTEM NAGRÓD (REWARD FUNCTIONS): To serce algorytmu. Zamiast człowieka, to małe
   funkcje w Pythonie oceniają model. Nagradzamy go, jeśli:
   - Używa poprawnego formatu (tagi XML).
   - Faktycznie "myśli" (jego sekcja myślenia jest długa i treściwa).
   - Odpowiedź ma sensowną długość (nie za krótka, nie za długa).
   - NOWOŚĆ: Udziela poprawnej merytorycznie odpowiedzi matematycznej (Verifiable Reward).
4. TRENOWANIE GRPO: Model generuje np. 8 różnych odpowiedzi na to samo pytanie.
   Algorytm porównuje te odpowiedzi między sobą (w grupie). Te, które dostały
   więcej punktów od naszych funkcji nagród, stają się wzorcem do naśladowania.

SPODZIEWANY EFEKT:
Po treningu model przestanie odpowiadać chaotycznie. Nauczysz go, że "opłaca mu się"
prowadzić wewnętrzny proces myślowy, co sprawi, że będzie rzadziej popełniał błędy
w trudnych zadaniach logicznych.
---------------------------------------------------------------------------
"""

import re
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# 1. KONFIGURACJA POCZĄTKOWA
# ==========================================================
# MODEL_ID to nazwa modelu na portalu Hugging Face.
# Qwen2-0.5B to bardzo lekka wersja, idealna na start.
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "grpo-reasoning-model" # Tutaj skrypt zapisze Twoją "mądrzejszą" wersję modelu.

# Automatyczne sprawdzanie mocy Twojego komputera (GPU):
# bf16/fp16 to sposoby zapisu liczb w pamięci karty graficznej.
# Pomagają zmieścić duży model w mniejszej ilości pamięci (VRAM).
device_supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not device_supports_bf16

# ==========================================================
# 2. DEFINICJA FUNKCJI NAGRODY (TWÓJ "NAUCZYCIEL")
# ==========================================================
# Każda funkcja przyjmuje listę odpowiedzi (completions) i zwraca listę ocen.

def format_reward_func(completions, **kwargs):
    """
    Nagradza model za trzymanie się formatu: <think>...</think><answer>...</answer>.
    To uczy model dyscypliny w strukturze odpowiedzi.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for completion in completions:
        # Re.DOTALL sprawia, że kropka w Regexie łapie też nowe linie (entery).
        if re.match(pattern, completion, re.DOTALL):
            rewards.append(1.0) # Maksymalna nagroda za poprawny format.
        else:
            rewards.append(0.0) # Brak nagrody za błędy w tagach.
    return rewards

def reasoning_content_reward_func(completions, **kwargs):
    """
    Nagradza model za to, że faktycznie się "rozpisał" w sekcji myślenia.
    Chcemy uniknąć sytuacji, gdzie model pisze pusty tag <think></think>.
    """
    rewards = []
    for completion in completions:
        # Szukamy tekstu pomiędzy tagami <think> i </think>.
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if match:
            thought_process = match.group(1).strip()
            # Jeśli myślenie ma więcej niż 20 znaków, dajemy pełną nagrodę.
            rewards.append(1.0 if len(thought_process) > 20 else 0.2)
        else:
            rewards.append(0.0)
    return rewards

def short_response_penalty_func(completions, **kwargs):
    """
    Dba o to, by model nie pisał nieskończenie długich elaboratów.
    Zbyt długie odpowiedzi często oznaczają, że model wpadł w pętlę.
    """
    rewards = []
    for completion in completions:
        length = len(completion)
        if length < 10:
            rewards.append(0.0) # Za krótka odpowiedź jest bezużyteczna.
        elif length > 512:
            rewards.append(0.5) # Kara (pół nagrody) za zbytnie gadulstwo.
        else:
            rewards.append(1.0) # Idealna długość.
    return rewards

def correctness_reward_func(completions, answer, **kwargs):
    """
    NOWOŚĆ (Rozdział 4): Nagradza model za poprawny wynik merytoryczny.
    Wyciąga tekst z tagów <answer> i porównuje go z 'prawdziwą' odpowiedzią.
    Uczy model, że myślenie musi prowadzić do prawdy, a nie tylko ładnie wyglądać.
    """
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        # Szukamy tego, co model wpisał jako ostateczny wynik wewnątrz <answer>
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            predicted_answer = match.group(1).strip()
            # Jeśli wynik zgadza się z kluczem odpowiedzi, dajemy najwyższą nagrodę (2.0)
            if predicted_answer == correct_answer:
                rewards.append(2.0) # Wysoka nagroda za poprawność matematyczną.
            else:
                rewards.append(0.0) # Brak nagrody za błędny wynik.
        else:
            rewards.append(0.0) # Brak nagrody, jeśli model zapomniał o tagu <answer>.
    return rewards

# ==========================================================
# 3. PRZYGOTOWANIE DANYCH (CO MODEL MA ROZWIĄZYWAĆ)
# ==========================================================

def get_gsm8k_questions():
    """
    Pobiera zbiór zadań matematycznych GSM8K i zamienia je na format czatu.
    To kluczowe, by model wiedział, kto mówi (system/user).
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    def format_prompt(example):
        # NOWOŚĆ: Wyciągamy czysty wynik liczbowy (np. '42') z pełnej odpowiedzi GSM8K
        target_answer = example["answer"].split("####")[-1].strip()
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You must first think about the problem inside <think> tags, and then provide the final answer inside <answer> tags."
                },
                {
                    "role": "user",
                    "content": example["question"]
                }
            ],
            # To pole 'answer' zostanie automatycznie przekazane do correctness_reward_func
            "answer": target_answer
        }

    # Mapujemy (przetwarzamy) cały zbiór danych przez naszą funkcję formatującą.
    # Usuwamy kolumnę 'question', bo mamy już gotowy 'prompt'.
    return dataset.map(format_prompt).remove_columns(["question"])

# ==========================================================
# 4. KONFIGURACJA I URUCHOMIENIE TRENINGU
# ==========================================================

if __name__ == "__main__":
    # Wczytujemy dane
    train_dataset = get_gsm8k_questions()

    # Rozmiar grupy (Group Size) to najważniejszy parametr GRPO.
    # Model wygeneruje 8 wersji odpowiedzi na KAŻDE pytanie, by móc je porównać.
    NUM_GENERATIONS = 8

    # Konfiguracja treningu (GRPOConfig)
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="moja-pierwsza-nauka-rozumowania-z-logika",
        learning_rate=1e-5,        # Bardzo mały krok nauki, by nie "zepsuć" modelu.
        num_train_epochs=1,        # Ile razy model przejrzy cały zbiór danych.

        # Ustawienia GRPO
        num_generations=NUM_GENERATIONS,
        generation_batch_size=NUM_GENERATIONS,

        # Parametry sprzętowe (dobrane do Twojego GPU)
        per_device_train_batch_size=1, # Przetwarzamy 1 pytanie na raz (ale z 8 odpowiedziami).
        gradient_accumulation_steps=4, # "Udajemy" większy batch, sumując wyniki z 4 kroków.
        bf16=device_supports_bf16,
        fp16=use_fp16,

        logging_steps=1,           # Pokazuj postępy (metryki) po każdym kroku.
        report_to="none",          # Nie wysyłaj danych do zewnętrznych serwisów.
    )

    # Inicjalizacja Trenera GRPO
    # GRPOTrainer łączy model, dane i funkcje nagrody w jeden proces.
    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[
            format_reward_func,
            reasoning_content_reward_func,
            short_response_penalty_func,
            correctness_reward_func # DODANO: Sprawdzanie poprawności wyniku (Rozdział 4)
        ],
    )

    print(f"--- START TRENINGU: Model uczy się myśleć I rozwiązywać zadania ---")
    print("Obserwuj logi. Szukaj wartości 'reward' - im wyższa, tym lepiej model się uczy!")

    # Uruchomienie właściwego uczenia
    trainer.train()

    # Zapisanie efektów
    trainer.save_model(OUTPUT_DIR)
    print(f"Sukces! Twój mądrzejszy model czeka w folderze: {OUTPUT_DIR}")