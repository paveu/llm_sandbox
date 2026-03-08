"""
https://huggingface.co/learn/llm-course/chapter12/6

SKRYPT: TRENOWANIE ZDOLNOŚCI ROZUMOWANIA (REASONING) PRZY UŻYCIU UNSLOTH I GRPO
-----------------------------------------------------------------------------------
OPIS DLA POCZĄTKUJĄCYCH:
Ten skrypt wykorzystuje bibliotekę Unsloth, aby przyspieszyć proces GRPO.
Uczymy model Gemma-3-1B rozwiązywania zadań matematycznych (GSM8K) przy użyciu
techniki Chain of Thought (łańcuch myśli) w formacie XML.

CEL:
Model ma nauczyć się pisać poprawne rozumowanie między tagami <reasoning>,
a poprawny wynik matematyczny między tagami <answer>.

SPODZIEWANY EFEKT:
Po około 200 krokach model zacznie wykazywać logiczną spójność i poprawnie
formatować odpowiedzi, co skutkuje wzrostem nagród w logach.
-----------------------------------------------------------------------------------
"""

from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# ==========================================================
# 1. KONFIGURACJA MODELU (UNSLOTH)
# ==========================================================
max_seq_length = 1024 # Długość tekstu (wliczając rozumowanie)
lora_rank = 32        # Rank LoRA - 32 jest złotym środkiem między inteligencją a szybkością

# Ładowanie modelu w 4-bitach, co pozwala na trening na darmowych GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6, # Zmniejsz, jeśli dostaniesz błąd "Out of Memory"
)

# Dodanie adapterów LoRA do konkretnych warstw modelu
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth", # Oszczędność pamięci przy długich tekstach
    random_state=3407,
)

# ==========================================================
# 2. DEFINICJA FORMATU I DANYCH (GSM8K)
# ==========================================================
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

def extract_xml_answer(text):
    """Wyciąga wynik znajdujący się między tagami <answer>."""
    try:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    except:
        return ""

def get_gsm8k_questions():
    """Ładuje i formatuje zadania matematyczne."""
    data = load_dataset("openai/gsm8k", "main")["train"]
    # Przygotowujemy prompt systemowy i wyciągamy poprawny wynik (po ####)
    return data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": x["answer"].split("####")[1].strip(),
    })

dataset = get_gsm8k_questions()

# ==========================================================
# 3. FUNKCJE NAGRODY (LOGIKA OCENIANIA)
# ==========================================

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Nagradza za poprawny wynik matematyczny (2.0 pkt)."""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def strict_format_reward_func(completions, **kwargs):
    """Nagradza za idealne trzymanie się formatu XML (0.5 pkt)."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

# ==========================================================
# 4. TRENING GRPO
# ==========================================================
training_args = GRPOConfig(
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, # Możesz zwiększyć do 4 dla stabilniejszego treningu
    num_generations=6,             # Zmniejsz, jeśli zabraknie pamięci GPU
    max_steps=250,                 # Trening jest krótki, abyś mógł szybko zobaczyć efekty
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

print("--- ROZPOCZYNAM TRENING REASONING (UNSLOTH) ---")
trainer.train()

# ==========================================================
# 5. TESTOWANIE
# ==========================================================
# Zapisujemy wyuczone wagi LoRA
model.save_lora("grpo_saved_lora")

# Prosty test wnioskowania
prompt_test = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "If I have 3 apples and gain 2 more, how many do I have?"},
], tokenize=False, add_generation_prompt=True)

# Unsloth posiada szybką metodę generowania
output = model.fast_generate(prompt_test, max_new_tokens=256)[0].outputs[0].text
print("\n--- ODPOWIEDŹ MODELU ---")
print(output)