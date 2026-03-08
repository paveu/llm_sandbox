import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset

"""
================================================================================
SKRYPT: TRENOWANIE MODELU KAUZALNEGO OD PODSTAW (ROZDZIAŁ 7.6)
================================================================================
1. ARCHITEKTURA KAUZALNA (Causal LM):
   Model uczy się przewidywać następny token w sekwencji (zadanie autoregresyjne).
   Wymaga to zastosowania maskowania, aby model nie widział "przyszłych" tokenów.

2. ZMIANY W WERSJI v5+:
   - 'evaluation_strategy' zastąpiono przez 'eval_strategy'.
   - 'tokenizer' w Trainerze zastąpiono przez 'processing_class'.

3. CEL:
   Stworzenie modelu do autouzupełniania kodu Python (biblioteki Data Science).
================================================================================
"""

# 1. PRZYGOTOWANIE DANYCH (Filtrowany podzbiór CodeParrot)
# Wykorzystujemy mniejszy, przefiltrowany zbiór danych dotyczący bibliotek Data Science.
print("Ładowanie zbioru danych...")
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

# 2. KONFIGURACJA TOKENIZERA I MODELU
model_checkpoint = "huggingface-course/code-search-net-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

context_length = 128 # Rozmiar okna kontekstowego (mniejszy dla szybkości treningu).

# Inicjalizacja konfiguracji modelu GPT-2.
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Utworzenie modelu z losowymi wagami na podstawie konfiguracji (od podstaw).
model = GPT2LMHeadModel(config)

# 3. PREPROCESSING (Tokenizacja i dzielenie na fragmenty)
def tokenize(element):
    # Dzielimy długie pliki na fragmenty o długości context_length (chunking).
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True, # Zwraca wiele fragmentów z jednego dokumentu.
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length: # Odrzucamy ostatni, niepełny fragment.
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

print("Tokenizacja zbioru danych...")
# Używamy próbki danych, aby przyspieszyć przykład (np. 10 000 skryptów).
tokenized_train = ds_train.shuffle(seed=42).select(range(10000)).map(
    tokenize, batched=True, remove_columns=ds_train.column_names
)
tokenized_valid = ds_valid.map(
    tokenize, batched=True, remove_columns=ds_valid.column_names
)

# 4. COLLATOR I ARGUMENTY TRENINGOWE
# MLM=False oznacza, że szkolimy model kauzalny (autoregresyjny).
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)



args = TrainingArguments(
    output_dir="codeparrot-ds-local",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",        # Zaktualizowano z 'evaluation_strategy'.
    eval_steps=500,
    logging_steps=500,
    gradient_accumulation_steps=8, # Efektywny batch size = 256 (32 * 8).
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    fp16=torch.cuda.is_available(), # Optymalizacja precyzji na GPU.
    push_to_hub=False,
    report_to="none"
)

# 5. INICJALIZACJA TRAINERA
trainer = Trainer(
    model=model,
    processing_class=tokenizer,   # Zaktualizowano z 'tokenizer' (standard v5).
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
)

# 6. TRENING
print("\n--- Rozpoczynam trenowanie modelu od podstaw ---")
trainer.train()

# 7. TESTOWANIE (Inference)
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

txt = """\
# create some data
import numpy as np
x = np.random.randn(100)
y = np.random.randn(100)
# create scatter plot with x, y
"""

print("\n--- TEST GENEROWANIA KODU ---")
# Model powinien spróbować dopisać 'plt.scatter(x, y)' lub podobną komendę.
print(pipe(txt, max_new_tokens=20, num_return_sequences=1)[0]["generated_text"])