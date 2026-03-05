import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import load_dataset

"""
================================================================================
CO ROBI TEN SKRYPT? (QUESTION ANSWERING - ROZDZIAŁ 7.7)
================================================================================
1. EKSTRAKCYJNE ODPOWIADANIE NA PYTANIA (Extractive QA):
   Model nie wymyśla odpowiedzi, lecz wskazuje konkretny fragment (span) 
   w tekście źródłowym (kontekście), który najlepiej odpowiada na pytanie.

2. ARCHITEKTURA TYLKO ENKODER (BERT):
   Modele typu BERT są idealne do wyodrębniania faktów z tekstu.
   Model przewiduje dwa logity dla każdego tokena: start_logits i end_logits.

3. OKNO PRZESUWNE (Sliding Window):
   Długie dokumenty są dzielone na mniejsze, nakładające się fragmenty (stride),
   aby model mógł przetworzyć kontekst przekraczający limit 512 tokenów.
================================================================================
"""

# 1. PRZYGOTOWANIE DANYCH (SQuAD - Stanford Question Answering Dataset)
print("Ładowanie zbioru danych SQuAD...")
raw_datasets = load_dataset("squad")

# 2. KONFIGURACJA MODELU I TOKENIZERA
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Parametry okna przesuwnego
max_length = 384  # Maksymalna długość wejścia
doc_stride = 128  # Nakładanie się fragmentów tekstu

# 3. PREPROCESSING (Przygotowanie etykiet start/end)
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second", # Ucinamy tylko kontekst, nie pytanie
        stride=doc_stride,
        return_overflowing_tokens=True, # Tworzy kilka cech z jednego przykładu
        return_offsets_mapping=True,     # Mapowanie tokenów na znaki w tekście
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Znalezienie początku i końca kontekstu w tokenach
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Jeśli odpowiedź nie mieści się w tym fragmencie, etykieta to (0, 0)
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Mapowanie pozycji znaków na indeksy tokenów
            curr_idx = context_start
            while curr_idx <= context_end and offsets[curr_idx][0] <= start_char:
                curr_idx += 1
            start_positions.append(curr_idx - 1)

            curr_idx = context_end
            while curr_idx >= context_start and offsets[curr_idx][1] >= end_char:
                curr_idx -= 1
            end_positions.append(curr_idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print("Przetwarzanie datasetu (tokenizacja i okno przesuwne)...")
train_dataset = raw_datasets["train"].select(range(5000)).map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# 4. MODEL I TRENING
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
data_collator = DefaultDataCollator()

args = TrainingArguments(
    output_dir="bert-finetuned-squad",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n--- Rozpoczynam dostrajanie modelu BERT (Extractive QA) ---")
trainer.train()

# 5. TESTOWANIE (Inference)
# Pipeline automatycznie zajmuje się oknem przesuwnym i wyborem najlepszej odpowiedzi
qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

print("\n--- TEST DZIAŁANIA PO TRENINGU ---")
context = "W 2026 roku firma Hugging Face stała się liderem w dziedzinie AI. Jej siedziba znajduje się w Nowym Jorku."
question = "Gdzie znajduje się siedziba Hugging Face?"

result = qa_pipe(question=question, context=context)
print(f"Kontekst: {context}")
print(f"Pytanie: {question}")
print(f"Odpowiedź: {result['answer']} (pewność: {round(result['score'], 4)})")