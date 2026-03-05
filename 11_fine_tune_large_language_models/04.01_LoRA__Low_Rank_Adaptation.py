"""
Co robi ten skrypt?

Skrypt w pełni automatyzuje proces dostrajania (fine-tuningu) dużego modelu językowego przy użyciu techniki  LoRA (Low-Rank Adaptation).
Zamiast trenować miliardy parametrów (co wymagałoby ogromnego klastra GPU), skrypt:
    1. Pobiera "surowy" model bazowy (DeepSeek-R1-Distill-Qwen-1.5B).
    2. "Zamraża" jego oryginalne wagi i dodaje małe, trenowalne macierze (adapter LoRA).
    3. Trenuje te małe macierze na przykładowym zbiorze danych (smoltalk) przez 50 kroków.
    4. Zapisuje wyuczony adapter na dysk.
    5. Dokonuje scalenia (merging) – wczytuje ponownie model bazowy, nakłada na niego wyuczony adapter i stapia je w jeden, nowy, gotowy do użycia model.

Spodziewany efekt po zakończeniu pracy:

Gdy skrypt zakończy swoje działanie bez błędów, w katalogu, z którego go uruchomiłeś, pojawią się dwa nowe foldery:
    ./lora_adapter: Zawiera same wyuczone "poprawki" (wagi LoRA) i tokenizator. Zajmuje bardzo mało miejsca (kilka-kilkanaście megabajtów).
    ./merged_model: Zawiera pełny, samodzielny model połączony z adapterem. Zajmuje kilka gigabajtów (tyle co model bazowy). Ten model jest gotowy do wgrania do dowolnej aplikacji (np. za pomocą potoków pipeline w transformers) i nie wymaga już instalacji biblioteki PEFT.

Kluczowe zalety LoRA

Wydajność pamięciowa (Memory Efficiency):
    - Tylko parametry adaptera są przechowywane w pamięci GPU.
    - Wagi modelu bazowego pozostają zamrożone i mogą być ładowane w niższej precyzji.
    - Umożliwia to dostrajanie (fine-tuning) dużych modeli na konsumenckich kartach graficznych (GPU).
Funkcje treningowe (Training Features):
    - Natywna integracja PEFT/LoRA wymagająca jedynie minimalnej konfiguracji.
    - Obsługa QLoRA (Squantyzowanej LoRA), co zapewnia jeszcze lepszą wydajność pamięciową.
Zarządzanie adapterami (Adapter Management):
    - Zapisywanie wag adaptera podczas tworzenia punktów kontrolnych (checkpoints).
    - Funkcje pozwalające na scalenie (merging) adapterów z powrotem z modelem bazowym.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def main():
    # ==========================================
    # 1. KONFIGURACJA ZMIENNYCH I DANYCH
    # ==========================================
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    dataset_id = "HuggingFaceTB/smoltalk"
    adapter_save_path = "./lora_adapter"
    merged_save_path = "./merged_model"

    print(f"Ładowanie tokenizatora dla modelu {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Dodanie tokenu wypełniającego (padding token), jeśli model go nie posiada.
    # Jest to wymagane do trenowania w partiach (batches).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Ładowanie zbioru danych {dataset_id}...")
    dataset = load_dataset(dataset_id, "all")

    # ==========================================
    # 2. ŁADOWANIE MODELU BAZOWEGO
    # ==========================================
    print("Ładowanie modelu bazowego w precyzji float16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,       # Używamy float16, by drastycznie zmniejszyć zużycie pamięci VRAM
        device_map="auto"          # Automatyczne załadowanie na GPU (jeśli dostępne)
    )

    # ==========================================
    # 3. KONFIGURACJA ADAPTERA LORA
    # ==========================================
    print("Przygotowywanie konfiguracji LoRA...")
    peft_config = LoraConfig(
        r=6,                                 # Rank (rząd macierzy). Niskie wartości (4-8) to większa kompresja.
        lora_alpha=8,                        # Mnożnik skalowania dla adaptera (często ustawiany na r * 1.5 lub r * 2).
        lora_dropout=0.05,                   # Szansa na wyzerowanie neuronu (zapobiega przeuczeniu).
        bias="none",                         # Nie trenujemy wyrazów wolnych (bias) dla optymalizacji pamięci.
        target_modules=["q_proj", "v_proj"], # Wskazujemy warstwy atencji (query i value), do których podłączamy adapter.
        task_type="CAUSAL_LM",               # Definiujemy, że to model przewidujący kolejny token.
    )

    # ==========================================
    # 4. ARGUMENTY TRENINGOWE (SFTConfig)
    # ==========================================
    print("Przygotowywanie argumentów treningowych (SFTConfig)...")
    training_args = SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=2,  # Ilość przykładów przetwarzanych naraz na jednym GPU
        gradient_accumulation_steps=4,  # Zbieranie gradientów przez x kroków przed aktualizacją wag (symuluje większy batch)
        learning_rate=2e-4,             # Szybkość uczenia (dla LoRA jest zazwyczaj wyższa niż dla pełnego fine-tuningu)
        max_steps=50,                   # Liczba kroków treningowych (ustawione na 50 dla szybkiego testu działania)
        logging_steps=10,               # Co ile kroków wypisywać logi w konsoli
        fp16=True,                      # Trenowanie w 16-bitowej precyzji
        save_strategy="no",             # Wyłączamy zapisywanie pośrednich punktów (dla przyspieszenia testu)
        max_length=512,                 # Ograniczenie długości sekwencji tekstu do 512 tokenów
    )

    # ==========================================
    # 5. INICJALIZACJA I URUCHOMIENIE TRENINGU
    # ==========================================
    print("Inicjalizacja SFTTrainer i start treningu...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Uruchomienie właściwej pętli uczącej
    trainer.train()

    # ==========================================
    # 6. ZAPIS ADAPTERA I CZYSZCZENIE PAMIĘCI
    # ==========================================
    print(f"Zapisywanie wag adaptera LoRA do folderu: {adapter_save_path}...")
    trainer.model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)

    print("Zwalnianie pamięci GPU...")
    # Usuwamy model i obiekt trenujący z pamięci RAM/VRAM, aby zrobić miejsce na łączenie wag
    del model
    del trainer
    torch.cuda.empty_cache()

    # ==========================================
    # 7. SCALANIE WAG (MERGING)
    # ==========================================
    print("\n--- Rozpoczynanie procesu łączenia (merging) wag ---")

    # Krok A: Wczytujemy z powrotem czysty model
    print("Ponowne ładowanie modelu bazowego...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto"
    )

    # Krok B: Nakładamy na niego wyuczony adapter ze ścieżki './lora_adapter'
    print("Nakładanie wag adaptera na model bazowy...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_save_path,
        torch_dtype=torch.float16 # Moduł PEFT specyficznie wymaga tego parametru pod tą nazwą
    )

    # Krok C: Łączymy wagi na stałe (fizyczne zsumowanie macierzy)
    print("Scalanie adaptera z modelem bazowym w jedną całość...")
    merged_model = peft_model.merge_and_unload()

    # Krok D: Zapisujemy gotowy potwór na dysk
    print(f"Zapisywanie gotowego (scalonego) modelu do folderu: {merged_save_path}...")
    merged_model.save_pretrained(merged_save_path)
    tokenizer.save_pretrained(merged_save_path)

    print("\nProces w pełni zakończony sukcesem! Twój dostrojony model jest gotowy do użycia.")

if __name__ == "__main__":
    main()