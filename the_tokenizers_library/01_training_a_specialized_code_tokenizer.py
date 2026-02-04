import os
from datasets import load_dataset
from transformers import AutoTokenizer

def get_training_corpus(dataset):
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        # 'content' to nazwa kolumny w zbiorze codeparrot-ds-train
        yield samples["content"]

def main():
    # 1. Pobieranie danych
    print("--- KROK 1: Ładowanie danych z Hugging Face Hub ---")
    raw_datasets = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    print(f"Załadowano {len(raw_datasets)} przykładów kodu Python.")

    # 2. Przygotowanie iteratora
    training_corpus = get_training_corpus(raw_datasets)

    # 3. Załadowanie bazowego tokenizera
    print("\n--- KROK 2: Przygotowanie bazowego tokenizera (GPT-2) ---")
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 4. Trenowanie nowego tokenizera
    print("\n--- KROK 3: Trenowanie (Tokenizer uczy się statystyk Twojego kodu) ---")
    # To potrwa kilka minut przy 600k przykładów.
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    print("Trenowanie zakończone!")

    # 5. Test i Porównanie
    example_code = """def add_numbers(a, b):
    \"\"\"Dodaj dwie liczby.\"\"\"
    return a + b"""

    old_tokens = old_tokenizer.tokenize(example_code)
    new_tokens = tokenizer.tokenize(example_code)

    print("\n--- PORÓWNANIE REZULTATÓW ---")
    print(f"Liczba tokenów (GPT-2): {len(old_tokens)}")
    print(f"Liczba tokenów (Nasz):  {len(new_tokens)}")
    print(f"\nTak widzi kod Twój nowy tokenizer:\n{new_tokens}")

    # 6. Zapisywanie
    save_path = "code-search-net-tokenizer"
    tokenizer.save_pretrained(save_path)
    print(f"\n--- ZAPISANO: {os.path.abspath(save_path)} ---")

if __name__ == "__main__":
    main()