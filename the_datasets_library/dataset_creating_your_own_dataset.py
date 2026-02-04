import time
import math
import requests
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset

# --- 1. KONFIGURACJA ---
GITHUB_TOKEN = "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"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
OWNER = "huggingface"
REPO = "datasets"
ISSUES_PATH = Path("./data")


def get_comments(issue_number):
    """Pobiera komentarze dla konkretnego zgłoszenia (Augmentacja)."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [r["body"] for r in response.json()]
    return []


# --- 2. PEŁNA FUNKCJA POBIERANIA (TWOJA STRUKTURA + PANDAS) ---

def fetch_issues(owner=OWNER, repo=REPO, num_issues=150, rate_limit=5000):
    if not ISSUES_PATH.is_dir():
        ISSUES_PATH.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    print("\n" + "=" * 60)
    print(f"KROK 1: Pobieranie danych z GitHub API ({owner}/{repo})")
    print("=" * 60)

    for page in tqdm(range(num_pages), desc="Pobieranie stron"):
        query = f"issues?page={page}&per_page={per_page}&state=all"
        url = f"{base_url}/{owner}/{repo}/{query}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"\n[!] Błąd API: {response.status_code}")
            break

        data = response.json()
        if not data: break
        batch.extend(data)

        # Twoja logika obsługi limitu rate_limit
        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []
            print(f"\n[!] Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)

    # TWOJA ANALIZA PANDAS (info, nunique, head)
    print("\n--- ANALIZA PANDAS (PRZED ZAPISEM) ---")
    df.info()
    if 'user' in df.columns:
        logins = df['user'].apply(lambda x: x['login'] if isinstance(x, dict) else None)
        print(f"\nUnikalni autorzy: {logins.nunique()}")
    print("\nPodgląd (df.head):")
    print(df[['number', 'title', 'state']].head())

    # Zapis lokalny JSONL
    file_path = ISSUES_PATH / f"{repo}-issues.jsonl"
    df.to_json(file_path, orient="records", lines=True)
    print(f"\n[OK] Dataset stored at {file_path}")

    return file_path


# --- 3. PROCES CLEAN UP & ANALIZA (LOGIKA Z ROZDZIAŁU 5) ---

def clean_and_analyze(file_path):
    print("\n" + "=" * 60)
    print("KROK 2: Cleaning up the data (Hugging Face Datasets)")
    print("=" * 60)

    # Ładowanie z pliku lokalnego
    issues_dataset = load_dataset("json", data_files=str(file_path), split="train")

    # --- INSPEKCJA PRÓBKI (shuffle + select + zip) ---
    print("\n[*] Tworzenie losowej próbki (seed=666) do porównania URL i PR:")
    sample = issues_dataset.shuffle(seed=666).select(range(3))

    for url, pr in zip(sample["html_url"], sample["pull_request"]):
        print(f">> URL: {url}")
        print(f">> Pull request: {pr}\n")

    # --- MAPOWANIE: Tworzenie kolumny is_pull_request ---
    print("[*] Mapowanie: Dodawanie kolumny 'is_pull_request'...")
    issues_dataset = issues_dataset.map(
        lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
    )

    # --- ZADANIE: ŚREDNI CZAS ZAMYKANIA (TRY IT OUT!) ---
    print("\n[*] Obliczanie średniego czasu zamykania zgłoszeń...")

    # Filtrowanie: Tylko zamknięte zgłoszenia (nie PR)
    closed_issues = issues_dataset.filter(lambda x: x["is_pull_request"] is False and x["state"] == "closed")

    # Tymczasowa konwersja na Pandas do obliczeń na datach
    closed_issues.set_format("pandas")
    df_closed = closed_issues[:]

    df_closed["created_at"] = pd.to_datetime(df_closed["created_at"])
    df_closed["closed_at"] = pd.to_datetime(df_closed["closed_at"])
    df_closed["time_to_close"] = df_closed["closed_at"] - df_closed["created_at"]

    print(f"Średni czas zamknięcia zgłoszenia: {df_closed['time_to_close'].mean()}")

    # Powrót do formatu Arrow
    closed_issues.reset_format()

    # --- AUGMENTACJA: DODAWANIE KOMENTARZY ---
    print("\n[*] KROK 3: Augmenting the dataset (Komentarze)...")
    # Dla testu wybieramy 10 rekordów
    final_sample = issues_dataset.select(range(min(10, len(issues_dataset))))
    final_dataset = final_sample.map(
        lambda x: {"comments": get_comments(x["number"])},
        desc="Pobieranie komentarzy"
    )

    print("\n" + "=" * 60)
    print("PODSUMOWANIE FINALNEGO DATASETU")
    print("=" * 60)
    print(final_dataset)
    return final_dataset


# --- START ---
if __name__ == "__main__":
    data_file = fetch_issues(num_issues=150)
    final_ds = clean_and_analyze(data_file)