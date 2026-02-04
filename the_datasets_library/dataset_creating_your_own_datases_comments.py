import time
import math
import requests
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset

# --- 1. CONFIGURATION ---
GITHUB_TOKEN = "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"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
OWNER = "huggingface"
REPO = "datasets"
ISSUES_PATH = Path("./data")


def get_comments(issue_number):
    """
    Fetches comments for a specific issue number.
    Extracts the 'body' field from each comment object.
    """
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)

    # Check for successful response to avoid crashing on rate limits or 404s
    if response.status_code == 200:
        return [r["body"] for r in response.json()]
    else:
        return []


# --- 2. FETCHING DATA ---

def fetch_issues(owner=OWNER, repo=REPO, num_issues=150):
    ISSUES_PATH.mkdir(exist_ok=True)

    all_issues = []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    print(f"\n>>> Step 1: Fetching issues from {owner}/{repo}")

    for page in tqdm(range(num_pages), desc="Downloading"):
        # We add 'state=all' to get both open and closed issues
        query = f"?page={page}&per_page={per_page}&state=all"
        response = requests.get(base_url + query, headers=headers)

        if response.status_code != 200:
            print(f"\n[!] API Error: {response.status_code}")
            break

        data = response.json()
        if not data: break
        all_issues.extend(data)

    df = pd.DataFrame.from_records(all_issues)

    # Save to JSONL for Hugging Face Datasets to load
    file_path = ISSUES_PATH / f"{repo}-issues_comments.jsonl"
    df.to_json(file_path, orient="records", lines=True)
    print(f"[OK] Saved {len(df)} issues to {file_path}")
    return file_path


# --- 3. AUGMENTATION & ANALYSIS ---

def process_and_augment(file_path):
    print("\n>>> Step 2: Processing with Hugging Face Datasets")

    # Load the local JSONL file
    issues_dataset = load_dataset("json", data_files=str(file_path), split="train")

    # 1. Add 'is_pull_request' column
    # GitHub issues API returns a 'pull_request' key if it's a PR, else None
    issues_dataset = issues_dataset.map(
        lambda x: {"is_pull_request": x.get("pull_request") is not None}
    )

    # 2. Augment with Comments
    # Note: We limit to a small sample (e.g., 15) to avoid hitting GitHub API limits
    print(">>> Step 3: Augmenting with comments (Fetching from API)...")
    sample_size = 15
    small_dataset = issues_dataset.select(range(min(sample_size, len(issues_dataset))))

    # Use Dataset.map to call our get_comments function for every row
    issues_with_comments = small_dataset.map(
        lambda x: {"comments": get_comments(x["number"])},
        desc="Fetching comments"
    )

    return issues_with_comments


# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure token is set
    if GITHUB_TOKEN == "YOUR_GITHUB_TOKEN":
        print("Please set your GITHUB_TOKEN first!")
    else:
        path = fetch_issues(num_issues=50)
        final_ds = process_and_augment(path)

        print("\n" + "=" * 30)
        print("FINAL DATASET PREVIEW")
        print("=" * 30)
        print(final_ds)
        # Show first entry comments
        print(f"\nFirst issue comments count: {len(final_ds[0]['comments'])}")