"""
=============================================================================
WSTĘP: ALGORYTM TRENOWANIA WORDPIECE (ROZDZIAŁ 6.6)
=============================================================================
WordPiece to algorytm tokenizacji subwordowej używany m.in. w modelu BERT.
Działa on w oparciu o statystyczne prawdopodobieństwo łączenia znaków.

# TEORIA (Quiz): WordPiece nie patrzy tylko na częstotliwość pary,
# ale na wynik (score) premiujący rzadkie elementy tworzące częstą parę.
# score = freq_pary / (freq_el1 * freq_el2)

CO ROBI TEN SKRYPT?
1. Inicjalizuje korpus słów i rozbija je na pojedyncze znaki (z prefiksem ##).
2. Oblicza częstotliwość występowania każdego tokenu oraz każdej pary sąsiadów.
3. Wykorzystuje wzór punktowy WordPiece do oceny, które połączenie jest 'najsilniejsze'.
4. Łączy wybraną parę i powtarza proces (iteracja).

CO CHCEMY, ABY BYŁO WYNIKIEM?
Wynikiem ma być ewolucja słownika – chcemy zobaczyć, jak z pojedynczych liter
powstają sensowne cząstki słów (subwordy), które najlepiej reprezentują nasz
korpus przy ograniczonej liczbie kroków.
=============================================================================
"""

import collections

# 1. Nasz korpus treningowy z rozdziału (słowo: liczba wystąpień)
corpus = {
    "hug": 10,
    "pug": 5,
    "pun": 12,
    "bun": 4,
    "hugs": 5
}

# 2. Przygotowanie początkowych podziałów
# WordPiece dodaje "##" do znaków, które nie są na początku słowa
# Przykład: "hug" -> ["h", "##u", "##g"]
splits = {word: [word[0]] + ["##" + char for char in word[1:]]
          for word in corpus.keys()}

print("--- ETAP 1: POCZĄTKOWE PODZIAŁY SŁÓW ---")
for word, split in splits.items():
    print(f"Słowo '{word}': {split}")

def compute_pair_scores(current_splits):
    """
    KROK STATYSTYCZNY:
    Oblicza wyniki dla par zgodnie ze wzorem WordPiece:
    score = freq_pary / (freq_element_1 * freq_element_2)
    """
    char_freqs = collections.defaultdict(int)
    pair_freqs = collections.defaultdict(int)

    # Najpierw liczymy jak często występuje każdy pojedynczy token oraz każda para
    for word, count in corpus.items():
        split = current_splits[word]
        for i in range(len(split)):
            char_freqs[split[i]] += count # Częstotliwość pojedynczego tokenu
            if i < len(split) - 1:
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += count # Częstotliwość pary

    # Obliczamy wynik (score) dla każdej pary
    scores = {}
    for pair, freq in pair_freqs.items():
        # WZÓR: wynik = freq_pary / (freq_el1 * freq_el2)
        # Promuje pary, których elementy rzadko występują osobno
        score = freq / (char_freqs[pair[0]] * char_freqs[pair[1]])
        scores[pair] = (score, freq)
    return scores, char_freqs

def merge_pair(pair, current_splits):
    """
    KROK AKTUALIZACJI:
    Łączy wybraną parę w nowy token w całym korpusie.
    """
    first, second = pair
    new_splits = {}
    for word in current_splits:
        split = current_splits[word]
        i = 0
        new_split = []
        while i < len(split):
            if i < len(split) - 1 and split[i] == first and split[i+1] == second:
                # Łączymy: usuwamy ## z drugiego elementu jeśli tam jest
                new_token = first + second.replace("##", "")
                new_split.append(new_token)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        new_splits[word] = new_split
    return new_splits

# --- SYMULACJA TRENOWANIA (Pętla łączenia) ---
# Chcemy stworzyć 3 nowe, złożone tokeny
num_merges = 3

for step in range(num_merges):
    print(f"\n--- KROK TRENINGU {step + 1} ---")

    # Obliczamy statystyki dla aktualnego stanu słów
    all_scores, freqs = compute_pair_scores(splits)

    if not all_scores:
        break

    # Sortujemy pary według wyniku, żeby pokazać najlepsze w rankingu
    sorted_pairs = sorted(all_scores.items(), key=lambda x: x[1][0], reverse=True)

    print("Ranking punktowy par (Top 3):")
    for pair, (score, freq) in sorted_pairs[:3]:
        f1 = freqs[pair[0]]
        f2 = freqs[pair[1]]
        print(f"  Para {pair}: wynik = {freq} / ({f1} * {f2}) = {score:.5f}")

    # Wybieramy najlepszą parę jako zwycięzcę tego kroku
    best_pair = sorted_pairs[0][0]
    best_score = sorted_pairs[0][1][0]

    print(f"\nDECYZJA: Zwycięzca to {best_pair} z wynikiem {best_score:.5f}")

    # Aktualizujemy podziały w całym korpusie (łączenie tokenów)
    splits = merge_pair(best_pair, splits)

    print("Aktualny stan słów w korpusie po złączeniu:")
    for word, split in splits.items():
        print(f"  {word}: {split}")

print("\n--- WYNIK KOŃCOWY ---")
print("Zakończono proces trenowania. Słowa zostały skondensowane w subwordy.")