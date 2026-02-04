"""
=============================================================================
WSTĘP: ALGORYTM UNIGRAM (ROZDZIAŁ 6.7)
=============================================================================
Unigram opiera się na założeniu, że prawdopodobieństwo sekwencji tokenów
to iloczyn prawdopodobieństw każdego tokena z osobna.

CO ROBI TEN SKRYPT?
1. Buduje początkowy słownik wszystkich możliwych podciągów (substrings).
2. Oblicza prawdopodobieństwo P(x) dla każdego tokena.
3. Dla słowa 'hug' znajduje wszystkie możliwe ścieżki tokenizacji.
4. Wybiera tę o najwyższym prawdopodobieństwie (najmniejsza strata).

CO CHCEMY, ABY BYŁO WYNIKIEM?
Wynikiem jest demonstracja, jak model Unigram porównuje różne sposoby
segmentacji tego samego słowa i wybiera najbardziej prawdopodobną.
=============================================================================
"""

import math

# 1. KORPUS I SŁOWNIK POCZĄTKOWY
corpus = {
    "hug": 10,
    "pug": 5,
    "pun": 12,
    "bun": 4,
    "hugs": 5
}

# Uproszczony słownik wszystkich podciągów z przykładu w kursie
vocab = [
    "h", "u", "g", "s", "p", "b", "n",
    "hu", "ug", "pu", "un", "bu",
    "hug", "pug", "pun", "bun", "hugs"
]

# Częstotliwości (zgodnie z tabelą w rozdziale)
token_counts = {
    "h": 15, "u": 36, "g": 20, "s": 5, "p": 17, "b": 4, "n": 16,
    "hu": 15, "ug": 20, "pu": 5, "un": 16, "bu": 4,
    "hug": 15, "pug": 5, "pun": 12, "bun": 4, "hugs": 5
}

# 2. OBLICZANIE PRAWDOPODOBIEŃSTW P(x)
total_count = sum(token_counts.values())
probabilities = {token: count / total_count for token, count in token_counts.items()}

print(f"--- ETAP 1: PRAWDOPODOBIEŃSTWA (Suma częstości: {total_count}) ---")
print(f"P('h')   = {probabilities['h']:.4f}")
print(f"P('hu')  = {probabilities['hu']:.4f}")
print(f"P('hug') = {probabilities['hug']:.4f}\n")


def get_tokenizations(word):
    """Funkcja pomocnicza: znajduje wszystkie możliwe podziały słowa na tokeny ze słownika"""

    def find_segments(text):
        if not text:
            return [[]]
        result = []
        for i in range(1, len(text) + 1):
            prefix = text[:i]
            if prefix in vocab:
                for suffix_segments in find_segments(text[i:]):
                    result.append([prefix] + suffix_segments)
        return result

    return find_segments(word)


# 3. ANALIZA SŁOWA "HUG"
word_to_analyze = "hug"
possible_segments = get_tokenizations(word_to_analyze)

print(f"--- ETAP 2: MOŻLIWE TOKENIZACJE DLA '{word_to_analyze}' ---")

best_segmentation = None
max_prob = -1

for seg in possible_segments:
    # P(sekwencja) = P(x1) * P(x2) * ...
    prob = 1.0
    for token in seg:
        prob *= probabilities[token]

    # Obliczamy ujemny logarytm (strata) - im niższy, tym lepiej
    loss = -math.log(prob)

    print(f"Podział: {str(seg):<15} | Prawdopodobieństwo: {prob:.6f} | Strata (loss): {loss:.2f}")

    if prob > max_prob:
        max_prob = prob
        best_segmentation = seg

# 4. WYNIK KOŃCOWY
print(f"\n--- WYNIK KOŃCOWY ---")
print(f"Najlepsza tokenizacja dla '{word_to_analyze}' to: {best_segmentation}")
print(f"To ten podział zostanie użyty do obliczenia całkowitej straty korpusu.")

"""
DALSZE KROKI W TRENOWANIU UNIGRAM:
W pełnym algorytmie teraz należałoby:
1. Obliczyć całkowitą stratę dla wszystkich słów w korpusie.
2. Spróbować usunąć każdy token i zobaczyć, jak zmieni się ta strata.
3. Usunąć 10-20% tokenów, które najmniej zwiększają stratę.
4. Powtarzać, aż osiągniemy docelowy rozmiar słownika.
"""