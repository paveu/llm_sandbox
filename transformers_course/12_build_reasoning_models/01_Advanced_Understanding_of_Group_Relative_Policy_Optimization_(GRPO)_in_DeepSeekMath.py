"""
Advanced Understanding of Group Relative Policy Optimization (GRPO) in DeepSeekMath

SKRYPT: Group Relative Policy Optimization (GRPO) - Implementacja Minimalna
AUTOR: Gemini AI (na podstawie DeepSeekMath)

OPIS DZIAŁANIA:
Niniejszy skrypt demonstruje zaawansowaną technikę uczenia wzmocnionego (RL),
wprowadzoną przez zespół DeepSeek. GRPO eliminuje potrzebę stosowania osobnego
modelu krytyka (Critic), zastępując go relatywnym porównaniem wyników wewnątrz
grupy generacji (Group Relative).

PROCES:
1. Próbkowanie grupowe: Dla każdego pytania generowane jest G odpowiedzi.
2. Obliczanie nagrody: Sprawdzana jest poprawność matematyczna każdej odpowiedzi.
3. Obliczanie przewagi (Advantage): Wyniki są standaryzowane wewnątrz grupy (r - mean)/std.
4. Aktualizacja polityki: Model jest optymalizowany, aby zwiększyć prawdopodobieństwo
   odpowiedzi lepszych niż średnia grupowa, przy zachowaniu stabilności.

EFEKT KOŃCOWY:
Po uruchomieniu skrypt wyświetli wygenerowane odpowiedzi matematyczne, przypisane
im nagrody oraz obliczone wartości "Advantage". Zobaczysz również końcową wartość
straty (Loss), która w prawdziwym procesie treningowym służyłaby do aktualizacji
wag modelu za pomocą backpropagation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

# --- KONFIGURACJA HIPERPARAMETRÓW (PARAMETRY UCZENIA) ---
MODEL_NAME = "Qwen/Qwen2-Math-1.5B"

# EPSILON (Clip Range): Określa, jak bardzo nowa polityka może odbiegać od starej.
# Wartość 0.2 oznacza, że zmiana prawdopodobieństwa danej odpowiedzi jest ograniczona do +/- 20%.
# Chroni to przed zbyt gwałtownymi zmianami, które mogłyby zdestabilizować model.
EPSILON = 0.2

# BETA (KL Coefficient): Waga kary za Dywergencję KL.
# Im wyższa beta, tym silniej model jest "trzymany" blisko oryginalnych wag modelu referencyjnego.
# Zapobiega to tzw. "rozpadowi modelu" (model collapse) lub generowaniu bełkotu.
BETA = 0.04

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Ładowanie modelu i tokenizera
    print(f"--- Inicjalizacja modelu: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # Model referencyjny (model_ref): To kopia modelu sprzed aktualizacji.
    # Używamy jej do obliczenia KL Divergence – sprawdzamy, jak bardzo "nowy" model
    # różni się od "oryginału". Model ten musi być w trybie .eval() i nie jest trenowany.
    model_ref = copy.deepcopy(model)
    model_ref.eval()

    # Definiujemy zadanie matematyczne (prompt)
    prompt = "Solve y = 2x + 1 for x = 2, y = "
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    # --- KROK 1: PRÓBKOWANIE GRUPOWE (PARAMETRY GENERACJI) ---
    # batch_size: Liczba unikalnych pytań przetwarzanych jednocześnie.
    batch_size = 2
    # G (Group Size): Liczba odpowiedzi generowanych dla KAŻDEGO pytania.
    # To kluczowy parametr GRPO – im większe G, tym lepsza estymacja średniej jakości odpowiedzi.
    G = 4

    print(f"Generowanie {batch_size * G} odpowiedzi (Batch Size={batch_size}, Group Size={G})...")

    # model.generate - proces tworzenia odpowiedzi przez model:
    outputs = model.generate(
        input_ids=input_ids.repeat(batch_size * G, 1),
        # max_new_tokens: Maksymalna liczba tokenów, które model może dopisać do promptu.
        max_new_tokens=5,
        # do_sample=True: Pozwala na losowość. Bez tego model zawsze wybierałby najbardziej
        # prawdopodobny token (greedy search), co uniemożliwiłoby eksplorację różnych ścieżek.
        do_sample=True,
        # temperature: Kontroluje kreatywność. 0.8 to umiarkowana losowość.
        # Niska temp. (np. 0.1) sprawia, że model jest pewny siebie, wysoka (np. 1.5) czyni go chaotycznym.
        temperature=0.8,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_ids = outputs.sequences
    decoded_outputs = [
        tokenizer.decode(g[input_ids.shape[-1]:], skip_special_tokens=True).strip()
        for g in generated_ids
    ]

    # --- KROK 2: OBLICZANIE NAGRÓD I PRZEWAGI (GRPO LOGIC) ---
    # Symulacja funkcji nagrody (Reward Function).
    # W matematyce nagroda jest binarna (1.0 za poprawny wynik, 0.0 za błąd).
    rewards_list = [1.0 if "5" in out else 0.0 for out in decoded_outputs]
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=DEVICE)

    # Grupowanie nagród: Zmieniamy kształt z [8] na [2, 4] (2 pytania, po 4 odpowiedzi każde).
    rewards_grouped = rewards.view(batch_size, G)
    # Obliczamy średnią (mean) i odchylenie standardowe (std) wewnątrz każdej grupy.
    mean_g = rewards_grouped.mean(dim=1, keepdim=True)
    std_g = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

    # Advantages (Przewaga): To serce GRPO.
    # Wynik dodatni oznacza, że ta konkretna odpowiedź była LEPSZA niż inne próby dla tego samego pytania.
    # Wynik ujemny oznacza, że była GORSZA. Model będzie dążył do wzmacniania tych dodatnich.
    advantages = (rewards_grouped - mean_g) / std_g
    advantages = advantages.view(-1, 1)

    # --- KROK 3: OBLICZANIE STRATY (MATEMATYKA RL) ---
    outputs_new = model(generated_ids)
    outputs_old = model_ref(generated_ids)

    # get_per_token_logps: Wyciąga logarytm prawdopodobieństwa tokenów, które model faktycznie wybrał.
    def get_per_token_logps(logits, target_ids):
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=target_ids.unsqueeze(2))
        return per_token_logps.squeeze(2)

    gen_token_ids = generated_ids[:, input_ids.shape[-1]:]

    # Prawdopodobieństwa tokenów w nowym modelu (new_logps) i starym (old_logps).
    new_logps = get_per_token_logps(outputs_new.logits[:, :-1, :], generated_ids[:, 1:])
    new_logps = new_logps[:, -gen_token_ids.shape[1]:]

    old_logps = get_per_token_logps(outputs_old.logits[:, :-1, :], generated_ids[:, 1:])
    old_logps = old_logps[:, -gen_token_ids.shape[1]:]

    # ratio (Współczynnik prawdopodobieństwa): Mówi nam, o ile częściej model wybiera teraz
    # daną odpowiedź w porównaniu do stanu sprzed sekundy.
    ratio = torch.exp(new_logps - old_logps.detach())

    # Policy Loss (Strata Polityki): Implementacja PPO.
    # Wybieramy mniejszą wartość (min) między surowym ratio a zclipowanym ratio.
    # Dzięki temu, jeśli model nagle chce zwiększyć prawdopodobieństwo odpowiedzi o 1000%,
    # funkcja 'clamp' zetnie to do bezpiecznego poziomu (np. 20%).
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL Divergence: Statystyczna miara "odległości" między dwoma rozkładami.
    # Kara KL sprawia, że model nie zapomina języka naturalnego w pogoni za samą poprawnością matematyczną.
    kl_div = torch.exp(old_logps - new_logps) - (old_logps - new_logps) - 1
    kl_loss = kl_div.mean()

    # Total Loss (Całkowita Strata): To, co optymalizator (np. AdamW) stara się zminimalizować.
    total_loss = policy_loss + BETA * kl_loss

    # --- WYŚWIETLANIE WYNIKÓW ---
    print("\n" + "="*60)
    print("ANALIZA GRUPY (GRPO):")
    print("="*60)
    for i, out in enumerate(decoded_outputs):
        status = "DOBRA" if rewards_list[i] > 0 else "ZŁA"
        print(f"Odp {i+1} [{status}]: '{out}' | Adv: {advantages[i].item():.4f}")

    print("-" * 60)
    print(f"Parametry: Epsilon={EPSILON}, Beta={BETA}, Temp={0.8}")
    print(f"Średnia Nagroda w Batchu: {rewards.mean().item():.2f}")
    print(f"Policy Loss (cel: ujemny): {policy_loss.item():.4f}")
    print(f"KL Loss (kara za zmianę): {kl_loss.item():.4f}")
    print(f"STRATA CAŁKOWITA: {total_loss.item():.4f}")
    print("="*60)
    print("Skrypt pomyślnie zasymulował jeden krok optymalizacji GRPO.")

if __name__ == "__main__":
    main()