"""
INSTRUKCJA MCP DLA CLAUDE CODE (LINUX/UBUNTU):

1. URUCHOMIENIE SERWERA:
   python app.py

2. DODAWANIE DO CLAUDE CODE:
   claude mcp add gradio-server -- npx -y mcp-remote http://127.0.0.1:7860/gradio_api/mcp/sse

3. USUWANIE Z CLAUDE CODE:
   claude mcp remove gradio-server

4. LISTA ZAREJESTROWANYCH SERWERÓW:
   claude mcp list

5. GOTOWE PROMPTY DO TESTÓW W CLAUDE CODE:
   - "Czy masz dostęp do narzędzi z gradio-server?"
   - "Policz litery 'a' w 'Abrakadabra'. Potem policz tylko WIELKIE 'A' używając case_sensitive=True."
   - "Przeanalizuj statystyki mojego pliku app.py narzędziem text_stats."
   - "Sprawdź narzędziem find_word, czy w moim kodzie jest fraza 'mcp_server=True'."
   - "Pobierz zawartość app.py, podaj statystyki i policz w nim wszystkie znaki '='"
"""

import gradio as gr

def letter_counter(word: str, letter: str, case_sensitive: bool = False) -> int:
    """
    Liczy wystąpienia konkretnej litery w podanym tekście.

    Args:
        word (str): Tekst lub słowo do przeszukania.
        letter (str): Pojedyncza litera, której szukamy.
        case_sensitive (bool): Jeśli True, wielkość liter ma znaczenie (A != a). Domyślnie False.

    Returns:
        int: Liczba znalezionych powtórzeń.
    """
    if not letter:
        return 0
    if not case_sensitive:
        word = word.lower()
        letter = letter.lower()
    return word.count(letter)

def text_stats(text: str) -> dict:
    """
    Analizuje tekst i zwraca statystyki: liczbę słów, znaków i średnią długość słowa.

    Args:
        text (str): Tekst do analizy.

    Returns:
        dict: Słownik ze statystykami (words, chars, avg_word_len).
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_len = char_count / word_count if word_count > 0 else 0

    return {
        "words": word_count,
        "chars": char_count,
        "avg_word_len": round(avg_len, 2)
    }

def find_word(text: str, search_query: str) -> bool:
    """
    Sprawdza, czy konkretne słowo lub fraza znajduje się w tekście.

    Args:
        text (str): Tekst, w którym szukamy.
        search_query (str): Słowo lub fraza do znalezienia.

    Returns:
        bool: True jeśli znaleziono, False w przeciwnym razie.
    """
    return search_query.lower() in text.lower()

# Budowa interfejsu Gradio
with gr.Blocks(title="Python MCP Server Pro") as demo:
    gr.Markdown("# 🛠️ Zaawansowany Serwer MCP")

    with gr.Tab("Licznik Liter"):
        w_in = gr.Textbox(label="Tekst")
        l_in = gr.Textbox(label="Litera", max_length=1)
        cs_in = gr.Checkbox(label="Rozróżniaj wielkość liter (Case Sensitive)")
        c_out = gr.Number(label="Wynik")
        btn_l = gr.Button("Licz")
        btn_l.click(fn=letter_counter, inputs=[w_in, l_in, cs_in], outputs=c_out)

    with gr.Tab("Statystyki"):
        t_in = gr.Textbox(label="Tekst do analizy")
        s_out = gr.JSON(label="Statystyki JSON")
        btn_s = gr.Button("Analizuj")
        btn_s.click(fn=text_stats, inputs=t_in, outputs=s_out)

    with gr.Tab("Wyszukiwarka"):
        f_text = gr.Textbox(label="Tekst")
        f_query = gr.Textbox(label="Szukana fraza")
        f_out = gr.Checkbox(label="Czy znaleziono?", interactive=False)
        btn_f = gr.Button("Szukaj")
        btn_f.click(fn=find_word, inputs=[f_text, f_query], outputs=f_out)

if __name__ == "__main__":
    # Uruchomienie serwera MCP
    demo.launch(mcp_server=True)