from mcp.server.fastmcp import FastMCP

# Inicjalizacja serwera MCP o nazwie "MojSerwer"
mcp = FastMCP("MojSerwer")

# --- IMPLEMENTACJA NARZĘDZIA (Tool) ---
@mcp.tool()
def get_weather(location: str) -> str:
    """Pobiera aktualną pogodę dla podanej lokalizacji."""
    # W prawdziwej aplikacji tutaj byłoby zapytanie do API
    return f"Pogoda w {location}: Słonecznie, 22°C"

# --- IMPLEMENTACJA ZASOBU (Resource) ---
@mcp.resource("memo://instrukcje")
def get_instructions() -> str:
    """Zwraca ważne instrukcje zapisane na serwerze."""
    return "To jest statyczny zasób tekstowy dostępny przez protokół MCP."

if __name__ == "__main__":
    # Uruchomienie serwera
    mcp.run()