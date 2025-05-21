import re
from langdetect import detect
from fuzzywuzzy import fuzz

SYMBOL_ALIASES = {
    "GOLD": ["gold", "or", "xauusd"],
    "SILVER": ["silver", "argent", "xagusd"],
    "BTCUSD": ["btc", "bitcoin", "btc/usd", "btc usd"],
    "ETHUSD": ["eth", "ethereum", "eth/usd", "eth usd"],
    "SOLUSD": ["sol", "solana", "sol/usd", "sol usd"],
    "XPTUSD": ["xpt", "platine", "platinum"," platinium", "xptusd"],
    "EURUSD": ["eur", "euro", "eurusd"],
    "AUDUSD": ["aud", "audusd", "australian dollar"],
    "USD": ["usd", "dollar américain"],
    "GBPUSD": ["gbp", "gbpusd", "pound", "Great Britain pound"]
}

# Dictionnaire inversé pour récupérer un alias "friendly" pour chaque symbole
SYMBOL_TO_NEWS_QUERY = {
    "GOLD": "gold",
    "SILVER": "silver",
    "BTCUSD": "bitcoin",
    "ETHUSD": "ethereum",
    "SOLUSD": "solana",
    "XPTUSD": "platinum",
    "EURUSD": "euro",
    "AUDUSD": "australian dollar",
    "USD": "usd",
    "GBPUSD": "gbp"
}

# Meta prompt global
META_PROMPT = """
You are a bilingual (French and English) AI assistant specialized exclusively in trading and financial markets. 
You respond fluently in the same language as the user (French or English).
You are a helpful and friendly trading assistant.  
Answer user questions about trading in a simple, clear, concise way.  
Avoid jargon and long technical explanations.  
Focus on teaching basic concepts and practical advice.

Your responsibilities:
- Answer only to trading-related questions (e.g., about forex, crypto, commodities like gold/silver, technical indicators like RSI/MACD, prediction, investment strategies, news, etc.)
- Always use **real-time data**:
    • For currencies and commodities (like XAUUSD, EURUSD...), get data from MetaTrader5.
    • For cryptocurrencies (like BTCUSDT, ETHUSDT...), get data from Binance API.
- Symbols and asset names should be interpreted dynamically: don't rely on static lists.
- If a user asks about general trading topics (e.g., “what is trading?”, “how to read a candlestick?”, “c’est quoi le scalping ?”), answer clearly.
- For trading news or general finance analysis, respond via Gemini.

Rules:
- If the question is not related to trading or finance, reply strictly with: “Sorry, I’m a trading assistant.” / “Désolé, je suis un assistant de trading.”
- Always provide clear and concise responses, and include numbers or predictions if relevant.
- If the user asks about buying or selling, analyze the trend using RSI, MACD, and LSTM prediction before answering.
- Always maintain a professional and financial tone.
"""

def detect_symbol(message):
    message = message.lower()
    best_score = 0
    best_symbol = None

    for symbol, aliases in SYMBOL_ALIASES.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, message):
                return symbol
            score = fuzz.partial_ratio(alias, message)
            if score > best_score and score >= 80:
                best_score = score
                best_symbol = symbol

    return best_symbol

def detect_horizon(message):
    message = message.lower()

    if "demain" in message or "tomorrow" in message:
        return 1
    if "cette semaine" in message or "this week" in message:
        return 7
    if "la semaine prochaine" in message or "next week" in message:
        return 7
    if "ce mois" in message or "ce mois-ci" in message or "this month" in message:
        return 30
    if "mois prochain" in message or "next month" in message:
        return 30

    day_match = re.search(r"(\d+)\s*(j|jrs|jour|jours|day|days)", message)
    if day_match:
        return int(day_match.group(1))

    week_match = re.search(r"(\d+)\s*(s|semaine?|semaines?|week?|weeks?)", message)
    if week_match:
        return int(week_match.group(1)) * 7

    return 1

def detect_language(message: str) -> str:
    try:
        lang = detect(message)
        return lang
    except:
        return "en"

# Détection basique d’une question de trading général
def is_trading_question(message: str) -> bool:
    base_keywords = [
        "trading", "investir", "scalping", "swing", "bougie", "chart", "price action",
        "indicateur", "rsi", "macd", "acheter", "vendre", "short", "long",
        "forex", "crypto", "marché", "analyse", "stratégie", "bourse", "stock", "market",
        "prediction", "prédiction", "prévision", "forecast", "estimation"
    ]

    # On génère aussi les mots avec un 's' pour les pluriels simples
    keywords = base_keywords + [kw + "s" for kw in base_keywords if not kw.endswith("s")]

    # Préparation du message
    message = message.lower()
    message = re.sub(r"[^\w\s]", " ", message)

    # Vérifie si au moins un mot clé (ou son pluriel) est dans le message
    return any(re.search(rf'\b{re.escape(kw)}\b', message) for kw in keywords)

def is_greeting(message: str) -> bool:
    greetings = ["hi", "hello", "hey", "salut", "bonjour", "bonsoir", "thank you"]
    message = message.lower()
    message = re.sub(r"[^\w\s]", " ", message)
    return any(re.search(rf'\b{re.escape(greet)}\b', message) for greet in greetings)

def is_news_request(message: str) -> bool:
    keywords = [
        "news", "latest news", "breaking news", "updates", "nouvelles", "actualités",
        "quoi de neuf", "quoi s'est passé", "ce qui se passe", "événements", "dernières infos",
        "recent news", "des nouvelles", "update", "actualisation", "info", "infos",
        "quelles sont les nouvelles", "what's the news", "any news", "trading news", "crypto news"
    ]

    message = message.lower()
    message = re.sub(r"[^\w\s]", " ", message)

    # Détection supplémentaire hors liste
    if "what's happening" in message or "ce qui se passe" in message:
        return True

    return any(re.search(rf'\b{re.escape(kw)}\b', message) for kw in keywords)
def is_advice_request(message: str) -> bool:
    keywords = [
        "advice", "advices", "tip", "tips", "recommendation", "recommendations",
        "suggestion", "suggestions", "guide", "how to start", "how to trade", "how to invest",
        "conseil", "conseils", "astuce", "astuces", "recommandation", "recommandations",
        "commencer", "débute", "débuter", "débutant", "investir"
    ]

    message = message.lower()
    message = re.sub(r"[^\w\s]", " ", message)

    return any(re.search(rf'\b{re.escape(kw)}\b', message) for kw in keywords)

def is_valid_response(response: str, symbol: str = None) -> bool:
    """
    Valide si la réponse est pertinente.
    Si un symbole est donné, la réponse doit contenir ce symbole.
    On rejette aussi les réponses génériques ou d'erreur.
    """
    if not response or response.strip() == "":
        return False

    if symbol and symbol.lower() not in response.lower():
        return False

    rejection_phrases = [
        "I am not able",
        "I don't understand",
        "Sorry, I cannot",
        "I can't answer",
        "Sorry, I'm a trading assistant."
    ]
    if any(phrase.lower() in response.lower() for phrase in rejection_phrases):
        return False

    return True
