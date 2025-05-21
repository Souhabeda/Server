# utils.py
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import pytz
import re
import secrets
import string
import re
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from db import snapshots_collection


# Fuseau horaire du serveur MetaTrader
TIMEZONE = pytz.timezone("Etc/UTC")

# Mapping langue, symboles, horizon
SYMBOL_MAPPING = {
    "gold": "XAUUSD",
    "or": "XAUUSD",
    "xauusd": "XAUUSD",
    "bitcoin": "BTCUSD",
    "btc": "BTCUSD",
    "btcusd": "BTCUSD",
    "eurusd": "EURUSD",
    "eur": "EURUSD",
    "usd": "EURUSD",
    "gbpusd": "GBPUSD",
    "gbp": "GBPUSD",
}



LANGUAGE_KEYWORDS = {
    "fr": ["bonjour", "demain", "aujourd'hui", "semaine", "or"],
    "en": ["hello", "tomorrow", "today", "week", "gold"],
}

def detect_language(text: str) -> str:
    text = text.lower()
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        if any(word in text for word in keywords):
            return lang
    return "en"  # par défaut


# Vérifie la force d'un mot de passe
def is_strong_password(password):
    if (len(password) < 8 or
        not re.search(r"\d", password) or
        not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
        return False
    return True

# Génère un mot de passe fort
def generate_strong_password(length=12):
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()"
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        if is_strong_password(password):
            return password

def send_email(to_email, subject, body):
    message = Mail(
        from_email=os.getenv("EMAIL_USER"),
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )

    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(f"Email envoyé avec le statut : {response.status_code}")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {e}")

def save_snapshot(symbol: str, timeframe: str, indicator: str, snapshot: dict, candles: list):
    """Sauvegarde le dernier snapshot dans MongoDB, avec les dernières bougies."""
    snapshot_doc = {
        "symbol": symbol,
        "timeframe": timeframe,
        "indicator": indicator,
        "snapshot": snapshot,
        "candles": candles,  # Ajoute ici les bougies brutes
        "timestamp": datetime.utcnow()
    }
    snapshots_collection.replace_one(
        {"symbol": symbol, "timeframe": timeframe, "indicator": indicator},
        snapshot_doc,
        upsert=True
    )

def fetch_last_snapshot(symbol: str, timeframe: str, indicator: str):
    """Récupère le dernier snapshot depuis MongoDB."""
    doc = snapshots_collection.find_one(
        {"symbol": symbol, "timeframe": timeframe, "indicator": indicator}
    )
    if doc:
        return doc["snapshot"]
    return None

def is_market_open(symbol):
    if not mt5.initialize():
        return False
    try:
        tick = mt5.symbol_info_tick(symbol)
        market_info = mt5.symbol_info(symbol)
        if market_info is None or not market_info.visible or not market_info.trade_mode:
            return False
        now = datetime.now()
        weekday = now.weekday()
        # Ici vous pourriez aussi vérifier les horaires exacts de trading si nécessaire
        return weekday < 5 and tick is not None and tick.bid > 0
    except Exception as e:
        print(f"Erreur lors de la vérification du marché: {e}")
        return False
    finally:
        mt5.shutdown()
        
def get_next_market_open():
    """Détermine la prochaine ouverture du marché."""
    now = datetime.now(timezone.utc)
    next_open = now
    if now.weekday() >= 5:  # Samedi ou Dimanche
        days_ahead = 7 - now.weekday()
        next_open = datetime(now.year, now.month, now.day) + timedelta(days=days_ahead)
    elif now.weekday() == 4 and now.hour >= 21:
        next_open = datetime(now.year, now.month, now.day) + timedelta(days=3)  
        next_open = next_open.replace(hour=0, minute=0)
    return next_open.strftime("%Y-%m-%d %H:%M:%S")


def is_market_open_close(symbol):
    """ Vérifie si le marché est ouvert via MT5 et calcule la prochaine ouverture si fermé. """
    
    if not mt5.initialize():
        return {"error": "MT5 init failed", "open": False, "next_open": None}

    now = datetime.now()
    symbol_info = mt5.symbol_info(symbol)

    if not symbol_info:
        return {"error": "Invalid symbol", "open": False, "next_open": None}

    market_status = symbol_info.trade_mode  # 0 = marché fermé, 4 = marché ouvert

    # Vérifier si le marché est ouvert (Forex : 24h sauf week-end)
    if market_status == 4 and now.weekday() in [0, 1, 2, 3, 4]:  # Lundi-Vendredi
        return {"open": True, "next_open": None}

    # Déterminer la prochaine ouverture (lundi matin si week-end)
    next_open = None
    if now.weekday() in [5, 6]:  # Samedi-Dimanche
        days_until_monday = (7 - now.weekday()) % 7
        next_open = now.replace(hour=0, minute=0, second=0) + timedelta(days=days_until_monday)

    return {"open": False, "next_open": next_open.strftime("%Y-%m-%d %H:%M:%S") if next_open else None}
