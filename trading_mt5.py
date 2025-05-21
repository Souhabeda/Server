# fichier: trading_mt5.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Charger le .env en haut
load_dotenv()

# Charger les variables ENV
MT5_LOGIN = int(os.getenv('MT5_LOGIN'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER')
MT5_PATH = os.getenv('MT5_PATH')

RISK_REWARD_RATIO = 1.75

SYMBOL_MAPPING = {
    "GBP": "GBPUSD",
    "AUD": "AUDUSD",
    "EUR": "EURUSD",
    "GOLD": "GOLD",
    "SILVER": "SILVER",
    "XPTUSD": "XPTUSD",
    "BTCUSD": "BTCUSD",
    "SOLUSD": "SOLUSD",
    "ETHUSD": "ETHUSD"
}

TIMEFRAME_MAPPING = {
    "1H": mt5.TIMEFRAME_H1,
    "15M": mt5.TIMEFRAME_M15
}

INDICATORS = ["RSI", "MACD"]

def initialize_mt5():
    if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print("❌ Échec de l'initialisation MT5 :", mt5.last_error())
        return False
    print("✅ MT5 initialisé avec succès.")
    return True

def fetch_data_today(symbol, timeframe):
    now = datetime.now()
    date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, now)
    df = pd.DataFrame(rates)

    if df.empty:
        print(f"⚠️ Pas de données pour {symbol} sur timeframe {timeframe}.")
        return pd.DataFrame()

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns=str.title, inplace=True)
    return df

def detect_fvg(data, body_multiplier=1.5):
    fvg_list = []
    for i in range(2, len(data)):
        first_high = data['High'].iloc[i-2]
        first_low = data['Low'].iloc[i-2]
        middle_open = data['Open'].iloc[i-1]
        middle_close = data['Close'].iloc[i-1]
        third_low = data['Low'].iloc[i]
        third_high = data['High'].iloc[i]

        prev_bodies = (data['Close'].iloc[max(0, i-10):i-1] - data['Open'].iloc[max(0, i-10):i-1]).abs()
        avg_body_size = prev_bodies.mean() or 0.001
        middle_body = abs(middle_close - middle_open)

        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(("Bullish FVG", first_high, third_low, data.index[i]))
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(("Bearish FVG", third_high, first_low, data.index[i]))
    return fvg_list

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line

def calculate_tp_sl(signal, entry_price, sl_price, risk_reward_ratio):
    risk = abs(entry_price - sl_price)
    tp = entry_price + (risk * risk_reward_ratio) if signal == "Buy" else entry_price - (risk * risk_reward_ratio)
    return sl_price, tp

def find_entry_signal(data_15m, daily_trend, buffer_zone=0.5, body_confirmation=True):
    calculate_rsi(data_15m)
    calculate_macd(data_15m)
    fvg_list = detect_fvg(data_15m)
    last_fvg = fvg_list[-1] if fvg_list else None

    if last_fvg:
        fvg_type, low, high, timestamp = last_fvg
        sl_buy = low
        sl_sell = high
    else:
        sl_buy = sl_sell = None

    for idx, row in data_15m.iterrows():
        current_close = row['Close']
        current_open = row['Open']
        current_rsi = row['RSI']
        current_macd = row['MACD']
        current_macd_signal = row['MACD_Signal']

        if pd.isna(current_rsi) or pd.isna(current_macd) or pd.isna(current_macd_signal):
            continue

        if last_fvg:
            in_zone = (low - buffer_zone) <= current_close <= (high + buffer_zone)
            if fvg_type == "Bullish FVG" and in_zone:
                if body_confirmation and current_close <= current_open:
                    continue
                if current_rsi < 30 and current_macd > current_macd_signal:
                    return build_signal("Buy", current_close, sl_buy, idx, row)
            elif fvg_type == "Bearish FVG" and in_zone:
                if body_confirmation and current_close >= current_open:
                    continue
                if current_rsi > 70 and current_macd < current_macd_signal:
                    return build_signal("Sell", current_close, sl_sell, idx, row)

        if current_rsi < 30:
            sl_buy = current_close - (current_close * 0.003)
            return build_signal("Buy", current_close, sl_buy, idx, row)
        elif current_rsi > 70:
            sl_sell = current_close + (current_close * 0.003)
            return build_signal("Sell", current_close, sl_sell, idx, row)

    return None

def build_signal(signal_type, entry_price, sl_price, timestamp, row):
    return {
        "signal": signal_type,
        "entry": entry_price,
        "sl": sl_price,
        "timestamp": timestamp,
        "rsi": row['RSI'],
        "macd": row['MACD'],
        "macd_signal": row['MACD_Signal'],
        "open": row['Open'],
        "close": row['Close'],
        "volume": row.get('Volume', 0)
    }

def lancer_trading(symbol_name, timeframe_name):
    if not initialize_mt5():
        return {"error": "Connexion MT5 impossible."}

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe_name)
    if not timeframe_code:
        mt5.shutdown()
        return {"error": "Timeframe invalide."}

    mapped_symbol = SYMBOL_MAPPING.get(symbol_name)
    if not mapped_symbol:
        mt5.shutdown()
        return {"error": "Symbole invalide."}

    print(f"🔍 Analyse de {mapped_symbol} sur {timeframe_name}...")

    data = fetch_data_today(mapped_symbol, timeframe_code)
    if data.empty:
        mt5.shutdown()
        return {"error": "Pas de données pour ce symbole/timeframe."}

    daily_trend = "Bullish" if data.iloc[-1]['Close'] > data.iloc[0]['Open'] else "Bearish"
    signal_info = find_entry_signal(data, daily_trend)
    mt5.shutdown()

    if signal_info:
        sl, tp = calculate_tp_sl(signal_info['signal'], signal_info['entry'], signal_info['sl'], RISK_REWARD_RATIO)
        return {
            "type": "Achat" if signal_info['signal'] == "Buy" else "Vente",
            "timestamp": str(signal_info['timestamp']),
            "entry": signal_info['entry'],
            "stop_loss": sl,
            "take_profit": tp,
            "rsi": round(signal_info['rsi'], 2),
            "macd": round(signal_info['macd'], 5),
            "macd_signal": round(signal_info['macd_signal'], 5),
            "open": signal_info['open'],
            "close": signal_info['close'],
            "volume": signal_info['volume']
        }
    else:
        return {"message": "Pas de signal trouvé actuellement."}
