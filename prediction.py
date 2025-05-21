import os
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError 
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time


# --- Environment Variable Loading ---
load_dotenv()
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")

# --- General Parameters ---
WINDOW_SIZE = 60
FUTURE_STEPS = 5  # Model output shape defines actual future steps
MODEL_DIR = "models_lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Mappings ---
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

# --- Core Logic Functions (from prediction.py) ---

def init_mt5() -> bool:
    """Initializes MetaTrader 5 connection."""
    if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER:
        print("MT5 connection details (login, password, server) are not fully set in .env file.")
        if not MT5_PATH:
            if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                print(f"MetaTrader5 initialize failed (no path), error code = {mt5.last_error()}")
                return False
            print("MetaTrader5 initialized (no path).")
            return True
        return False
    if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MetaTrader5 initialize failed with path {MT5_PATH}, error code = {mt5.last_error()}")
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            print(f"MetaTrader5 initialize failed (fallback without path), error code = {mt5.last_error()}")
            return False
        print("MetaTrader5 initialized (fallback, without explicit path).")
        return True
    print(f"MetaTrader5 initialized with path: {MT5_PATH}")
    return True

def fetch_data(symbol: str, timeframe: int, count: int = 300):
    """Fetches market data from MT5."""
    if timeframe not in TIMEFRAME_MAPPING.values():
        print(f"Error: Invalid timeframe code {timeframe}.")
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} with timeframe {timeframe}. Error: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns=str.lower, inplace=True)
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df

def compute_rsi(df_input):
    """Computes RSI indicator."""
    df = df_input.copy()
    if "close" not in df.columns or df["close"].isnull().all():
        print("Cannot compute RSI: 'close' column missing or all NaN.")
        df["rsi"] = np.nan
        df["RSI_Signal"] = "Neutral"
        return df
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[rs == np.inf, "rsi"] = 100
    df["RSI_Signal"] = "Neutral"
    df.loc[df["rsi"] < 30, "RSI_Signal"] = "Buy"
    df.loc[df["rsi"] > 70, "RSI_Signal"] = "Sell"
    return df

def compute_macd(df_input) -> pd.DataFrame:
    """Computes MACD indicator."""
    df = df_input.copy()
    if "close" not in df.columns or df["close"].isnull().all():
        print("Cannot compute MACD: 'close' column missing or all NaN.")
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["MACD_Cross"] = "Neutral"
        return df
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["MACD_Cross"] = "Neutral"
    df.loc[(df["macd"] > df["macd_signal"]) & (df["macd"].shift(1).fillna(method="bfill") <= df["macd_signal"].shift(1).fillna(method="bfill")), "MACD_Cross"] = "Buy"
    df.loc[(df["macd"] < df["macd_signal"]) & (df["macd"].shift(1).fillna(method="bfill") >= df["macd_signal"].shift(1).fillna(method="bfill")), "MACD_Cross"] = "Sell"
    return df

def recreate_and_save_scaler(df_for_fitting, symbol: str, timeframe: str, indicator: str = "rsi"):
    """Recreates and saves a MinMaxScaler."""
    if indicator.lower() == "rsi":
        feature_columns = ["close", "rsi"]
    elif indicator.lower() == "macd":
        feature_columns = ["close", "macd", "macd_signal"]
    else:
        print(f"Invalid indicator '{indicator}' for recreate_and_save_scaler.")
        return None
    actual_feature_columns = [col for col in feature_columns if col in df_for_fitting.columns]
    if not actual_feature_columns or len(actual_feature_columns) != len(feature_columns):
        print(f"Warning: Not all expected columns for scaler. Expected {feature_columns}, found {actual_feature_columns}")
        return None
    scaler = MinMaxScaler()
    scaler.fit(df_for_fitting[actual_feature_columns].dropna().values)
    scaler_filename = f"{symbol}_{timeframe}_{indicator}_scaler.save"
    path = os.path.join(MODEL_DIR, symbol, scaler_filename)
    os.makedirs(os.path.join(MODEL_DIR, symbol), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved: {path}")
    return scaler

def load_scaler(symbol: str, timeframe: str, indicator: str = "rsi") -> MinMaxScaler:
    """Loads a previously saved MinMaxScaler."""
    scaler_filename = f"{symbol}_{timeframe}_{indicator}_scaler.save"
    path = os.path.join(MODEL_DIR, symbol, scaler_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)

def save_model(model, symbol: str, timeframe: str, indicator: str = "rsi"):
    """Saves the Keras model."""
    model_filename = f"{symbol}_{timeframe}_{indicator}_lstm.h5"
    model_path = os.path.join(MODEL_DIR, symbol, model_filename)
    os.makedirs(os.path.join(MODEL_DIR, symbol), exist_ok=True)
    model.save(model_path)
    print(f"Model saved: {model_path}")

def load_lstm_model(symbol: str, timeframe: str, indicator: str = "rsi"):
    """Loads a Keras LSTM model."""
    model_filename = f"{symbol}_{timeframe}_{indicator}_lstm.h5"
    path = os.path.join(MODEL_DIR, symbol, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    # ✅ Corriger ici en passant custom_objects
    return load_model(path, custom_objects={'mse': MeanSquaredError()})

# get_future_predictions is essentially model.predict, called directly in the route.
def generate_future_predictions(last_candle_time, future_price_predictions, num_steps=10, time_delta_minutes=5):
    time_delta_per_step = timedelta(minutes=time_delta_minutes)
    
    # Liste des prédictions futures de prix
    future_timestamps = [round(p, 2) for p in future_price_predictions]  # Utiliser directement les valeurs des prédictions

    predicted_final_price = future_price_predictions[-1]

    return {
        "future_prediction": [round(p, 5) for p in future_price_predictions],
        "last_prediction": round(predicted_final_price, 5),
        "future_timestamps": future_timestamps  # Maintenant ce sont des valeurs de prix, pas des timestamps Unix
    }

def ensure_lstm_model_exists(symbol: str, timeframe: str, indicator: str = "rsi"):
    """Vérifie si le modèle LSTM et le scaler existent. Sinon, les génère et sauvegarde."""
    model_filename = f"{symbol}_{timeframe}_{indicator}_lstm.h5"
    model_path = os.path.join(MODEL_DIR, symbol, model_filename)
    scaler_filename = f"{symbol}_{timeframe}_{indicator}_scaler.save"
    scaler_path = os.path.join(MODEL_DIR, symbol, scaler_filename)

    # Crée le dossier si nécessaire
    os.makedirs(os.path.join(MODEL_DIR, symbol), exist_ok=True)

    # Si le modèle existe déjà, rien à faire
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("✅ Modèle et scaler déjà existants.")
        return

    print("⚠️ Modèle ou scaler manquant, génération en cours...")

    # Récupère les données
    mt5_timeframe = TIMEFRAME_MAPPING.get(timeframe)
    df = fetch_data(symbol, mt5_timeframe, count=500)
    if df is None or df.empty:
        print("❌ Impossible de récupérer les données pour entraîner le modèle.")
        return

    # Calcul des indicateurs
    if indicator == "rsi":
        df = compute_rsi(df)
        features = ["close", "rsi"]
    elif indicator == "macd":
        df = compute_macd(df)
        features = ["close", "macd", "macd_signal"]
    else:
        print("❌ Indicateur non supporté.")
        return

    # Nettoyage des données
    df.dropna(subset=features, inplace=True)
    if len(df) < WINDOW_SIZE + FUTURE_STEPS:
        print("❌ Pas assez de données pour entraîner un modèle.")
        return

    # Normalisation
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features].values)

    # Enregistre le scaler
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé : {scaler_path}")

    # Préparation des données d'entraînement
    X, y = [], []
    for i in range(WINDOW_SIZE, len(data_scaled) - FUTURE_STEPS):
        X.append(data_scaled[i - WINDOW_SIZE:i])
        y.append(data_scaled[i + FUTURE_STEPS - 1][0])  # Prédire "close" future

    X, y = np.array(X), np.array(y)

    # Création du modèle LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Sauvegarde du modèle
    model.save(model_path)
    print(f"✅ Modèle LSTM sauvegardé : {model_path}")


def calculate_tp_sl(predicted_final_price: float, current_price: float, signal: str, risk_reward_ratio: float = 2.0, sl_pips: float = None, tp_pips: float = None, atr_val: float = None):
    """Calculates Take Profit (TP) and Stop Loss (SL)."""
    if tp_pips and sl_pips:
        tp = current_price + tp_pips if signal == "buy" else current_price - tp_pips
        sl = current_price - sl_pips if signal == "buy" else current_price + sl_pips
        return tp, sl
    if atr_val:
        sl_offset = 1.5 * atr_val
        tp_offset = risk_reward_ratio * sl_offset
        if signal == "buy":
            tp = current_price + tp_offset
            sl = current_price - sl_offset
        else:  # sell
            tp = current_price - tp_offset
            sl = current_price + sl_offset
        return tp, sl
    price_diff = predicted_final_price - current_price
    if signal == "buy":
        if price_diff <= 0: price_diff = abs(current_price * 0.005) # Minimal expected profit
        tp = current_price + abs(price_diff) * risk_reward_ratio
        sl = current_price - abs(price_diff)
    else:  # sell
        if price_diff >= 0: price_diff = -abs(current_price * 0.005)
        tp = current_price - abs(price_diff) * risk_reward_ratio
        sl = current_price + abs(price_diff)
    return tp, sl

def calculate_fluctuation_levels(df, window: int = 5):
    """
    Calcule la fluctuation sur les prix de clôture des X dernières bougies.
    """
    if df is None or df.empty or len(df) < window:
        return {
            "upper": None,
            "lower": None,
            "median": None,
            "fluctuation": None,
            "percent_fluctuation": None
        }

    closes = df["close"].tail(window)
    upper = closes.max()
    lower = closes.min()
    median = closes.median()
    fluctuation = upper - lower
    percent_fluctuation = (fluctuation / closes.mean()) * 100

    return {
        "upper": float(upper),
        "lower": float(lower),
        "median": float(median),
        "fluctuation": float(fluctuation),
        "percent_fluctuation": float(percent_fluctuation)
    }

def get_ohlc(df, num_candles=50):
    """Extracts the last num_candles OHLC data."""
    if df is not None and not df.empty:
        ohlc_data = df.tail(num_candles)[["time", "open", "high", "low", "close"]]
        ohlc_data["time"] = ohlc_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")  # Format date pour JSON
        return ohlc_data.to_dict(orient="records")  # Convertir en liste de dictionnaires JSON
    return []


def get_entry_price(df, window: int = 5):
    """
    Calcule un prix d'entrée basé sur la moyenne des prix de clôture
    des dernières bougies pour lisser le bruit de marché.
    """
    if df is None or df.empty or len(df) < window:
        return float(df["close"].iloc[-1]) if not df.empty else None

    closes = df["close"].tail(window)
    entry_price = closes.mean()
    return float(entry_price)

def get_signal(trend):
    """Determines buy/sell signal based on trend."""
    if trend == "up":
        return "buy"
    elif trend == "down":
        return "sell"
    return "neutral"
