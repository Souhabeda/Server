import os
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from keras.models import load_model
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import time
from flask import Flask, Blueprint, request, jsonify

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
    "USD": "USD",
    "AUD": "AUDUSD",
    "EUR": "EURUSD",
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",
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

# Note: normalize_data from prediction.py is not directly used by the route, 
# as normalization is handled within the route itself using the loaded scaler.
# However, it can be kept for other potential uses or removed if strictly unused.

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
    return load_model(path)

# get_future_predictions is essentially model.predict, called directly in the route.

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

def compute_fluctuations(unscaled_predictions_array, current_price: float):
    """Computes price fluctuations based on predictions."""
    delta = np.array(unscaled_predictions_array) - current_price
    return {
        "predicted_changes": delta.tolist(),
        "upper_change": float(np.max(delta)) if delta.size > 0 else 0.0,
        "lower_change": float(np.min(delta)) if delta.size > 0 else 0.0,
        "median_change": float(np.median(delta)) if delta.size > 0 else 0.0
    }

def get_ohlc(df):
    """Extracts the latest OHLC data."""
    if df is not None and not df.empty:
        last_candle = df.iloc[-1]
        return {
            "open": float(last_candle["open"]),
            "high": float(last_candle["high"]),
            "low": float(last_candle["low"]),
            "close": float(last_candle["close"])
        }
    return {"open": None, "high": None, "low": None, "close": None}

def get_entry_price(df):
    """Suggests an entry price (e.g., last close price)."""
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1])
    return None

def get_signal(trend):
    """Determines buy/sell signal based on trend."""
    if trend == "up":
        return "buy"
    elif trend == "down":
        return "sell"
    return "neutral"

# --- Flask Application Setup ---
test = Flask(__name__)
prediction_bp = Blueprint("prediction_bp", __name__)

# --- API Route (from prediction_routes.py) ---
@prediction_bp.route("/analyze", methods=["POST"])
def analyze_route(): # Renamed to avoid conflict if Flask app instance is also named 'analyze'
    data = request.json
    symbol_name = data.get("symbol")
    timeframe_str = data.get("timeframe")
    indicator = data.get("indicator", "rsi").lower()

    if not init_mt5():
        return jsonify({"error": "MT5 init failed"}), 500

    symbol_code = SYMBOL_MAPPING.get(symbol_name)
    timeframe_code = TIMEFRAME_MAPPING.get(timeframe_str)
    if not symbol_code or not timeframe_code:
        return jsonify({"error": "Invalid symbol or timeframe"}), 400

    df_data = fetch_data(symbol_code, timeframe_code, count=300)
    if df_data is None or df_data.empty:
        return jsonify({"error": "No data from MT5"}), 400

    # Apply indicators
    df_with_indicators = df_data.copy()
    if indicator == "rsi":
        df_with_indicators = compute_rsi(df_with_indicators)
    elif indicator == "macd":
        df_with_indicators = compute_macd(df_with_indicators)
    
    try:
        model = load_lstm_model(symbol_code, timeframe_str, indicator)
        scaler = load_scaler(symbol_code, timeframe_str, indicator)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # Prepare data for scaling and prediction
    if indicator == "rsi":
        feature_cols_for_norm = ["close", "rsi"]
    elif indicator == "macd":
        feature_cols_for_norm = ["close", "macd", "macd_signal"]
    else: # Default if indicator is unknown or not specified for features
        feature_cols_for_norm = ["close"]

    missing_cols = [col for col in feature_cols_for_norm if col not in df_with_indicators.columns]
    if missing_cols:
        return jsonify({"error": f"Missing columns for normalization after indicator computation: {missing_cols}"}), 400

    data_for_normalization = df_with_indicators[feature_cols_for_norm].dropna()
    if len(data_for_normalization) < WINDOW_SIZE:
        return jsonify({"error": f"Not enough data for LSTM input. Need {WINDOW_SIZE}, got {len(data_for_normalization)}"}), 400

    last_sequence_values = data_for_normalization.tail(WINDOW_SIZE).values
    if last_sequence_values.shape[0] != WINDOW_SIZE:
         return jsonify({"error": f"Could not form sequence of length {WINDOW_SIZE}. Shape: {last_sequence_values.shape}"}), 400

    last_sequence_scaled = scaler.transform(last_sequence_values)
    input_model_scaled = np.expand_dims(last_sequence_scaled, axis=0)

    raw_predictions_scaled = model.predict(input_model_scaled)[0]
    
    num_features_scaler = scaler.n_features_in_
    if raw_predictions_scaled.ndim == 2 and raw_predictions_scaled.shape[1] == num_features_scaler:
        unscaled_predictions = scaler.inverse_transform(raw_predictions_scaled)
        future_price_predictions = unscaled_predictions[:, 0]
    elif raw_predictions_scaled.ndim == 1 or (raw_predictions_scaled.ndim == 2 and raw_predictions_scaled.shape[1] == 1):
        predictions_reshaped = raw_predictions_scaled.reshape(-1, 1)
        dummy_for_inverse = np.zeros((predictions_reshaped.shape[0], num_features_scaler))
        dummy_for_inverse[:, 0] = predictions_reshaped[:, 0]
        unscaled_predictions_full = scaler.inverse_transform(dummy_for_inverse)
        future_price_predictions = unscaled_predictions_full[:, 0]
    else:
        return jsonify({"error": f"Unexpected prediction shape: {raw_predictions_scaled.shape}. Scaler expects {num_features_scaler} features."}), 500

    current_price = float(df_data["close"].iloc[-1])
    entry_price_val = get_entry_price(df_data)
    predicted_final_price = float(future_price_predictions[-1])
    trend = "up" if predicted_final_price > current_price else "down"
    signal_val = get_signal(trend)
    tp_val, sl_val = calculate_tp_sl(predicted_final_price, current_price, signal_val)
    fluctuations_data = compute_fluctuations(future_price_predictions, current_price)
    last_candle_time = df_data["time"].iloc[-1]
    
    time_delta_per_step = pd.Timedelta(minutes=0)
    if "H" in timeframe_str: time_delta_per_step = pd.Timedelta(hours=int(timeframe_str.replace("H", "")))
    elif "M" in timeframe_str: time_delta_per_step = pd.Timedelta(minutes=int(timeframe_str.replace("M", "")))
    
    future_timestamps_list = [(last_candle_time + time_delta_per_step * (i + 1)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(future_price_predictions))]
    ohlc_data_val = get_ohlc(df_data)

    response = {
        "symbol": symbol_code,
        "timeframe": timeframe_str,
        "indicator_used": indicator,
        "real_price": current_price,
        "entry_price": entry_price_val,
        "ohlc": ohlc_data_val,
        "signal": signal_val,
        "tp": round(tp_val, 5),
        "sl": round(sl_val, 5),
        "future_prediction": [round(p, 5) for p in future_price_predictions.tolist()],
        "last_prediction": round(predicted_final_price, 5),
        "future_timestamps": future_timestamps_list,
        "fluctuation": {
            "predicted_changes": [round(c, 5) for c in fluctuations_data["predicted_changes"]],
            "upper_change": round(fluctuations_data["upper_change"], 5),
            "lower_change": round(fluctuations_data["lower_change"], 5),
            "median_change": round(fluctuations_data["median_change"], 5)
        },
        "current_candle_time": last_candle_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(response)

test.register_blueprint(prediction_bp)

# --- Main Execution Block ---
if __name__ == "_main_":
    # Note: For this to run, MT5 must be connectable, and model/scaler files must exist.
    print("Starting Flask server for Unified Prediction Service...")
    print(f"Listening on http://0.0.0.0:5000")
    print("Send POST to /analyze with JSON body like:")
    print("{ \"symbol\": \"GOLD\", \"timeframe\": \"1H\", \"indicator\": \"rsi\" }")
    test.run(host="0.0.0.0", port=5000, debug=True)