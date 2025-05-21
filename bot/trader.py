# trader.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import talib


# === Initialisation MT5 ===
if not mt5.initialize():
    print("Connection error to MT5.")
    exit()

# === Vérifie si un symbole est crypto (on suppose que crypto = binance) ===
def is_crypto(symbol: str):
    return symbol.upper().endswith("USDT") or symbol.upper() in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']

# === Récupère les données de Binance ===
def get_binance_data(symbol, interval='1m', limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Binance data fetch error for {symbol}")
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]

# === Récupère les données de MT5 ===
def get_mt5_data(symbol, nb_candles=500, timeframe=mt5.TIMEFRAME_M1):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, nb_candles)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} on MT5")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})

# === Préparation LSTM ===
def prepare_data_lstm(series, lookback=10):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)

# === Entraînement ou chargement ===
def train_and_save_model(X, y, model_path):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    model.save(model_path)
    return model

def get_or_train_model(scaled_prices, symbol, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_lstm_{symbol}.h5")
    X, y = prepare_data_lstm(scaled_prices)
    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
    else:
        model = train_and_save_model(X, y, model_path)
    return model

# === Prédiction ===
def predict_next_price(model, last_seq):
    return model.predict(last_seq)[0][0]

# === Indicateurs ===
def compute_indicators(df):
    close = df['close'].values
    df['rsi'] = talib.RSI(close, timeperiod=14)
    macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = signal
    return df

# === Recommandation dynamique selon source (crypto ou pas) ===
def get_recommendation(symbol: str, horizon_days: int = 1):
    df = get_binance_data(symbol) if is_crypto(symbol) else get_mt5_data(symbol)
    df = compute_indicators(df)

    prices = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    model = get_or_train_model(scaled, symbol)
    last_seq = scaled[-10:].reshape(1, 10, 1)
    pred_scaled = predict_next_price(model, last_seq)
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    current_price = prices[-1][0]

    trend = "UP" if pred_price > current_price else "DOWN"
    recommendation = "BUY" if trend == "UP" else "SELL"

    rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    macd_trend = "UP" if macd > macd_signal else "DOWN"

    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "current_price": float(current_price),
            "predicted_price": float(pred_price),
            "trend": trend,
            "rsi": float(rsi),
            "macd": float(macd),
            "macd_signal": float(macd_signal),
            "macd_trend": macd_trend,
            "recommendation": recommendation
        }
    }

# === Exécution trade MT5 uniquement ===
def execute_trade(symbol, action, lot=0.1):
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": f"Trade auto {action}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Error during order execution. :", result)
    else:
        print(f"Order {action} executed successfully.")

def place_trade(symbol: str, action: str):
    if is_crypto(symbol):
        return {"error": "Live crypto trading not supported in this setup."}
    try:
        execute_trade(symbol, action)
        return {"message": f"Order {action} executed successfully for {symbol}"}
    except Exception as e:
        return {"error": str(e)}
    
# === NewsApi ===
def get_news_newsapi(query, api_key, language='en', page_size=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": page_size
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        news_list = []
        for article in articles:
            title = article.get("title")
            source = article.get("source", {}).get("name")
            published_at = article.get("publishedAt", "")[:10]
            news_list.append(f"{published_at} - {title} ({source})")
        return news_list
    except Exception as e:
        return [f"Error fetching news: {str(e)}"]