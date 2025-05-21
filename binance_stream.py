# binance_stream.py
import websocket
import json
import numpy as np
from collections import deque
from tabulate import tabulate
import threading
import requests
import time

symbols = [
    "btcusdt", "ethusdt", "solusdt", "bnbusdt",
    "suiusdt", "adausdt", "xrpusdt"
]

symbol_to_name = {
    "btcusdt": "Bitcoin",
    "ethusdt": "Ethereum",
    "solusdt": "Solana",
    "bnbusdt": "Binance",
    "suiusdt": "Sui",
    "adausdt": "Cardano",
    "xrpusdt": "XRP",
}

streams = [f"{symbol}@ticker" for symbol in symbols]
sma_window = 5
prices = {symbol: deque(maxlen=sma_window) for symbol in symbols}
symbol_data = {}
market_caps = {}
socketio = None

def init_socketio(socketio_instance):
    global socketio
    socketio = socketio_instance

def fetch_market_caps():
    while True:
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": "bitcoin,ethereum,solana,binancecoin,cardano,ripple,sui",
            }
            response = requests.get(url, params=params)
            for coin in response.json():
                symbol = coin["symbol"] + "usdt"
                market_caps[symbol] = f"${coin['market_cap']:,.0f}"
        except Exception as e:
            print(f"❌ fetch_market_caps error: {e}")
        time.sleep(60)

def display_table():
    headers = ["#", "Nom", "Dernier Prix", "24h %", "Market Cap", "sma_signal", "Trade"]
    rows = []

    for idx, symbol in enumerate(symbol_data.keys(), start=1):
        entry = symbol_data[symbol]
        name = symbol_to_name.get(symbol, symbol.upper())
        price = f"${entry['price']:,.2f}"
        change = f"{entry['change_24h']:+.2f}%"
        market_cap = market_caps.get(symbol, "N/A")

        # Ajout des flèches
        sma_signal = entry.get("sma_signal")
        if sma_signal == "UP":
            sma_signal_display = "🔼"
        elif sma_signal == "DOWN":
            sma_signal_display = "🔽"
        else:
            sma_signal_display = "–"

        trade = "🟢 Trade"
        rows.append([idx, name, price, change, market_cap, sma_signal_display, trade])

    # print("\033c", end="")
    # print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    
def on_message(ws, message):
    try:
        msg = json.loads(message)
        stream_data = msg.get("data")
        if not stream_data:
            print("⚠️ Missing data in message:", msg)
            return

        symbol = stream_data['s'].lower()
        price = float(stream_data['c'])
        change_percent = float(stream_data['P'])

        symbol_data[symbol] = {
            "price": price,
            "change_24h": change_percent,
            "market_cap": market_caps.get(symbol, "N/A"),
            "icon": symbol.replace("usdt", ""),
            "name": symbol_to_name.get(symbol, symbol.upper())
        }

        prices[symbol].append(price)
        if len(prices[symbol]) == sma_window:
            sma = np.mean(prices[symbol])
            direction = "UP" if price > sma else "DOWN"
            symbol_data[symbol]["sma_signal"] = direction

        if socketio:
            print(f"📤 Emitting SocketIO for {symbol} :", symbol_data[symbol])  # 🪵 Log utile
            socketio.emit("crypto_update", symbol_data)
        else:
            print("❌ SocketIO is None: not emitted")

    except Exception as e:
        print(f"❌ on_message error: {e}")

def on_error(ws, error):
    print("❌ WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed")

def on_open(ws):
    print("✅ WebSocket connection opened")
    params = {
        "method": "SUBSCRIBE",
        "params": streams,
        "id": 1
    }
    ws.send(json.dumps(params))

def start_binance_stream():
    print("📡 Starting real-time Binance stream...")
    threading.Thread(target=fetch_market_caps, daemon=True).start()
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
    ws = websocket.WebSocketApp(url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open)
    ws.run_forever()
