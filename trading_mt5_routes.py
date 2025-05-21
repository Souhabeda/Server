# fichier: trading_mt5_routes.py

from flask import Blueprint, jsonify, request
from trading_mt5 import initialize_mt5, fetch_data_today, calculate_rsi, calculate_macd,calculate_tp_sl,RISK_REWARD_RATIO, TIMEFRAME_MAPPING, detect_fvg, SYMBOL_MAPPING,INDICATORS
import MetaTrader5 as mt5
import math
from lstm_predictor import load_and_predict_lstm
from utils import save_snapshot, fetch_last_snapshot, is_market_open

routes = Blueprint('routes', __name__)

def is_market_closed(df):
    """ Vérifie si le marché est fermé : uniquement si le DataFrame est vide. """
    return df.empty
    
@routes.route('/candlestick-data', methods=['POST'])
def candlestick_data():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')

    if not symbol or not timeframe:
        return jsonify({"error": "Symbol et timeframe sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)
    if not timeframe_code:
        mt5.shutdown()
        return jsonify({"error": "Timeframe invalide."}), 400

    df = fetch_data_today(symbol, timeframe_code)
    mt5.shutdown()

    if is_market_closed(df):
        return jsonify({
            "market_status": "closed",
            "message": "Marché actuellement fermé. Essayez pendant les heures de trading."
        }), 403

    candles = df[['Open', 'High', 'Low', 'Close']].copy()
    candles['Time'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    response = candles.to_dict(orient='records')

    return jsonify(response)

@routes.route('/indicator-data', methods=['POST'])
def indicator_data():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')
    indicator = data.get('indicator')

    if not symbol or not timeframe or not indicator:
        return jsonify({"error": "Tous les champs sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)
    if not timeframe_code:
        mt5.shutdown()
        return jsonify({"error": "Timeframe invalide."}), 400

    df = fetch_data_today(symbol, timeframe_code)
    mt5.shutdown()

    if is_market_closed(df):
        return jsonify({ "market_status": "closed",
        "message": "Marché actuellement fermé. Essayez pendant les heures de trading."}), 403

    last_candle = df.iloc[-1]
    open_price = last_candle['Open']
    close_price = last_candle['Close']
    volume = last_candle.get('Tick Volume', 0)
    time_stamp = last_candle.name

    if indicator.upper() == "RSI":
        calculate_rsi(df)
        last_rsi = df["RSI"].iloc[-1]
        tendance = "Bullish" if last_rsi < 30 else "Bearish" if last_rsi > 70 else "Neutre"

        return jsonify({
            "indicator": "RSI",
            "trend": tendance,
            "timestamp": str(time_stamp),
            "RSI": round(last_rsi, 2),
            "open": open_price,
            "close": close_price,
            "volume": volume
        })

    elif indicator.upper() == "MACD":
        calculate_macd(df)
        last_macd = df["MACD"].iloc[-1]
        last_signal = df["MACD_Signal"].iloc[-1]
        tendance = "Bullish" if last_macd > last_signal else "Bearish"

        return jsonify({
            "indicator": "MACD",
            "trend": tendance,
            "timestamp": str(time_stamp),
            "macd_value": round(last_macd, 5),
            "signal_value": round(last_signal, 5),
            "open": open_price,
            "close": close_price,
            "volume": volume
        })

    else:
        return jsonify({"error": "Indicateur non supporté."}), 400

@routes.route('/fvg-data', methods=['POST'])
def fvg_data():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')

    if not symbol or not timeframe:
        return jsonify({"error": "Symbol et timeframe sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)
    if not timeframe_code:
        mt5.shutdown()
        return jsonify({"error": "Timeframe invalide."}), 400

    df = fetch_data_today(symbol, timeframe_code)
    mt5.shutdown()

    if is_market_closed(df):
        return jsonify({ "market_status": "closed",
        "message": "Marché actuellement fermé. Essayez pendant les heures de trading."}), 403

    fvg_list = detect_fvg(df)

    if not fvg_list:
        return jsonify({"message": "Pas de FVG détecté."})

    latest_fvg = fvg_list[-1]
    fvg_type, from_price, to_price, timestamp = latest_fvg

    return jsonify({
        "type": "Bullish" if "Bullish" in fvg_type else "Bearish",
        "strategy": "FVG",
        "from_price": round(from_price, 2),
        "to_price": round(to_price, 2),
        "timestamp": str(timestamp)
    })

@routes.route('/market-signal', methods=['POST'])
def market_signal():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')

    if not symbol or not timeframe:
        return jsonify({"error": "Symbol et timeframe sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)
    if not timeframe_code:
        mt5.shutdown()
        return jsonify({"error": "Timeframe invalide."}), 400

    df = fetch_data_today(symbol, timeframe_code)
    mt5.shutdown()

    if is_market_closed(df):
        return jsonify({"market_status": "closed",
        "message": "Marché actuellement fermé. Essayez pendant les heures de trading."}), 403

    # RSI
    calculate_rsi(df)
    last_rsi = df["RSI"].iloc[-1]
    rsi_trend = "Bullish" if last_rsi < 30 else "Bearish" if last_rsi > 70 else "Neutre"

    # MACD
    calculate_macd(df)
    last_macd = df["MACD"].iloc[-1]
    last_signal = df["MACD_Signal"].iloc[-1]
    macd_trend = "Bullish" if last_macd > last_signal else "Bearish"

    # FVG
    fvg_detected = detect_fvg(df)
    latest_fvg = None
    if fvg_detected:
        fvg_type, from_price, to_price, fvg_timestamp = fvg_detected[-1]
        latest_fvg = {
            "type": "Bullish" if "Bullish" in fvg_type else "Bearish",
            "from_price": round(from_price, 2),
            "to_price": round(to_price, 2),
            "timestamp": str(fvg_timestamp)
        }

    return jsonify({
        "RSI": {
            "value": round(last_rsi, 2),
            "trend": rsi_trend
        },
        "MACD": {
            "macd_value": round(last_macd, 5),
            "signal_value": round(last_signal, 5),
            "trend": macd_trend
        },
        "FVG": latest_fvg if latest_fvg else "No FVG detected"
    })

@routes.route('/full-analysis', methods=['POST'])
def full_analysis():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')
    indicator = data.get('indicator')

    if not symbol or not timeframe or not indicator:
        return jsonify({"error": "Tous les champs sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)
    if not timeframe_code:
        mt5.shutdown()
        return jsonify({"error": "Timeframe invalide."}), 400

    mapped_symbol = SYMBOL_MAPPING.get(symbol)
    if not mapped_symbol:
        mt5.shutdown()
        return jsonify({"error": "Symbole invalide."}), 400

    # Essayer de récupérer les données d'aujourd'hui
    df = fetch_data_today(mapped_symbol, timeframe_code)
    mt5.shutdown()

    # Si pas de données ou marché fermé → snapshot
    if df.empty or not is_market_open():
        last_snapshot = fetch_last_snapshot(mapped_symbol, timeframe, indicator)
        if last_snapshot is None:
            return jsonify({
                "market_status": "closed",
                "message": "Market is closed. Displaying last available data and prediction."
            }), 403

        return jsonify({
            "market_status": "closed",
            "candles": last_snapshot['candle_data'],
            "indicator_value": last_snapshot['indicator_value'],
            "trend": last_snapshot['trend'],
            "signal_info": last_snapshot['signal_info']
        })

    # Si des données et marché ouvert → calculs en live
    candles = df[['Open', 'High', 'Low', 'Close']].copy()
    candles['Time'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    candle_data = candles.to_dict(orient='records')

    last_value = None
    trend = "Neutral"
    signal_info = None

    if indicator.upper() == "RSI":
        calculate_rsi(df)
        last_value = round(df['RSI'].iloc[-1], 2)
        if math.isnan(last_value):
            last_value = None
        if last_value is not None:
            if last_value < 30:
                trend = "Bullish"
                signal_info = {
                    "signal": "Buy",
                    "timestamp": df.index[-1],
                    "entry": df['Close'].iloc[-1],
                    "sl": 0.5,
                }
            elif last_value > 70:
                trend = "Bearish"
                signal_info = {
                    "signal": "Sell",
                    "timestamp": df.index[-1],
                    "entry": df['Close'].iloc[-1],
                    "sl": 0.5,
                }

    elif indicator.upper() == "MACD":
        calculate_macd(df)
        last_macd = round(df["MACD"].iloc[-1], 5)
        last_signal = round(df["MACD_Signal"].iloc[-1], 5)
        if math.isnan(last_macd):
            last_macd = None
        if math.isnan(last_signal):
            last_signal = None
        if last_macd is not None and last_signal is not None:
            trend = "Bullish" if last_macd > last_signal else "Bearish"
            signal_info = {
                "signal": "Buy" if last_macd > last_signal else "Sell",
                "timestamp": df.index[-1],
                "entry": df['Close'].iloc[-1],
                "sl": 0.5,
            }
        last_value = last_macd

    else:
        return jsonify({"error": "Indicateur non supporté."}), 400

    if signal_info:
        sl, tp = calculate_tp_sl(signal_info['signal'], signal_info['entry'], signal_info['sl'], risk_reward_ratio=2)
        signal_info['stop_loss'] = sl
        signal_info['take_profit'] = tp

    # Sauvegarder le snapshot live
    save_snapshot(mapped_symbol, timeframe, indicator, candle_data, last_value, trend, signal_info)

    return jsonify({
        "market_status": "open",
        "candles": candle_data,
        "indicator_value": last_value,
        "trend": trend,
        "signal_info": signal_info
    })

@routes.route('/settings', methods=['GET'])
def get_settings():
    return jsonify({
        "symbols": list(SYMBOL_MAPPING.keys()),
        "timeframes": list(TIMEFRAME_MAPPING.keys()),
        "indicators": INDICATORS
    })
@routes.route('/lstm-prediction', methods=['POST'])
def lstm_prediction():
    print("Requête reçue /lstm-prediction")
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')
    indicator = data.get('indicator')  # Nouveau champ
    print(f"Symbole: {symbol}, Timeframe: {timeframe}, Indicateur: {indicator}")

    if not symbol or not timeframe:
        return jsonify({"error": "Symbole et timeframe sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    mapped_symbol = SYMBOL_MAPPING.get(symbol)
    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)

    if not mapped_symbol or not timeframe_code:
        mt5.shutdown()
        return jsonify({"error": "Symbole ou timeframe invalide."}), 400

    df = fetch_data_today(mapped_symbol, timeframe_code)
    print("Données récupérées:", df.tail())
    print("Taille des données récupérées:", len(df))

    mt5.shutdown()

    if df.empty or len(df) < 5:
        return jsonify({"error": "Pas assez de données pour la prédiction."}), 403

    result = load_and_predict_lstm(df, symbol, timeframe)
    predicted_price = result.get("predicted_price", 0)
    current_price = result.get("current_price", 0)
    trend = result.get("trend", "UNKNOWN")

    # Derniers points d'entrée utilisés par le modèle (prédiction passée)
    last_points_prediction = df[['Close']].tail(10).copy()
    last_points_prediction['Time'] = df.index[-10:].strftime('%Y-%m-%d %H:%M:%S')
    last_points_prediction = last_points_prediction.to_dict(orient='records')

    # Bougies OHLC + Time
    candles = df[['Open', 'High', 'Low', 'Close']].copy()
    candles['Time'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    candles = candles.to_dict(orient='records')

    # Bande de fluctuation (±1% autour de la prédiction)
    fluctuation_range = 0.01 * predicted_price
    fluctuation_upper = round(predicted_price + fluctuation_range, 3)
    fluctuation_lower = round(predicted_price - fluctuation_range, 3)

    # Réponse finale
    response = {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "trend": trend,
        "indicator": indicator,
        "candles": candles,
        "last_points_prediction": last_points_prediction,
        "lstm_prediction": predicted_price,
        "fluctuation_upper": fluctuation_upper,
        "fluctuation_lower": fluctuation_lower
    }

    print("Résultat de la prédiction enrichie:", response)
    return jsonify(response)


@routes.route('/full-lstm-analysis', methods=['POST'])
def full_lstm_analysis():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')
    indicator = data.get('indicator')

    if not symbol or not timeframe or not indicator:
        return jsonify({"error": "Tous les champs sont requis."}), 400

    if not initialize_mt5():
        return jsonify({"error": "Connexion MT5 impossible."}), 500

    timeframe_code = TIMEFRAME_MAPPING.get(timeframe)
    mapped_symbol = SYMBOL_MAPPING.get(symbol)

    if not timeframe_code or not mapped_symbol:
        mt5.shutdown()
        return jsonify({"error": "Symbole ou timeframe invalide."}), 400

    df = fetch_data_today(mapped_symbol, timeframe_code)
    mt5.shutdown()

    # Fallback si marché fermé
    if df.empty or not is_market_open():
        last_snapshot = fetch_last_snapshot(mapped_symbol, timeframe, indicator)
        if last_snapshot is None:
            return jsonify({
                "market_status": "closed",
                "message": "Market is closed. Displaying last available data and prediction."
            }), 403

        return jsonify({
            "market_status": "closed",
            "candles": last_snapshot['candle_data'],
            "indicator_value": last_snapshot['indicator_value'],
            "trend": last_snapshot['trend'],
            "signal_info": last_snapshot['signal_info']
        })

    # === Données bougies ===
    candles = df[['Open', 'High', 'Low', 'Close']].copy()
    candles['Time'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    candle_data = candles.to_dict(orient='records')

    # === Calcul de l'indicateur ===
    last_value = None
    trend = "Neutral"
    signal_info = None

    if indicator.upper() == "RSI":
        calculate_rsi(df)
        last_value = round(df['RSI'].iloc[-1], 2)
        if not math.isnan(last_value):
            if last_value < 30:
                trend = "Bullish"
                signal_info = {
                    "signal": "Buy",
                    "timestamp": df.index[-1],
                    "entry": df['Close'].iloc[-1],
                    "sl": 0.5,
                }
            elif last_value > 70:
                trend = "Bearish"
                signal_info = {
                    "signal": "Sell",
                    "timestamp": df.index[-1],
                    "entry": df['Close'].iloc[-1],
                    "sl": 0.5,
                }

    elif indicator.upper() == "MACD":
        calculate_macd(df)
        last_macd = round(df["MACD"].iloc[-1], 5)
        last_signal = round(df["MACD_Signal"].iloc[-1], 5)
        if not math.isnan(last_macd) and not math.isnan(last_signal):
            trend = "Bullish" if last_macd > last_signal else "Bearish"
            signal_info = {
                "signal": "Buy" if last_macd > last_signal else "Sell",
                "timestamp": df.index[-1],
                "entry": df['Close'].iloc[-1],
                "sl": 0.5,
            }
            last_value = last_macd
    else:
        return jsonify({"error": "Indicateur non supporté."}), 400

    if signal_info:
        sl, tp = calculate_tp_sl(signal_info['signal'], signal_info['entry'], signal_info['sl'], risk_reward_ratio=2)
        signal_info['stop_loss'] = sl
        signal_info['take_profit'] = tp

    # === Prédiction LSTM ===
    if len(df) < 5:
        return jsonify({"error": "Pas assez de données pour la prédiction."}), 403

    result = load_and_predict_lstm(df, symbol, timeframe)
    predicted_price = result.get("predicted_price", 0)
    current_price = result.get("current_price", 0)
    lstm_trend = result.get("trend", "UNKNOWN")

    last_points_prediction = df[['Close']].tail(10).copy()
    last_points_prediction['Time'] = df.index[-10:].strftime('%Y-%m-%d %H:%M:%S')
    last_points_prediction = last_points_prediction.to_dict(orient='records')

    fluctuation_range = 0.01 * predicted_price
    fluctuation_upper = round(predicted_price + fluctuation_range, 3)
    fluctuation_lower = round(predicted_price - fluctuation_range, 3)

    # === Snapshot (optionnel) ===
    save_snapshot(mapped_symbol, timeframe, indicator, candle_data, last_value, trend, signal_info)

    return jsonify({
        "candles": candle_data,
        "indicator": indicator,
        "indicator_value": last_value,
        "indicator_trend": trend,
        "signal_info": signal_info,
        "lstm_prediction": {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "trend": lstm_trend,
            "last_points_prediction": last_points_prediction,
            "fluctuation_data": result.get("fluctuation_data", [])

        }
    })
