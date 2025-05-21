from flask import Blueprint, request, jsonify
import MetaTrader5 as mt5
from prediction import (
    init_mt5, fetch_data, get_entry_price,
     SYMBOL_MAPPING, TIMEFRAME_MAPPING,
    load_lstm_model, load_scaler, compute_rsi, compute_macd,
    WINDOW_SIZE,get_ohlc, calculate_tp_sl, get_signal,  ensure_lstm_model_exists, calculate_fluctuation_levels, generate_future_predictions)
import numpy as np
import pandas as pd
from utils import is_market_open_close
from db import last_snapshots_collection
from datetime import datetime
from copy import deepcopy

prediction_route = Blueprint('prediction_route', __name__)

@prediction_route.route('/analyze', methods=['POST'])
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

     # Vérification du statut du marché
    market_info = is_market_open_close(symbol_code)

    # 📌 Si le marché est fermé, récupérer la dernière snapshot
    if not market_info["open"]:
        last_saved_snapshot = last_snapshots_collection.find_one(
            {
                "symbol": symbol_code,
                "timeframe": timeframe_str,
                "indicator_used": indicator
            },
            sort=[("timestamp", -1)]
        )

        if last_saved_snapshot:
            last_saved_snapshot["_id"] = str(last_saved_snapshot["_id"])
            return jsonify({
                "message": f"⚠️ The market is currently closed. It will reopen on {market_info['next_open']}.",
                "last_snapshot": last_saved_snapshot
            })
        else:
            return jsonify({
                "message": (
                    f"❌ No snapshot found for symbol {symbol_code} "
                    f"with timeframe {timeframe_str} and indicator {indicator}. "
                    f"\n⚠️ The market is currently closed. It will reopen on {market_info['next_open']}."
                ),
                "last_snapshot": None
            })

    # 📌 Si le marché est ouvert, effectuer l'analyse en temps réel
    df_data = fetch_data(symbol_code, timeframe_code, count=300)
    if df_data is None or df_data.empty:
        return jsonify({"error": "No data from MT5"}), 400

    # Apply indicators
    df_with_indicators = df_data.copy()
    if indicator == "rsi":
        df_with_indicators = compute_rsi(df_with_indicators)
    elif indicator == "macd":
        df_with_indicators = compute_macd(df_with_indicators)
    
    
    ensure_lstm_model_exists(symbol_code, timeframe_str, indicator)
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

    raw_predictions_scaled = model.predict(input_model_scaled)
    # Assure-toi que c’est bien un tableau 2D (ex: (1, 10))
    if raw_predictions_scaled.ndim == 2:
        raw_predictions_scaled = raw_predictions_scaled[0]
    
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

    fluctuation_info = calculate_fluctuation_levels(df_data, window=5)
    last_candle_time = df_data["time"].iloc[-1]
    # Intégration de la fonction generate_future_predictions
    future_predictions = generate_future_predictions(last_candle_time, future_price_predictions, num_steps=10, time_delta_minutes=5)

    time_delta_per_step = pd.Timedelta(minutes=0)
    if "H" in timeframe_str: time_delta_per_step = pd.Timedelta(hours=int(timeframe_str.replace("H", "")))
    elif "M" in timeframe_str: time_delta_per_step = pd.Timedelta(minutes=int(timeframe_str.replace("M", "")))

    response = {
    "symbol": symbol_code,
    "timeframe": timeframe_str,
    "indicator_used": indicator,
    "real_price": current_price,
    "entry_price": entry_price_val,
    "ohlc": get_ohlc(df_data, num_candles=50),  # Ensure this is correctly formatted
    "signal": signal_val,
    "trend": trend,
    "tp": round(tp_val, 5),
    "sl": round(sl_val, 5),
    "future_prediction": future_predictions["future_prediction"],
    "last_prediction": future_predictions["last_prediction"],
    "future_timestamps": future_predictions["future_timestamps"],
    "fluctuation": {
        "upper": round(fluctuation_info["upper"], 5),
        "lower": round(fluctuation_info["lower"], 5),
        "median": round(fluctuation_info["median"], 5),
        "fluctuation": round(fluctuation_info["fluctuation"], 5),
        "percent_fluctuation": round(fluctuation_info["percent_fluctuation"], 2)
    },
    "current_candle_time": last_candle_time.strftime("%Y-%m-%d %H:%M:%S"),
    "market_open": market_info["open"],
    "next_open": market_info["next_open"]
    }

     # ✅ Crée un snapshot complet basé sur la réponse
    last_snapshot = deepcopy(response)
    last_snapshot["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ Sauvegarde dans MongoDB
    last_snapshots_collection.insert_one(last_snapshot)

    return jsonify(response)

@prediction_route.route('/last-snapshot/<symbol>/<timeframe>/<indicator_used>', methods=['GET'])
def get_last_snapshot(symbol, timeframe, indicator_used):
    snapshot = last_snapshots_collection.find_one(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicator_used": indicator_used
        },
        sort=[("timestamp", -1)]
    )
    if snapshot:
        snapshot["_id"] = str(snapshot["_id"])  # Convertir l'ObjectId pour JSON
        return jsonify(snapshot)
    return jsonify({"error": "No snapshot found"}), 404