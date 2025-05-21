# fichier: lstm_predictor.py

import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data_for_lstm(df, lookback=10):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def get_model_paths(symbol, timeframe):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{timeframe}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_{timeframe}_scaler.save")
    return model_path, scaler_path

def train_and_predict_lstm(df, symbol, timeframe, lookback=10):
    if len(df) < lookback + 1:
        return {"error": "Pas assez de données pour entraîner le modèle."}

    X, y, scaler = prepare_data_for_lstm(df, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)

    model_path, scaler_path = get_model_paths(symbol, timeframe)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    last_sequence = X[-1].reshape((1, lookback, 1))
    predicted = model.predict(last_sequence, verbose=0)
    predicted_price = float(scaler.inverse_transform(predicted)[0][0])
    current_price = float(df['Close'].iloc[-1])

    trend = "UP" if predicted_price > current_price else "DOWN"
    return {
        "current_price": round(current_price, 3),
        "predicted_price": round(predicted_price, 3),
        "trend": trend
    }

def load_and_predict_lstm(df, symbol, timeframe, lookback=10):
    model_path, scaler_path = get_model_paths(symbol, timeframe)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return train_and_predict_lstm(df, symbol, timeframe, lookback)

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    data = df[['Close']].values
    scaled_data = scaler.transform(data)

    X = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
    if not X:
        return {"error": "Pas assez de données pour la prédiction."}

    X = np.array(X).reshape((-1, lookback, 1))
    last_sequence = X[-1].reshape((1, lookback, 1))

    predicted = model.predict(last_sequence, verbose=0)
    predicted_price = float(scaler.inverse_transform(predicted)[0][0])
    current_price = float(df['Close'].iloc[-1])

    trend = "UP" if predicted_price > current_price else "DOWN"

    # Nouvelle prédiction future (sur plusieurs points)
    future_predictions = []
    sequence = X[-1]
    for _ in range(5):  # prédire 5 points futurs
        pred = model.predict(sequence.reshape(1, lookback, 1), verbose=0)
        future_predictions.append(float(scaler.inverse_transform(pred)[0][0]))
        sequence = np.append(sequence[1:], pred, axis=0)

    # Prédiction passée (restaurée depuis les X utilisés)
    last_points_prediction = [float(scaler.inverse_transform(model.predict(x.reshape(1, lookback, 1), verbose=0))[0][0]) for x in X[-5:]]

    # Bande de fluctuation simple : +/-1% autour de chaque valeur future
    fluctuation_upper = [round(p * 1.01, 3) for p in future_predictions]
    fluctuation_lower = [round(p * 0.99, 3) for p in future_predictions]

     # Générer les timestamps futurs pour les fluctuations
    last_time = df.index[-1]
    time_interval = df.index[-1] - df.index[-2] if len(df) >= 2 else pd.Timedelta(minutes=1)

    fluctuation_data = []
    for i in range(5):
        fluctuation_data.append({
            "Time": (last_time + (i + 1) * time_interval).strftime('%Y-%m-%d %H:%M:%S'),
            "Upper": fluctuation_upper[i],
            "Lower": fluctuation_lower[i]
        })
    return {
        "current_price": round(current_price, 3),
        "predicted_price": round(predicted_price, 3),
        "trend": trend,
        "last_points_prediction": [round(p, 3) for p in last_points_prediction],
        "lstm_prediction": [round(p, 3) for p in future_predictions],
        "fluctuation_upper": fluctuation_upper,
        "fluctuation_lower": fluctuation_lower,
        "fluctuation_data": fluctuation_data
    }