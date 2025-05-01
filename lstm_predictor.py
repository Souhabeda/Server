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
    return {
        "current_price": round(current_price, 3),
        "predicted_price": round(predicted_price, 3),
        "trend": trend
    }
