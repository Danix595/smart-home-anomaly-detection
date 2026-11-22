# src/models/debug_predict.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "models"
TRAIN_CSV = "data/simulated_data_1year.csv"

def load_model():
    model = joblib.load(Path(MODEL_DIR) / "iforest.joblib")
    scaler = joblib.load(Path(MODEL_DIR) / "scaler.joblib")
    meta = joblib.load(Path(MODEL_DIR) / "meta.joblib")
    return model, scaler, meta

def make_features_row(timestamp, water, power, presence, motion, window_open):
    ts = pd.to_datetime(timestamp)
    hour = ts.hour + ts.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    weekday = ts.dayofweek

    x = np.array([[water, power, presence, motion, window_open, hour_sin, hour_cos, weekday]], dtype=float)
    info = {"hour": hour, "hour_sin": hour_sin, "hour_cos": hour_cos, "weekday": weekday}

    return x, info

def training_scores_threshold(model, scaler, contamination=None):
    if not Path(TRAIN_CSV).exists():
        print("No se encontró el CSV de entrenamiento:", TRAIN_CSV)
        return None, None

    df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday"] = df["timestamp"].dt.weekday

    x = df[["water_L_per_min", "power_W", "presence", "motion", "window_open", "hour_sin", "hour_cos", "weekday"]].values
    xs = scaler.transform(x)
    scores = model.decision_function(xs)

    if contamination is None:
        contamination = getattr(model, "contamination", None)

    if contamination is None:
        contamination = 0.01

    thresh = np.percentile(scores, 100.0 * contamination)

    return scores, thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timestamp")
    parser.add_argument("water", type=float)
    parser.add_argument("power", type=float)
    parser.add_argument("presence", type=int)
    parser.add_argument("motion", type=int)
    parser.add_argument("window_open", type=int)
    args = parser.parse_args()

    model, scaler, meta = load_model()
    print("Cargado modelo y scaler desde:", MODEL_DIR)

    x_raw, info = make_features_row(args.timestamp, args.water, args.power, args.presence, args.motion, args.window_open)
    print("Raw features:", x_raw.tolist())

    xs = scaler.transform(x_raw)
    print("Scaled features:", xs.tolist())

    pred = model.predict(xs)[0]
    score = model.decision_function(xs)[0]
    print("Model predict:", int(pred), "decision_function score:", float(score))

    scores, thresh = training_scores_threshold(model, scaler)
    if scores is not None:
        print(f"Contamination (modelo): {getattr(model, 'contamination', None)}")
        print(f"Umbral calculado desde training: {thresh:.6f}")
        pct = (scores < score).mean() * 100.0
        print(f"Porcentaje de puntos de entrenamiento con score menor que este: {pct:.4f}%")
    else:
        print("No se pudo calcular umbral por falta de CSV de entrenamiento.")

if __name__ == "__main__":
    main()