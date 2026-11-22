import numpy as np
import pandas as pd
from typing import Dict, Any

def make_time_features(timestamp: str) -> Dict[str, Any]:
    ts = pd.to_datetime(timestamp)
    hour = ts.hour + ts.minute / 60.0
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    weekday = float(ts.weekday())
    return {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "weekday": weekday
    }

def make_features_from_row(timestamp: str, water: float, power: float, presence: int, motion: int, window_open: int) -> np.ndarray:
    time_features = make_time_features(timestamp)
    features = np.array([[
        float(water),
        float(power),
        float(presence),
        float(motion),
        float(window_open),
        time_features["hour_sin"],
        time_features["hour_cos"],
        time_features["weekday"]
    ]], dtype=float)
    return features

def add_time_features_df(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday"] = df["timestamp"].dt.weekday
    return df