# src/models/train_isolation_forest.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Optional
from src.models.features_utils import add_time_features_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = BASE_DIR / "data" / "simulated_data_1year.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el dataset en: {csv_path}. Ejecuta 'python src/simulate_yearly.py' primero.")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df


def build_feature_matrix(data: pd.DataFrame) -> np.ndarray:
    cols = ["water_L_per_min", "power_W", "presence", "motion", "window_open", "hour_sin", "hour_cos", "weekday"]
    return data[cols].values


def train_and_save(data: pd.DataFrame, contamination: float = 0.005, random_state: int = 42,
                   model_dir: Optional[Path] = None) -> None:
    if model_dir is None:
        model_dir = MODEL_DIR

    df_with_time = add_time_features_df(data)
    features = build_feature_matrix(df_with_time)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Entrenar el modelo
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(features_scaled)

    # Cálculo del umbral dinámico basado en los scores del entrenamiento
    logger.info("Calculando umbral dinámico...")
    scores = model.decision_function(features_scaled)
    threshold = float(np.percentile(scores, 100.0 * contamination))
    logger.info(f"Umbral calculado: {threshold}")

    model_path = model_dir / "iforest.joblib"
    scaler_path = model_dir / "scaler.joblib"
    meta_path = model_dir / "meta.joblib"

    joblib.dump(model, str(model_path))
    joblib.dump(scaler, str(scaler_path))
    joblib.dump({
        "columns": ["water_L_per_min", "power_W", "presence", "motion", "window_open", "hour_sin", "hour_cos", "weekday"],
        "contamination": contamination,
        "random_state": random_state,
        "threshold": threshold
    }, str(meta_path))

    logger.info("Entrenamiento completado y modelos guardados.")

if __name__ == "__main__":
    data_df = load_data()
    train_and_save(data_df)