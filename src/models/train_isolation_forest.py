# src/models/train_isolation_forest.py
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.features_utils import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    build_feature_matrix,
    fit_routine_profile,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"


def load_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = BASE_DIR / "data" / "simulated_data_1year.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el dataset en: {csv_path}. "
            "Ejecuta 'python -m src.simulate_yearly' primero."
        )
    return pd.read_csv(csv_path, parse_dates=["timestamp"])


def fit_isolation_forest(
    data: pd.DataFrame,
    contamination: float = 0.004,
    n_estimators: int = 200,
    max_samples: int = 2048,
    random_state: int = 42,
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """Ajusta el perfil no supervisado de rutinas y entrena el Isolation Forest.

    Nota técnica:
    - El perfil horario por (hora, fin_de_semana, presencia) se ajusta únicamente sobre
      `data` (sin etiquetas ni fuga temporal), convirtiendo sensores continuos a excesos
      fuera de [P01, P99] y sensores binarios a autoinformación (-log P).
    - Al ser un algoritmo basado en árboles aleatorios paralelos a los ejes, es invariante
      ante transformaciones afines por columna (StandardScaler resulta redundante).
    """
    profile = fit_routine_profile(data)
    features = build_feature_matrix(data, profile=profile)

    effective_max_samples = min(max_samples, len(features))
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=effective_max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features)

    # decision_function ya centra la frontera de decisión en 0.0 según 'contamination'
    scores = model.decision_function(features)
    meta: Dict[str, Any] = {
        "raw_columns": list(FEATURE_COLUMNS),
        "model_columns": list(MODEL_FEATURE_COLUMNS),
        "profile": profile,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "max_samples": effective_max_samples,
        "random_state": random_state,
        "threshold": 0.0,
        "training_rows": int(len(features)),
        "score_min": float(np.min(scores)),
        "score_p01": float(np.percentile(scores, 1.0)),
        "score_median": float(np.median(scores)),
    }
    return model, meta


def train_and_save(
    data: pd.DataFrame,
    contamination: float = 0.004,
    n_estimators: int = 200,
    max_samples: int = 2048,
    random_state: int = 42,
    model_dir: Optional[Path] = None,
) -> Tuple[IsolationForest, Dict[str, Any]]:
    target_dir = model_dir if model_dir is not None else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Entrenando Isolation Forest (%d filas, %d features del modelo)...",
        len(data),
        len(MODEL_FEATURE_COLUMNS),
    )
    model, meta = fit_isolation_forest(
        data,
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    )

    joblib.dump(model, target_dir / "iforest.joblib")
    joblib.dump(meta, target_dir / "meta.joblib")
    logger.info("Entrenamiento completado. Artefactos guardados en %s", target_dir)
    return model, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_df = load_data()
    train_and_save(data_df)
