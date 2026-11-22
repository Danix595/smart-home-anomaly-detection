# src/models/detector.py
import logging
from pathlib import Path
from typing import Dict, Any

import joblib

from src.models.features_utils import make_features_from_row

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

_MODEL = None
_SCALER = None
_META = None

# Carga el modelo si no esta cargado
def get_model_cached():
    global _MODEL, _SCALER, _META
    if _MODEL is None:
        logger.info("Cargando modelo desde disco...")
        model_path = MODEL_DIR / "iforest.joblib"
        scaler_path = MODEL_DIR / "scaler.joblib"
        meta_path = MODEL_DIR / "meta.joblib"

        if not model_path.exists():
            raise FileNotFoundError("Modelo no encontrado. Ejecuta train_isolation_forest.py primero.")

        _MODEL = joblib.load(model_path)
        _SCALER = joblib.load(scaler_path)
        _META = joblib.load(meta_path)
    return _MODEL, _SCALER, _META


def predict_single(timestamp: str, water: float, power: float, presence: int, motion: int, window_open: int) -> Dict[str, Any]:
    model, scaler, meta = get_model_cached()

    # 1. Preparar datos para IA
    features = make_features_from_row(timestamp, water, power, presence, motion, window_open)
    features_scaled = scaler.transform(features)

    # 2. Predicción
    score = float(model.decision_function(features_scaled)[0])

    # 3. Leer umbral
    thresh = meta.get("threshold")

    if thresh is not None:
        is_anom = score < thresh
        return {
            "prediction": -1 if is_anom else 1,
            "score": score,
            "threshold": thresh,
            "reason": "ml_score_threshold" if is_anom else "normal"
        }

    else:
        # Fallback de seguridad por si el metadata es antiguo
        pred = int(model.predict(features_scaled)[0])
        return {
            "prediction": pred,
            "score": score,
            "reason": "ml_default_predict"
        }