# src/models/detector.py
import logging
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

import joblib

from src.models.features_utils import (
    MODEL_FEATURE_COLUMNS,
    make_features_from_row,
    make_features_from_state,
)

logger = logging.getLogger("tfg_domotica")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

_MODEL: Any = None
_META: Optional[Dict[str, Any]] = None
_MODEL_LOCK = threading.Lock()


def reset_model_cache() -> None:
    """Limpia la caché en memoria del modelo (útil en tests o recargas)."""
    global _MODEL, _META
    with _MODEL_LOCK:
        _MODEL = None
        _META = None


def get_model_cached(model_dir: Optional[Path] = None) -> Tuple[Any, Dict[str, Any]]:
    """Carga el modelo entrenado de forma segura para hilos y valida esquema y perfil."""
    global _MODEL, _META
    target_dir = model_dir if model_dir is not None else MODEL_DIR

    if _MODEL is not None and _META is not None and model_dir is None:
        return _MODEL, _META

    with _MODEL_LOCK:
        if _MODEL is not None and _META is not None and model_dir is None:
            return _MODEL, _META

        model_path = target_dir / "iforest.joblib"
        meta_path = target_dir / "meta.joblib"

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {target_dir}. "
                "Ejecuta 'python -m src.models.train_isolation_forest' primero."
            )

        logger.info("Cargando modelo Isolation Forest desde %s...", target_dir)
        model = joblib.load(model_path)
        # En inferencia de 1 sola lectura, n_jobs=1 evita el overhead de joblib/hilos
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        meta = joblib.load(meta_path)

        saved_cols = meta.get("model_columns")
        if saved_cols != MODEL_FEATURE_COLUMNS or "profile" not in meta:
            raise ValueError(
                "Los artefactos del modelo están desactualizados respecto al esquema actual. "
                "Reentrena ejecutando 'python -m src.models.train_isolation_forest'."
            )

        if model_dir is None:
            _MODEL = model
            _META = meta
        return model, meta


def preload_model() -> bool:
    """Intenta precargar el modelo al arrancar la API sin impedir el arranque si aún no existe."""
    try:
        get_model_cached()
        return True
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("%s (las reglas deterministas seguirán operativas).", exc)
        return False


def _score(model: Any, meta: Dict[str, Any], features) -> Dict[str, Any]:
    score = float(model.decision_function(features)[0])
    thresh = float(meta.get("threshold", 0.0))
    is_anom = score < thresh
    return {
        "prediction": -1 if is_anom else 1,
        "score": score,
        "threshold": thresh,
        "reason": "ml_score_threshold" if is_anom else "normal",
    }


def predict_single(
    timestamp: str,
    water: float,
    power: float,
    presence: int,
    motion: int,
    window_open: int,
    door_open: int = 0,
    heating_on: int = 0,
    fp2_active: int = 0,
    model_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    model, meta = get_model_cached(model_dir=model_dir)
    features = make_features_from_row(
        timestamp=timestamp,
        water=water,
        power=power,
        presence=presence,
        motion=motion,
        window_open=window_open,
        door_open=door_open,
        heating_on=heating_on,
        fp2_active=fp2_active,
        profile=meta["profile"],
    )
    return _score(model, meta, features)


def predict_from_state(
    timestamp: str,
    state: Dict[str, Any],
    model_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    model, meta = get_model_cached(model_dir=model_dir)
    features = make_features_from_state(timestamp, state, profile=meta["profile"])
    return _score(model, meta, features)
