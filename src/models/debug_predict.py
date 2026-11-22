# src/models/debug_predict.py
import argparse
from pathlib import Path

import pandas as pd

from src.models.detector import MODEL_DIR, get_model_cached, predict_single
from src.models.features_utils import (
    MODEL_FEATURE_COLUMNS,
    build_feature_matrix,
    make_features_from_row,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = BASE_DIR / "data" / "simulated_data_1year.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspecciona el score de Isolation Forest para una lectura individual."
    )
    parser.add_argument("timestamp", help="Timestamp ISO-8601 (ej. 2026-07-02T03:00:00)")
    parser.add_argument("water", type=float, help="Caudal de agua (L/min)")
    parser.add_argument("power", type=float, help="Potencia global (W)")
    parser.add_argument("presence", type=int, choices=[0, 1], help="Presencia (0 o 1)")
    parser.add_argument("motion", type=int, choices=[0, 1], help="Movimiento (0 o 1)")
    parser.add_argument("window_open", type=int, choices=[0, 1], help="Ventana abierta (0 o 1)")
    parser.add_argument("--door-open", type=int, choices=[0, 1], default=0, help="Puerta abierta (0 o 1)")
    parser.add_argument("--heating-on", type=int, choices=[0, 1], default=0, help="Calefacción ON (0 o 1)")
    parser.add_argument("--fp2-active", type=int, choices=[0, 1], default=0, help="Presencia estática FP2 (0 o 1)")
    args = parser.parse_args()

    model, meta = get_model_cached()
    print(f"Modelo cargado desde: {MODEL_DIR}")
    print(f"Features del modelo ({len(MODEL_FEATURE_COLUMNS)}): {MODEL_FEATURE_COLUMNS}")

    x_vec = make_features_from_row(
        timestamp=args.timestamp,
        water=args.water,
        power=args.power,
        presence=args.presence,
        motion=args.motion,
        window_open=args.window_open,
        door_open=args.door_open,
        heating_on=args.heating_on,
        fp2_active=args.fp2_active,
        profile=meta["profile"],
    )
    print("Vector contextual:", dict(zip(MODEL_FEATURE_COLUMNS, [round(v, 4) for v in x_vec[0]])))

    result = predict_single(
        timestamp=args.timestamp,
        water=args.water,
        power=args.power,
        presence=args.presence,
        motion=args.motion,
        window_open=args.window_open,
        door_open=args.door_open,
        heating_on=args.heating_on,
        fp2_active=args.fp2_active,
    )
    print(
        f"Predicción: {result['prediction']} ({result['reason']}) | "
        f"score={result['score']:.6f} | threshold={result['threshold']:.6f}"
    )

    if TRAIN_CSV.exists():
        df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
        scores = model.decision_function(build_feature_matrix(df, profile=meta["profile"]))
        pct = float((scores < result["score"]).mean() * 100.0)
        print(f"Contamination: {meta.get('contamination')}")
        print(f"Percentil respecto al dataset ({len(df)} filas): {pct:.4f}%")
    else:
        print(f"Dataset no encontrado en {TRAIN_CSV}; omitido cálculo de percentil.")


if __name__ == "__main__":
    main()
