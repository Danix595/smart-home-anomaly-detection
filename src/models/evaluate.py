# src/models/evaluate.py
"""Evaluación reproducible sin fuga temporal del modelo Isolation Forest.

Split cronológico 80 % entrenamiento (solo rutinas normales) / 20 % test,
inyectando en el conjunto de test 6 escenarios de anomalías conductuales que
NO activan las reglas deterministas (evaluando así el aporte real de la capa IA).
"""

import argparse
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.features_utils import build_feature_matrix
from src.models.train_isolation_forest import fit_isolation_forest, load_data
from src.simulate_yearly import simulate_yearly_data


def inject_behavioral_anomalies(
    df_test: pd.DataFrame,
    anomaly_ratio: float = 0.02,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Inyecta anomalías conductuales que superan las reglas deterministas de rules.py.

    Escenarios inyectados (todos por debajo de los umbrales de rules.py):
      1. night_water_leak: caudal moderado (4,5–12 L/min) de madrugada (01:00–05:00).
      2. away_or_night_power: consumo elevado (1200–2800 W) fuera de casa o de madrugada.
      3. night_door_or_window: puerta o ventana abierta de madrugada con presencia y sin calefacción.
      4. heating_unusual: calefacción encendida fuera de casa o de madrugada (ventanas cerradas).
      5. night_activity_burst: actividad inusual de madrugada (movimiento + potencia + agua).
      6. daytime_fp2_static_power: presencia estática FP2 diurna inusual + calefactor alto (1500–2500 W).
    """
    rng = np.random.default_rng(seed)
    df = df_test.copy().reset_index(drop=True)
    n = len(df)
    y_true = np.zeros(n, dtype=int)
    scenario = np.full(n, "normal", dtype=object)

    hours = (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0).to_numpy()
    night_idx = np.where((hours >= 1.0) & (hours < 5.0))[0]
    away_idx = np.where(df["presence"].to_numpy() == 0)[0]
    day_home_idx = np.where((df["presence"].to_numpy() == 1) & (hours >= 10.0) & (hours < 17.0))[0]

    total_target = max(6, int(round(n * anomaly_ratio)))
    per_scenario = max(1, total_target // 6)
    used: set[int] = set()

    def sample_indices(pool: np.ndarray, count: int) -> np.ndarray:
        available = np.array([i for i in pool if i not in used], dtype=int)
        if len(available) == 0:
            return np.array([], dtype=int)
        chosen = rng.choice(available, size=min(count, len(available)), replace=False)
        used.update(chosen.tolist())
        return chosen

    # 1. Fuga moderada / uso de agua de madrugada (4,5–12 L/min < 15 L/min)
    idx1 = sample_indices(night_idx, per_scenario)
    if len(idx1):
        df.loc[idx1, "water_L_per_min"] = np.round(rng.uniform(4.5, 12.0, size=len(idx1)), 3)
        df.loc[idx1, "power_W"] = np.round(rng.uniform(50.0, 120.0, size=len(idx1)), 2)
        y_true[idx1] = 1
        scenario[idx1] = "night_water_leak"

    # 2. Consumo eléctrico alto de madrugada o con casa vacía (1200–2800 W < 3500 W)
    pool2 = np.union1d(night_idx, away_idx)
    idx2 = sample_indices(pool2, per_scenario)
    if len(idx2):
        df.loc[idx2, "window_open"] = 0
        df.loc[idx2, "door_open"] = 0
        df.loc[idx2, "power_W"] = np.round(rng.uniform(1200.0, 2800.0, size=len(idx2)), 2)
        y_true[idx2] = 1
        scenario[idx2] = "away_or_night_power"

    # 3. Puerta o ventana abierta de madrugada estando en casa (sin calefacción -> no salta regla)
    idx3 = sample_indices(night_idx, per_scenario)
    if len(idx3):
        df.loc[idx3, "presence"] = 1
        df.loc[idx3, "heating_on"] = 0
        open_door = rng.random(len(idx3)) < 0.5
        df.loc[idx3[open_door], "door_open"] = 1
        df.loc[idx3[~open_door], "window_open"] = 1
        y_true[idx3] = 1
        scenario[idx3] = "night_door_or_window"

    # 4. Calefacción encendida estando fuera de casa o de madrugada (ventanas cerradas)
    pool4 = np.union1d(away_idx, night_idx)
    idx4 = sample_indices(pool4, per_scenario)
    if len(idx4):
        df.loc[idx4, "window_open"] = 0
        df.loc[idx4, "door_open"] = 0
        df.loc[idx4, "heating_on"] = 1
        df.loc[idx4, "power_W"] = np.round(rng.uniform(450.0, 900.0, size=len(idx4)), 2)
        y_true[idx4] = 1
        scenario[idx4] = "heating_unusual"

    # 5. Actividad nocturna anómala combinada (movimiento + potencia + agua a las 01:00–05:00)
    idx5 = sample_indices(night_idx, per_scenario)
    if len(idx5):
        df.loc[idx5, "presence"] = 1
        df.loc[idx5, "motion"] = 1
        df.loc[idx5, "fp2_active"] = 0
        df.loc[idx5, "power_W"] = np.round(rng.uniform(600.0, 1500.0, size=len(idx5)), 2)
        df.loc[idx5, "water_L_per_min"] = np.round(rng.uniform(3.0, 8.0, size=len(idx5)), 3)
        y_true[idx5] = 1
        scenario[idx5] = "night_activity_burst"

    # 6. Presencia estática FP2 diurna inusual + consumo alto de calefactor/electrodoméstico
    idx6 = sample_indices(day_home_idx, per_scenario)
    if len(idx6):
        df.loc[idx6, "presence"] = 1
        df.loc[idx6, "motion"] = 0
        df.loc[idx6, "fp2_active"] = 1
        df.loc[idx6, "power_W"] = np.round(rng.uniform(1500.0, 2500.0, size=len(idx6)), 2)
        y_true[idx6] = 1
        scenario[idx6] = "daytime_fp2_static_power"

    return df, y_true, scenario


def evaluate_pipeline(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    contamination: float = 0.004,
    anomaly_ratio: float = 0.02,
    seed: int = 42,
) -> Dict[str, Any]:
    """Entrena con el split cronológico inicial y evalúa sobre el tramo final con anomalías inyectadas."""
    split_idx = int(len(data) * train_ratio)
    train_df = data.iloc[:split_idx].copy()
    test_df_clean = data.iloc[split_idx:].copy()

    model, meta = fit_isolation_forest(
        train_df,
        contamination=contamination,
        random_state=seed,
    )

    test_df, y_true, scenarios = inject_behavioral_anomalies(
        test_df_clean,
        anomaly_ratio=anomaly_ratio,
        seed=seed,
    )

    x_test = build_feature_matrix(test_df, profile=meta["profile"])
    scores = model.decision_function(x_test)
    threshold = float(meta["threshold"])
    y_pred = (scores < threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, -scores))
    pr_auc = float(average_precision_score(y_true, -scores))

    per_scenario: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(scenarios)):
        if name == "normal":
            continue
        mask = scenarios == name
        detected = int(y_pred[mask].sum())
        total = int(mask.sum())
        per_scenario[name] = {
            "detected": detected,
            "total": total,
            "recall": float(detected / total) if total else 0.0,
        }

    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "contamination": contamination,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "per_scenario": per_scenario,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación reproducible (split cronológico 80/20) del modelo Isolation Forest."
    )
    parser.add_argument("--days", type=int, default=365, help="Días a simular si se usa --generate")
    parser.add_argument("--generate", action="store_true", help="Generar datos en memoria en lugar de leer CSV")
    parser.add_argument("--contamination", type=float, default=0.004, help="Contaminación de Isolation Forest")
    parser.add_argument("--anomaly-ratio", type=float, default=0.02, help="Proporción de anomalías en test")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    if args.generate:
        df = simulate_yearly_data(days=args.days, seed=args.seed)
    else:
        try:
            df = load_data()
        except FileNotFoundError:
            print("CSV no encontrado; generando dataset sintético en memoria...")
            df = simulate_yearly_data(days=args.days, seed=args.seed)

    results = evaluate_pipeline(
        df,
        contamination=args.contamination,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )

    print("\n=== Evaluación Isolation Forest (Split cronológico 80/20 sin fuga) ===")
    print(f"Train (solo normal): {results['train_rows']:,} filas | Test: {results['test_rows']:,} filas")
    print(f"Contamination: {results['contamination']}")
    print(
        f"Matriz de confusión -> TN: {results['tn']:,} | FP: {results['fp']:,} | "
        f"FN: {results['fn']:,} | TP: {results['tp']:,}"
    )
    print(f"Precision : {results['precision'] * 100:.2f} %")
    print(f"Recall    : {results['recall'] * 100:.2f} %")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"FPR       : {results['fpr'] * 100:.3f} %")
    print(f"ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"PR-AUC    : {results['pr_auc']:.4f}")
    print("\nDesglose de Recall por escenario de anomalía conductual:")
    for name, stats in results["per_scenario"].items():
        print(f"  - {name:26s}: {stats['detected']:4d}/{stats['total']:4d} ({stats['recall'] * 100:.1f} %)")


if __name__ == "__main__":
    main()
