import math
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.models import detector
from src.models.evaluate import evaluate_pipeline
from src.models.features_utils import (
    MODEL_FEATURE_COLUMNS,
    SENSOR_COLUMNS,
    build_feature_matrix,
    fit_routine_profile,
    make_features_from_row,
    make_time_features,
)
from src.simulate_yearly import simulate_yearly_data


def test_simulator_produces_exact_rows_and_realistic_routines():
    days = 7
    df = simulate_yearly_data(days=days, seed=42, end_date="2026-07-02")
    assert len(df) == days * 1440
    # Siempre alineado a medianoche 00:00
    assert df["timestamp"].iloc[0].hour == 0 and df["timestamp"].iloc[0].minute == 0
    for col in ["timestamp"] + SENSOR_COLUMNS:
        assert col in df.columns
    assert not df.isna().any().any()
    assert (df["water_L_per_min"] >= 0).all()
    assert (df["power_W"] >= 0).all()

    # Presencia por bloques continuos (no cientos de cambios diarios)
    transitions_per_day = float((df["presence"].diff().abs() == 1).sum()) / days
    assert transitions_per_day < 5.0

    # Consumo nocturno en standby (~50 W, no 250 W)
    night_mean = float(df[(df["timestamp"].dt.hour >= 1) & (df["timestamp"].dt.hour < 6)]["power_W"].mean())
    assert 35.0 <= night_mean <= 70.0


def test_cyclic_time_features_preserve_boundary_continuity():
    t_late = make_time_features("2026-07-05T23:59:00")  # Domingo 23:59
    t_early = make_time_features("2026-07-06T00:01:00")  # Lunes 00:01

    hour_dist = math.hypot(t_late["hour_sin"] - t_early["hour_sin"], t_late["hour_cos"] - t_early["hour_cos"])
    assert hour_dist < 0.02

    # Distancia entre domingo (6) y lunes (0) igual que entre lunes (0) y martes (1)
    t_tue = make_time_features("2026-07-07T00:01:00")
    d_sun_mon = math.hypot(
        t_late["weekday_sin"] - t_early["weekday_sin"],
        t_late["weekday_cos"] - t_early["weekday_cos"],
    )
    d_mon_tue = math.hypot(
        t_early["weekday_sin"] - t_tue["weekday_sin"],
        t_early["weekday_cos"] - t_tue["weekday_cos"],
    )
    assert math.isclose(d_sun_mon, d_mon_tue, rel_tol=1e-9)


def test_batch_and_single_row_features_match_exactly():
    df = simulate_yearly_data(days=5, seed=42)
    profile = fit_routine_profile(df)
    x_batch = build_feature_matrix(df.iloc[:50], profile=profile)
    assert x_batch.shape == (50, len(MODEL_FEATURE_COLUMNS))

    for i in range(50):
        row = df.iloc[i]
        x_single = make_features_from_row(
            timestamp=row["timestamp"].isoformat(),
            water=float(row["water_L_per_min"]),
            power=float(row["power_W"]),
            presence=int(row["presence"]),
            motion=int(row["motion"]),
            window_open=int(row["window_open"]),
            door_open=int(row["door_open"]),
            heating_on=int(row["heating_on"]),
            fp2_active=int(row["fp2_active"]),
            profile=profile,
        )
        assert np.allclose(x_batch[i], x_single[0], atol=1e-12)


def test_detector_raises_on_missing_or_outdated_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        detector.get_model_cached(model_dir=tmp_path)

    # Artefacto antiguo sin perfil contextual
    joblib.dump({"dummy": True}, tmp_path / "iforest.joblib")
    joblib.dump({"columns": ["old_col"]}, tmp_path / "meta.joblib")
    with pytest.raises(ValueError, match="desactualizados"):
        detector.get_model_cached(model_dir=tmp_path)


def test_evaluate_pipeline_smoke_test():
    df = simulate_yearly_data(days=30, seed=42, end_date="2026-07-02")
    res = evaluate_pipeline(df, contamination=0.004, anomaly_ratio=0.02, seed=42)
    assert res["roc_auc"] > 0.95
    assert res["f1"] > 0.75
    assert len(res["per_scenario"]) == 6
