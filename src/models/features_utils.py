from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

CONT_COLUMNS: List[str] = ["water_L_per_min", "power_W"]
BIN_COLUMNS: List[str] = ["motion", "window_open", "door_open", "heating_on", "fp2_active"]

SENSOR_COLUMNS: List[str] = [
    "water_L_per_min",
    "power_W",
    "presence",
    "motion",
    "window_open",
    "door_open",
    "heating_on",
    "fp2_active",
]

TIME_COLUMNS: List[str] = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

FEATURE_COLUMNS: List[str] = SENSOR_COLUMNS + TIME_COLUMNS

MODEL_FEATURE_COLUMNS: List[str] = [
    "water_excess",
    "power_excess",
    "motion_surprise",
    "window_surprise",
    "door_surprise",
    "heating_surprise",
    "fp2_surprise",
    "presence_surprise",
    "max_signal_surprise",
    "sum_signal_surprise",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

_CONT_SCALE_FLOORS: Dict[str, float] = {
    "water_L_per_min": 0.25,
    "power_W": 30.0,
}
_SURPRISE_EPS = 1e-4


def _parse_datetime(timestamp: str) -> datetime:
    if isinstance(timestamp, datetime):
        return timestamp
    return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))


def make_time_features(timestamp: str) -> Dict[str, float]:
    """Codifica hora del día y día de la semana como variables cíclicas (seno/coseno)."""
    ts = _parse_datetime(timestamp)
    hour = ts.hour + ts.minute / 60.0
    weekday = float(ts.weekday())
    return {
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "weekday_sin": float(np.sin(2.0 * np.pi * weekday / 7.0)),
        "weekday_cos": float(np.cos(2.0 * np.pi * weekday / 7.0)),
    }


def extract_sensor_features_from_state(state: Dict[str, Any]) -> Dict[str, float]:
    """Extrae las variables de sensores a partir del estado anidado de la vivienda."""
    security = state.get("security", {})
    comfort = state.get("comfort", {})
    health = state.get("health", {})

    return {
        "water_L_per_min": float(state.get("water", {}).get("water_L_per_min", 0.0)),
        "power_W": float(state.get("energy", {}).get("global_power_W", 0.0)),
        "presence": float(health.get("presence", 0)),
        "motion": float(health.get("motion", 0)),
        "window_open": 1.0 if any(security.get("windows_open", {}).values()) else 0.0,
        "door_open": 1.0 if any(security.get("doors_open", {}).values()) else 0.0,
        "heating_on": 1.0 if any(comfort.get("heating_on", {}).values()) else 0.0,
        "fp2_active": 1.0 if any(health.get("fp2_zones", {}).values()) else 0.0,
    }


def fit_routine_profile(data: pd.DataFrame) -> Dict[str, Any]:
    """Aprende de forma no supervisada el perfil de rutinas por (hora, fin_de_semana, presencia).

    - Sensores continuos (agua, potencia): envolvente robusta [P01, P99] y escala (P99 - P50).
    - Sensores binarios (movimiento, ventana, puerta, calefacción, FP2): probabilidad empírica.
    - Presencia: probabilidad empírica por (hora, fin_de_semana).
    """
    missing = [col for col in SENSOR_COLUMNS + ["timestamp"] if col not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para ajustar el perfil: {missing}")

    tmp = data.copy()
    tmp["_h"] = tmp["timestamp"].dt.hour.astype(int)
    tmp["_we"] = (tmp["timestamp"].dt.weekday >= 5).astype(int)

    cont_p01 = np.zeros((24, 2, 2, len(CONT_COLUMNS)), dtype=float)
    cont_p99 = np.zeros((24, 2, 2, len(CONT_COLUMNS)), dtype=float)
    cont_scale = np.ones((24, 2, 2, len(CONT_COLUMNS)), dtype=float)
    bin_prob = np.full((24, 2, 2, len(BIN_COLUMNS)), 0.5, dtype=float)
    bucket_seen = np.zeros((24, 2, 2), dtype=bool)

    g_p01 = [float(np.percentile(tmp[c], 1)) for c in CONT_COLUMNS]
    g_p99 = [float(np.percentile(tmp[c], 99)) for c in CONT_COLUMNS]
    g_scale = [
        max(g_p99[i] - g_p01[i], _CONT_SCALE_FLOORS[c])
        for i, c in enumerate(CONT_COLUMNS)
    ]
    g_bin = [float(tmp[c].mean()) for c in BIN_COLUMNS]

    for (hh, ww, pp), grp in tmp.groupby(["_h", "_we", "presence"]):
        h_idx, w_idx, p_idx = int(hh), int(ww), int(pp)
        bucket_seen[h_idx, w_idx, p_idx] = True
        for i, col in enumerate(CONT_COLUMNS):
            vals = grp[col].to_numpy(dtype=float)
            p01 = float(np.percentile(vals, 1))
            p50 = float(np.percentile(vals, 50))
            p99 = float(np.percentile(vals, 99))
            cont_p01[h_idx, w_idx, p_idx, i] = p01
            cont_p99[h_idx, w_idx, p_idx, i] = p99
            cont_scale[h_idx, w_idx, p_idx, i] = max(p99 - p50, _CONT_SCALE_FLOORS[col])
        for j, col in enumerate(BIN_COLUMNS):
            bin_prob[h_idx, w_idx, p_idx, j] = float(grp[col].mean())

    for hh in range(24):
        for ww in range(2):
            for pp in range(2):
                if not bucket_seen[hh, ww, pp]:
                    cont_p01[hh, ww, pp] = g_p01
                    cont_p99[hh, ww, pp] = g_p99
                    cont_scale[hh, ww, pp] = g_scale
                    bin_prob[hh, ww, pp] = g_bin

    pres_prob = np.full((24, 2), float(tmp["presence"].mean()), dtype=float)
    for (hh, ww), grp in tmp.groupby(["_h", "_we"]):
        pres_prob[int(hh), int(ww)] = float(grp["presence"].mean())

    return {
        "cont_p01": cont_p01,
        "cont_p99": cont_p99,
        "cont_scale": cont_scale,
        "bin_prob": bin_prob,
        "pres_prob": pres_prob,
    }


def add_time_features_df(data: pd.DataFrame) -> pd.DataFrame:
    """Añade las variables temporales cíclicas (hora y día de la semana) a un DataFrame."""
    df = data.copy()
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    weekday = df["timestamp"].dt.weekday.astype(float)
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    df["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    return df


def build_feature_matrix(
    data: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Construye la matriz numérica para Isolation Forest (contextual + cíclica si se aporta perfil)."""
    df_t = add_time_features_df(data)
    missing = [col for col in FEATURE_COLUMNS if col not in df_t.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    if profile is None:
        return df_t[FEATURE_COLUMNS].to_numpy(dtype=float)

    h = df_t["timestamp"].dt.hour.to_numpy(dtype=int)
    we = (df_t["timestamp"].dt.weekday >= 5).astype(int).to_numpy()
    pres = np.clip(df_t["presence"].to_numpy(dtype=int), 0, 1)

    c_vals = df_t[CONT_COLUMNS].to_numpy(dtype=float)
    above = np.maximum(0.0, c_vals - profile["cont_p99"][h, we, pres]) / profile["cont_scale"][h, we, pres]
    below = np.maximum(0.0, profile["cont_p01"][h, we, pres] - c_vals) / profile["cont_scale"][h, we, pres]
    cont_exc = np.log1p(above + below)

    b_vals = df_t[BIN_COLUMNS].to_numpy(dtype=float)
    p1 = profile["bin_prob"][h, we, pres]
    p_obs = np.where(b_vals > 0.5, p1, 1.0 - p1)
    bin_surp = -np.log(np.clip(p_obs, 0.0, 1.0) + _SURPRISE_EPS)

    pp1 = profile["pres_prob"][h, we]
    p_pres = np.where(pres == 1, pp1, 1.0 - pp1)
    pres_surp = (-np.log(np.clip(p_pres, 0.0, 1.0) + _SURPRISE_EPS)).reshape(-1, 1)

    signals = np.hstack([cont_exc, bin_surp, pres_surp])
    max_sig = np.max(signals, axis=1, keepdims=True)
    sum_sig = np.sum(signals, axis=1, keepdims=True)
    time_cyc = df_t[TIME_COLUMNS].to_numpy(dtype=float)

    return np.hstack([signals, max_sig, sum_sig, time_cyc])


def make_features_from_row(
    timestamp: str,
    water: float,
    power: float,
    presence: int,
    motion: int,
    window_open: int,
    door_open: int = 0,
    heating_on: int = 0,
    fp2_active: int = 0,
    profile: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Construye el vector (1, N) en O(1) para inferencia en tiempo real sin overhead de Pandas."""
    ts = _parse_datetime(timestamp)
    time_features = make_time_features(timestamp)

    if profile is None:
        return np.array(
            [
                [
                    float(water),
                    float(power),
                    float(presence),
                    float(motion),
                    float(window_open),
                    float(door_open),
                    float(heating_on),
                    float(fp2_active),
                    time_features["hour_sin"],
                    time_features["hour_cos"],
                    time_features["weekday_sin"],
                    time_features["weekday_cos"],
                ]
            ],
            dtype=float,
        )

    h = ts.hour
    we = 1 if ts.weekday() >= 5 else 0
    pres = 1 if int(presence) > 0 else 0

    c_vals = np.array([float(water), float(power)], dtype=float)
    above = np.maximum(0.0, c_vals - profile["cont_p99"][h, we, pres]) / profile["cont_scale"][h, we, pres]
    below = np.maximum(0.0, profile["cont_p01"][h, we, pres] - c_vals) / profile["cont_scale"][h, we, pres]
    cont_exc = np.log1p(above + below)

    b_vals = np.array(
        [float(motion), float(window_open), float(door_open), float(heating_on), float(fp2_active)],
        dtype=float,
    )
    p1 = profile["bin_prob"][h, we, pres]
    p_obs = np.where(b_vals > 0.5, p1, 1.0 - p1)
    bin_surp = -np.log(np.clip(p_obs, 0.0, 1.0) + _SURPRISE_EPS)

    pp1 = float(profile["pres_prob"][h, we])
    p_pres = pp1 if pres == 1 else (1.0 - pp1)
    pres_surp = float(-np.log(min(max(p_pres, 0.0), 1.0) + _SURPRISE_EPS))

    signals = np.concatenate([cont_exc, bin_surp, np.array([pres_surp], dtype=float)])
    max_sig = float(np.max(signals))
    sum_sig = float(np.sum(signals))

    return np.array(
        [
            [
                *signals.tolist(),
                max_sig,
                sum_sig,
                time_features["hour_sin"],
                time_features["hour_cos"],
                time_features["weekday_sin"],
                time_features["weekday_cos"],
            ]
        ],
        dtype=float,
    )


def make_features_from_state(
    timestamp: str,
    state: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Construye el vector (1, N) directamente desde un diccionario de estado completo."""
    sensors = extract_sensor_features_from_state(state)
    return make_features_from_row(
        timestamp=timestamp,
        water=sensors["water_L_per_min"],
        power=sensors["power_W"],
        presence=int(sensors["presence"]),
        motion=int(sensors["motion"]),
        window_open=int(sensors["window_open"]),
        door_open=int(sensors["door_open"]),
        heating_on=int(sensors["heating_on"]),
        fp2_active=int(sensors["fp2_active"]),
        profile=profile,
    )
