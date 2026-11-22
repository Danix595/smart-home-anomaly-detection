from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def simulate_yearly_data(
    days: int = 365,
    seed: int = 42,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Genera telemetría sintética minuto a minuto con rutinas realistas por bloques diarios.

    Alinea siempre el inicio a medianoche (00:00) para que los desplazamientos diarios
    (`d0 + minutos`) coincidan exactamente con la hora del reloj independientemente de la
    hora local o zona horaria de ejecución.
    """
    rng_np = np.random.default_rng(seed)

    # Exactamente days * 1440 minutos (525.600 filas para days=365), empezando a las 00:00
    n = days * 24 * 60
    end_midnight = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else pd.Timestamp.now().normalize()
    )
    start = end_midnight - pd.Timedelta(days=days)
    timestamps = pd.date_range(start=start, periods=n, freq="1min")

    df = pd.DataFrame({"timestamp": timestamps})
    hours = (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0).to_numpy()
    weekdays = df["timestamp"].dt.weekday.to_numpy()
    is_weekend = weekdays >= 5

    # 1. PRESENCIA por bloques continuos realistas (sin teletransportes minuto a minuto)
    presence = np.ones(n, dtype=int)
    door_open = np.zeros(n, dtype=int)
    window_open = np.zeros(n, dtype=int)

    for day_idx in range(days):
        d0 = day_idx * 1440
        d1 = d0 + 1440
        day_is_weekend = bool(is_weekend[d0])

        # Salida del domicilio por bloque continuo
        if not day_is_weekend:
            # ~80 % días de oficina fuera; ~20 % días de teletrabajo en casa
            if rng_np.random() < 0.80:
                leave_min = int(rng_np.integers(8 * 60 + 15, 8 * 60 + 55))
                return_min = int(rng_np.integers(17 * 60 + 25, 18 * 60 + 35))
                presence[d0 + leave_min : d0 + return_min] = 0
                # Pulso breve de puerta al salir y al volver (1–2 min)
                door_open[d0 + leave_min - 1 : d0 + leave_min + 1] = 1
                door_open[d0 + return_min : d0 + return_min + 2] = 1
        else:
            # Fines de semana: salida opcional continua de 2–4 h entre las 11:00 y las 19:00
            if rng_np.random() < 0.55:
                leave_min = int(rng_np.integers(11 * 60, 15 * 60))
                duration = int(rng_np.integers(120, 240))
                return_min = min(1435, leave_min + duration)
                presence[d0 + leave_min : d0 + return_min] = 0
                door_open[d0 + leave_min - 1 : d0 + leave_min + 1] = 1
                door_open[d0 + return_min : d0 + return_min + 2] = 1

        # Hábito de ventilación matinal continua (15–30 min entre 08:00 y 08:50 si está en casa)
        if rng_np.random() < 0.70:
            vent_start = int(rng_np.integers(8 * 60, 8 * 60 + 20))
            vent_dur = int(rng_np.integers(15, 31))
            vent_slice = slice(d0 + vent_start, min(d1, d0 + vent_start + vent_dur))
            # Solo ventila mientras hay presencia en casa
            window_open[vent_slice] = presence[vent_slice]

    # Aseguramos coherencia: la puerta nunca queda abierta cuando la casa está vacía en rutina normal
    door_open = door_open * presence

    # 2. MOVIMIENTO (activo de día, casi nulo durmiendo, 0 si está fuera)
    prob_motion = np.full(n, 0.005)
    active_wd = (~is_weekend) & (hours >= 7.0) & (hours < 23.0)
    active_we = is_weekend & (hours >= 9.0) & (hours < 23.5)
    prob_motion[active_wd] = 0.35
    prob_motion[active_we] = 0.40
    motion = ((rng_np.random(n) < prob_motion) & (presence == 1)).astype(int)

    # 3. PRESENCIA ESTÁTICA FP2 (descanso en cama por la noche o sofá por la tarde sin movimiento)
    sleeping = ((~is_weekend) & ((hours >= 23.0) | (hours < 7.0))) | (
        is_weekend & ((hours >= 23.5) | (hours < 9.0))
    )
    evening_rest = (hours >= 19.5) & (hours < 23.0)
    fp2_active = ((presence == 1) & (motion == 0) & (sleeping | evening_rest)).astype(int)

    # 4. CALEFACCIÓN (encendida en franjas frías si hay presencia y ventanas cerradas)
    heating_schedule = ((hours >= 6.5) & (hours < 8.0)) | ((hours >= 19.0) & (hours < 22.5))
    heating_on = ((presence == 1) & (window_open == 0) & heating_schedule).astype(int)

    # 5. AGUA (ducha/aseo matinal 07:00–08:30 y noche 20:00–21:00; ruido residual mínimo el resto)
    water = rng_np.normal(0.0, 0.05, n)
    mask_morning = (hours >= 7.0) & (hours < 8.5) & (presence == 1)
    mask_night = (hours >= 20.0) & (hours < 21.0) & (presence == 1)
    if mask_morning.any():
        water[mask_morning] += rng_np.gamma(2.0, 1.8, size=int(mask_morning.sum()))
    if mask_night.any():
        water[mask_night] += rng_np.gamma(1.5, 1.8, size=int(mask_night.sum()))
    water = np.clip(water, 0.0, 14.5)

    # 6. POTENCIA ELÉCTRICA (standby 50 W fuera/durmiendo; actividad y luces al anochecer)
    power = rng_np.normal(50.0, 5.0, n)
    mask_awake_home = (presence == 1) & (~sleeping)
    if mask_awake_home.any():
        power[mask_awake_home] += rng_np.normal(70.0, 20.0, size=int(mask_awake_home.sum()))
    mask_evening_peak = (presence == 1) & (hours >= 19.0) & (hours < 23.0)
    if mask_evening_peak.any():
        power[mask_evening_peak] += rng_np.normal(180.0, 45.0, size=int(mask_evening_peak.sum()))
    mask_heating_load = heating_on == 1
    if mask_heating_load.any():
        power[mask_heating_load] += rng_np.normal(120.0, 25.0, size=int(mask_heating_load.sum()))
    power = np.clip(power, 30.0, 3000.0)

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "water_L_per_min": np.round(water, 3),
            "power_W": np.round(power, 2),
            "presence": presence,
            "motion": motion,
            "window_open": window_open,
            "door_open": door_open,
            "heating_on": heating_on,
            "fp2_active": fp2_active,
        }
    )


if __name__ == "__main__":
    print("Generando 1 año de datos (525.600 minutos)...")
    dataset = simulate_yearly_data(days=365)
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "simulated_data_1year.csv"
    dataset.to_csv(output_path, index=False)
    print(f"¡Hecho! Datos guardados en {output_path} ({len(dataset)} filas)")
