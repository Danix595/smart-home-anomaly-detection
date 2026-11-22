from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def simulate_yearly_data(days=365, seed=42):
    np.random.seed(seed)
    end = pd.Timestamp.now().floor("min")
    start = end - pd.Timedelta(days=days)
    rng = pd.date_range(start=start, end=end, freq="1min")
    datos_df = pd.DataFrame({"timestamp": rng})
    n = len(datos_df)

    datos_df['hour'] = datos_df['timestamp'].dt.hour + datos_df['timestamp'].dt.minute / 60.0
    datos_df['weekday'] = datos_df['timestamp'].dt.weekday
    datos_df['is_weekend'] = datos_df['weekday'] >= 5

    presence = np.ones(n, dtype=int)
    prob_out = np.zeros(n)
    mask_work = (~datos_df['is_weekend']) & (datos_df['hour'] > 8.5) & (datos_df['hour'] < 18.0)
    prob_out[mask_work] = 0.8
    mask_weekend_out = (datos_df['is_weekend']) & (datos_df['hour'] > 11.0) & (datos_df['hour'] < 15.0)
    prob_out[mask_weekend_out] = 0.4
    rand_vals = np.random.rand(n)
    is_out = (pd.Series(prob_out).rolling(window=60, min_periods=1).mean() > rand_vals)
    presence[is_out.to_numpy()] = 0

    prob_motion = np.zeros(n)
    active_mask_wd = (~datos_df['is_weekend']) & (datos_df['hour'] >= 7) & (datos_df['hour'] < 23)
    active_mask_we = (datos_df['is_weekend']) & (datos_df['hour'] >= 9) & (datos_df['hour'] < 24)
    prob_motion[active_mask_wd] = 0.35
    prob_motion[active_mask_we] = 0.40
    prob_motion[~(active_mask_wd | active_mask_we)] = 0.01
    motion = (np.random.rand(n) < prob_motion).astype(int)
    motion = motion * presence

    water = np.random.normal(0, 0.1, n)
    mask_morning = (datos_df['hour'] >= 7) & (datos_df['hour'] < 8.5)
    mask_night = (datos_df['hour'] >= 20) & (datos_df['hour'] < 21)
    water[mask_morning & (presence == 1)] += np.random.gamma(2, 2, size=mask_morning.sum()) if mask_morning.sum() > 0 else 0
    water[mask_night & (presence == 1)] += np.random.gamma(1.5, 2, size=mask_night.sum()) if mask_night.sum() > 0 else 0
    water = np.clip(water, 0, None)

    power = np.random.normal(50, 5, n)
    mask_lights = (presence == 1) & ((datos_df['hour'] > 19) | (datos_df['hour'] < 7) | (motion == 1))
    power[mask_lights] += np.random.normal(200, 50, size=mask_lights.sum())
    power = np.clip(power, 0, None)

    window_open = np.zeros(n, dtype=int)
    # Hábito: Ventilar de 8 a 9 AM si está en casa
    mask_vent_morning = (presence == 1) & (datos_df['hour'] >= 8.0) & (datos_df['hour'] < 9.0)
    window_open[mask_vent_morning] = (np.random.rand(mask_vent_morning.sum()) < 0.7).astype(int)
    # Suavizar para que no se abra y cierre por minuto
    window_open = (pd.Series(window_open).rolling(window=15, min_periods=1).max() > 0).astype(int).to_numpy()

    df_out = pd.DataFrame({
        "timestamp": datos_df["timestamp"],
        "water_L_per_min": np.round(water, 3),
        "power_W": np.round(power, 2),
        "presence": presence,
        "motion": motion,
        "window_open": window_open
    })
    return df_out

if __name__ == "__main__":
    print("Generando 1 año de datos...")
    df = simulate_yearly_data(days=365)
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "simulated_data_1year.csv"
    df.to_csv(output_path, index=False)
    print(f"¡Hecho! Datos guardados en {output_path} ({len(df)} filas)")