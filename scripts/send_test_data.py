# scripts/send_test_data.py
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/detect")
API_KEY = os.getenv("API_KEY", "").strip()
REQUEST_TIMEOUT = 10.0


def make_reading(timestamp_obj: datetime) -> dict:
    """Genera una lectura simulada coherente con las rutinas normales de simulate_yearly.py."""
    h = timestamp_obj.hour + timestamp_obj.minute / 60.0
    is_weekend = timestamp_obj.weekday() >= 5

    if not is_weekend and 8.75 <= h < 17.5:
        presence = 0
    else:
        presence = 1

    sleeping = ((not is_weekend) and (h >= 23.0 or h < 7.0)) or (
        is_weekend and (h >= 23.5 or h < 9.0)
    )
    heating_on = bool(presence == 1 and ((6.5 <= h < 8.0) or (19.0 <= h < 22.5)))

    if presence == 1:
        if (7.0 <= h < 8.5) or (20.0 <= h < 21.0):
            water = max(0.0, random.gauss(3.0, 0.8))
        else:
            water = max(0.0, random.gauss(0.0, 0.03))

        motion = 1 if (not sleeping and random.random() < 0.35) else 0
        fp2_static = (motion == 0) and (sleeping or (19.5 <= h < 23.0))

        power = 50.0 + random.gauss(0, 5.0)
        if not sleeping:
            power += 70.0 + random.gauss(0, 15.0)
        if 19.0 <= h < 23.0:
            power += 180.0 + random.gauss(0, 35.0)
        if heating_on:
            power += 120.0 + random.gauss(0, 20.0)
    else:
        water = 0.0
        power = 50.0 + random.gauss(0, 4.0)
        motion = 0
        fp2_static = False

    power = max(35.0, round(power, 2))

    return {
        "timestamp": timestamp_obj.isoformat(),
        "energy": {
            "global_power_W": power,
            "plugs_W": {"enchufe_tv_aqara": round(power * 0.3, 2), "enchufe_calefactor_bano": 0.0},
            "lights_on": {"luz_pasillo": motion == 1},
        },
        "health": {
            "presence": int(presence),
            "motion": int(motion),
            "fp2_zones": {"zona_sofa": fp2_static and not sleeping, "cama": fp2_static and sleeping, "suelo_bano": False},
        },
        "water": {
            "water_L_per_min": round(water, 3),
        },
        "security": {
            "doors_open": {"puerta_principal": False},
            "windows_open": {"ventana_salon": False, "ventana_bano": False},
        },
        "comfort": {
            "heating_on": {"termostato_tado_salon": heating_on, "termostato_tado_bano": False},
            "target_temp": {"termostato_tado_salon": 21.0},
            "blinds_open": {"persiana_salon": 8.0 <= h < 20.0},
        },
    }


def _post_payload(client: httpx.Client, payload: dict, headers: dict) -> dict:
    resp = client.post(API_URL, json=payload, headers=headers)
    if resp.status_code in (403, 503):
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    if not API_KEY:
        print(
            "Error: API_KEY no definida. Copia .env.example a .env y configura API_KEY antes de ejecutar este script.",
            file=sys.stderr,
        )
        sys.exit(1)

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        try:
            print(f"--- 1. Escenario de EMERGENCIA DETERMINISTA (Capa Reglas) -> {API_URL} ---")
            emergency_payload = {
                "timestamp": datetime.now().isoformat(),
                "energy": {
                    "global_power_W": 3500.0,  # Sobrecarga crítica (>= 3500 W)
                    "plugs_W": {"enchufe_calefactor_bano": 2000.0},
                },
                "health": {
                    "presence": 1,
                    "motion": 0,
                    "fp2_zones": {"suelo_bano": True},
                },
                "water": {
                    "water_L_per_min": 120.0,  # Fuga masiva (> 15 L/min)
                },
                "security": {
                    "doors_open": {"puerta_principal": False},
                    "windows_open": {"ventana_bano": True},
                },
                "comfort": {
                    "heating_on": {"termostato_tado_bano": True},
                },
            }
            print(f"Respuesta API: {_post_payload(client, emergency_payload, headers)}")
            print("-" * 60 + "\n")

            print("--- 2. Escenario de ANOMALÍA CONDUCTUAL (Capa IA - Isolation Forest) ---")
            # Por debajo de los umbrales de rules.py, pero altamente anómalo a las 03:15 AM
            night_ts = datetime.now().replace(hour=3, minute=15, second=0, microsecond=0)
            ai_anomaly_payload = {
                "timestamp": night_ts.isoformat(),
                "energy": {"global_power_W": 1800.0},
                "health": {"presence": 1, "motion": 1, "fp2_zones": {"suelo_bano": False}},
                "water": {"water_L_per_min": 6.5},
                "security": {
                    "doors_open": {"puerta_principal": True},
                    "windows_open": {"ventana_bano": False, "ventana_salon": False},
                },
                "comfort": {
                    "heating_on": {"termostato_tado_bano": False, "termostato_tado_salon": False},
                },
            }
            print(f"Respuesta API: {_post_payload(client, ai_anomaly_payload, headers)}")
            print("-" * 60 + "\n")

            print("--- 3. Simulación de lecturas normales (10 minutos) ---")
            now = datetime.now()
            for i in range(10):
                ts = now + timedelta(minutes=i)
                reading = make_reading(ts)
                body = _post_payload(client, reading, headers)

                status = f"🔴 ANOMALÍA ({body.get('reason')})" if body.get("anomaly") else "🟢 Normal"
                pres = reading["health"]["presence"]
                mov = reading["health"]["motion"]
                agua = reading["water"]["water_L_per_min"]
                luz = reading["energy"]["global_power_W"]
                print(f"[{reading['timestamp']}] Presencia:{pres} Mov:{mov} Agua:{agua} Luz:{luz}W -> {status}")
                time.sleep(0.5)

        except httpx.ConnectError:
            print(f"Error: No se pudo conectar a {API_URL}.", file=sys.stderr)
            print(
                "Asegúrate de que el servidor esté corriendo (uvicorn src.api.main:app --reload)",
                file=sys.stderr,
            )
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            print(f"Error HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
