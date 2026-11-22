# scripts/send_test_data.py
import requests
from datetime import datetime, timedelta
import random
import time
import sys

API_URL = "http://127.0.0.1:8000/detect"
API_KEY = ""  # Set via API_KEY in .env

# Genera una lectura simulada
def make_reading(timestamp_obj):
    h = timestamp_obj.hour

    # 1. Lógica Base (Presencia, Agua, Luz, Movimiento)
    if 9 <= h < 18:
        presence = 1 if random.random() < 0.6 else 0
    else:
        presence = 1

    if presence == 1:
        base_water = 4.0 if (7 <= h < 9) or (19 <= h < 21) else 0.2
        water = max(0.0, random.gauss(base_water, 0.5))

        power = 300.0 if 18 <= h < 23 else 80.0
        power += random.gauss(0, 30)

        prob_motion = 0.25 if (8 <= h < 23) else 0.02
        motion = 1 if random.random() < prob_motion else 0
    else:
        water = 0.0
        power = 50.0  # Consumo fantasma (nevera, router, tele en standby...)
        motion = 0

    power = max(50.0, round(power, 2))

    # 2. Construcción del JSON Anidado
    return {
        "timestamp": timestamp_obj.isoformat(),
        "energy": {
            "global_power_W": round(power, 2),
            "plugs_W": {"enchufe_tv_aqara": round(power * 0.3, 2), "enchufe_calefactor_bano": 0.0},
            "lights_on": {"luz_pasillo": motion == 1}
        },
        "health": {
            "presence": int(presence),
            "motion": int(motion),
            "fp2_zones": {"zona_sofa": presence == 1 and motion == 0, "suelo_bano": False}
        },
        "water": {
            "water_L_per_min": round(water, 3)
        },
        "security": {
            "doors_open": {"puerta_principal": False},
            "windows_open": {"ventana_salon": False, "ventana_bano": False}
        },
        "comfort": {
            "heating_on": {"termostato_tado_salon": presence == 1 and h < 8, "termostato_tado_bano": False},
            "target_temp": {"termostato_tado_salon": 21.0},
            "blinds_open": {"persiana_salon": 8 <= h < 20}
        }
    }

if __name__ == "__main__":
    print(f"--- 1. Enviando prueba de ANOMALÍA COMPLEJA a {API_URL} ---")

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    # Caso Extremo: Posible caída/accidente en el baño + fuga
    anomaly_payload = {
        "timestamp": datetime.now().isoformat(),
        "energy": {
            "global_power_W": 3500.0,  # Consumo altísimo (posible calefactor)
            "plugs_W": {"enchufe_calefactor_bano": 2000.0}
        },
        "health": {
            "presence": 1,
            "motion": 0,
            "fp2_zones": {"suelo_bano": True}  # FP2 detecta bulto estático
        },
        "water": {
            "water_L_per_min": 120.0  # Fuga masiva
        },
        "security": {
            "windows_open": {"ventana_bano": True}  # Y encima se está escapando el calor
        },
        "comfort": {
            "heating_on": {"termostato_tado_bano": True}
        }
    }

    try:
        resp = requests.post(API_URL, json=anomaly_payload, headers=headers)
        if resp.status_code == 403:
            print("Error 403: Acceso denegado. Comprueba tu API_KEY.")
            sys.exit(1)

        print(f"Respuesta API: {resp.json()}")
        print("-" * 50 + "\n")

    except requests.exceptions.ConnectionError:
        print(f"Error: No se pudo conectar a {API_URL}.", file=sys.stderr)
        print("Asegúrate de que el servidor esté corriendo (uvicorn src.api.main:app --reload)", file=sys.stderr)
        sys.exit(1)

    print("--- 2. Iniciando simulación de lecturas normales (bucle) ---")

    now = datetime.now()

    for i in range(10):
        ts = now + timedelta(minutes=i)
        reading = make_reading(ts)

        try:
            resp = requests.post(API_URL, json=reading, headers=headers)
            if resp.status_code == 403:
                print("Error 403: Acceso denegado en el bucle.")
                break

            status = "🔴 ANOMALÍA" if resp.json().get("anomaly") else "🟢 Normal"

            # Extraemos los datos para el print
            pres = reading['health']['presence']
            mov = reading['health']['motion']
            agua = reading['water']['water_L_per_min']
            luz = reading['energy']['global_power_W']

            print(f"[{reading['timestamp']}] Presencia:{pres} Mov:{mov} Agua:{agua} Luz:{luz}W -> {status}")

        except requests.exceptions.ConnectionError:
            print("Error de conexión.")
            break

        time.sleep(1)