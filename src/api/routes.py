from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
import asyncio
import time
import logging
from datetime import datetime

from src.api.schemas import Reading
from src.api.dependencies import get_api_key
from src.core.config import COOLDOWN
from src.core.rules import evaluate_business_rules
from src.services.telegram import build_alert_text, send_telegram_message
from src.db.database import get_state, save_state, get_last_alert_time, save_last_alert_time
from src.models.detector import predict_single

logger = logging.getLogger("tfg_domotica")
router = APIRouter()
state_lock = asyncio.Lock()

# Actualiza recursivamente un diccionario anidado, permitiendo actualizar solo partes específicas sin borrar el resto.

def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = deep_update(d.get(k, {}), v)
        elif v is not None:
            d[k] = v
    return d

@router.post("/detect", dependencies=[Depends(get_api_key)])
async def detect(reading: Reading):
    current_timestamp = reading.timestamp if reading.timestamp else datetime.now().isoformat()

    try:
        current_unix_time = datetime.fromisoformat(current_timestamp).timestamp()
    except ValueError:
        current_unix_time = time.time()

    async with state_lock:
        current_state = get_state()

        # Guardamos la presencia anterior para calcular si acaba de llegar a casa
        old_presence = current_state.get("health", {}).get("presence", 0)

        # Convertimos la lectura a diccionario ignorando los valores nulos (lo que no se ha enviado)
        reading_dict = reading.model_dump(exclude_unset=True, exclude_none=True)
        if "timestamp" in reading_dict:
            del reading_dict["timestamp"]

        # Fusionamos el estado actual de la casa con los nuevos datos recibidos
        current_state = deep_update(current_state, reading_dict)

        # Lógica especial de Salud: Actualizamos el last_motion_timestamp
        new_presence = current_state.get("health", {}).get("presence", 0)
        new_motion = current_state.get("health", {}).get("motion", 0)

        # Si hay movimiento, o si acaba de llegar a casa (presencia pasa de 0 a 1)
        if new_motion == 1 or (old_presence == 0 and new_presence == 1):
            current_state["health"]["last_motion_timestamp"] = current_unix_time

        save_state(current_state)

    try:
        # Extraemos los valores críticos para el ML
        water_val = current_state.get("water", {}).get("water_L_per_min", 0.0)
        power_val = current_state.get("energy", {}).get("global_power_W", 0.0)
        presence_val = current_state.get("health", {}).get("presence", 0)
        motion_val = current_state.get("health", {}).get("motion", 0)

        # Calcula si hay alguna ventana abierta = 1, si no = 0)
        windows_open_dict = current_state.get("security", {}).get("windows_open", {})
        window_open_val = 1 if any(windows_open_dict.values()) else 0

        # Evaluamos reglas de negocio deterministas (fugas, ventanas abiertas sin nadie...)
        reason = evaluate_business_rules(current_state, current_timestamp)

        # Guardamos el resultado
        is_anom = (reason != "normal")
        score = 0.0

        # Si no hay alerta directa por reglas, pasamos los datos a la Inteligencia Artificial
        if not is_anom:
            res = await run_in_threadpool(
                predict_single, current_timestamp, water_val, power_val, presence_val, motion_val, window_open_val
            )
            is_anom = (res["prediction"] == -1)
            reason = res.get("reason", "unknown") if is_anom else "normal"
            score = res.get("score", 0.0)

        # Gestor de Notificaciones a Telegram
        if is_anom:
            now = time.time()
            if now - get_last_alert_time(reason) >= COOLDOWN:
                if await send_telegram_message(build_alert_text(reason, current_timestamp, current_state, score)):
                    save_last_alert_time(reason, now)
                    logger.info(f"Alerta enviada: {reason}")

        return {
            "status": "processed",
            "anomaly": is_anom,
            "reason": reason,
            "current_state": current_state,
            "ai_score": score
        }

    except Exception as e:
        logger.exception(f"Error procesando lectura: {e}")
        raise HTTPException(status_code=500, detail="Error interno procesando la lectura.")


@router.get("/")
def root():
    return {"status": "online"}


@router.get("/state", dependencies=[Depends(get_api_key)])
def read_state():
    return {"status": "online", "state": get_state()}
