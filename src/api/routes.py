import asyncio
from datetime import datetime
import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import get_api_key
from src.api.schemas import Reading
from src.core import config
from src.core.rules import evaluate_all_business_rules
from src.db.database import (
    get_last_alert_time,
    get_state,
    save_last_alert_time,
    save_state,
)
from src.models.detector import predict_from_state
from src.services.telegram import build_alert_text, send_telegram_message

logger = logging.getLogger("tfg_domotica")
router = APIRouter()
state_lock = asyncio.Lock()
cooldown_lock = asyncio.Lock()


def deep_update(d: dict, u: dict) -> dict:
    """Actualiza recursivamente un diccionario anidado sin borrar claves ausentes ni sobreescribir con {}."""
    for k, v in u.items():
        if isinstance(v, dict):
            if not v:
                continue
            existing = d.get(k)
            d[k] = deep_update(existing if isinstance(existing, dict) else {}, v)
        elif v is not None:
            d[k] = v
    return d


def _parse_unix_time(timestamp_str: str) -> float:
    try:
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError):
        return time.time()


@router.post("/detect", dependencies=[Depends(get_api_key)])
async def detect(reading: Reading):
    current_timestamp = reading.timestamp if reading.timestamp else datetime.now().isoformat()
    current_unix_time = _parse_unix_time(current_timestamp)

    async with state_lock:
        current_state = await run_in_threadpool(get_state)

        # Guardamos la presencia anterior para detectar si acaba de llegar a casa
        old_presence = current_state.get("health", {}).get("presence", 0)

        # Convertimos la lectura a diccionario ignorando campos no enviados o nulos
        reading_dict = reading.model_dump(exclude_unset=True, exclude_none=True)
        reading_dict.pop("timestamp", None)

        # Fusionamos el estado actual de la vivienda con los nuevos datos recibidos
        current_state = deep_update(current_state, reading_dict)

        # Lógica de Salud: solo actualizamos last_motion_timestamp si esta lectura reporta
        # explícitamente movimiento (motion == 1) o una transición de llegada a casa (0 -> 1).
        incoming_health = reading_dict.get("health", {})
        incoming_motion = incoming_health.get("motion") if isinstance(incoming_health, dict) else None
        new_presence = current_state.get("health", {}).get("presence", 0)

        if incoming_motion == 1 or (old_presence == 0 and new_presence == 1):
            current_state["health"]["last_motion_timestamp"] = current_unix_time

        await run_in_threadpool(save_state, current_state)

    try:
        # 1. Capa determinista: evaluamos todas las reglas de negocio
        triggered_rules: List[str] = evaluate_all_business_rules(current_state, current_timestamp)
        is_anom = bool(triggered_rules)
        reason = triggered_rules[0] if triggered_rules else "normal"
        reasons = list(triggered_rules)
        score = 0.0

        # 2. Capa predictiva (IA): si las reglas deterministas están en orden, consultamos Isolation Forest
        if not is_anom:
            res = await run_in_threadpool(predict_from_state, current_timestamp, current_state)
            is_anom = res["prediction"] == -1
            reason = res.get("reason", "unknown") if is_anom else "normal"
            reasons = [reason] if is_anom else []
            score = res.get("score", 0.0)

        # 3. Gestor de notificaciones a Telegram protegido contra condiciones de carrera
        if is_anom:
            async with cooldown_lock:
                now = time.time()
                pending_reasons: List[str] = []
                for alert_key in reasons:
                    last_sent = await run_in_threadpool(get_last_alert_time, alert_key)
                    if now - last_sent >= config.COOLDOWN:
                        pending_reasons.append(alert_key)

                if pending_reasons:
                    primary = pending_reasons[0]
                    alert_text = build_alert_text(
                        primary,
                        current_timestamp,
                        current_state,
                        score,
                        extra_reasons=pending_reasons[1:],
                    )
                    if await send_telegram_message(alert_text):
                        for alert_key in pending_reasons:
                            await run_in_threadpool(save_last_alert_time, alert_key, now)
                        logger.info(f"Alerta enviada: {', '.join(pending_reasons)}")

        return {
            "status": "processed",
            "anomaly": is_anom,
            "reason": reason,
            "reasons": reasons,
            "current_state": current_state,
            "ai_score": score,
        }

    except Exception as exc:
        logger.exception(f"Error procesando lectura: {exc}")
        raise HTTPException(status_code=500, detail="Error interno procesando la lectura.")


@router.get("/")
async def root():
    return {"status": "online"}


@router.get("/state", dependencies=[Depends(get_api_key)])
async def read_state():
    state = await run_in_threadpool(get_state)
    return {"status": "online", "state": state}
