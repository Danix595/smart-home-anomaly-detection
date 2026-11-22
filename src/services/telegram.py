from datetime import datetime
import html
import logging
from typing import List, Optional

import httpx

from src.core import config

logger = logging.getLogger("tfg_domotica")

REASON_TRANSLATIONS = {
    "water_critical_high": "🌊 Consumo de agua excesivo (Posible fuga)",
    "power_high": "⚡ Consumo Eléctrico Crítico",
    "presence_no_motion": "⚠️ Alerta: Persona inmóvil en casa",
    "security_element_open_away": "🚪 Casa vacía con puerta/ventana abierta",
    "efficiency_heating_window_open": "💸 Derroche: Calefacción y ventana abiertas",
    "ml_score_threshold": "🤖 Rutina inusual detectada (IA)",
    "ml_default_predict": "🤖 Anomalía de comportamiento general",
    "unknown": "❓ Anomalía desconocida",
}


def format_timestamp_friendly(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y - %H:%M:%S")
    except (ValueError, TypeError, AttributeError):
        return str(iso_str)


def _translate_reason(reason_code: str) -> str:
    return REASON_TRANSLATIONS.get(reason_code, f"Anomalía: {html.escape(str(reason_code))}")


def build_alert_text(
    reason_code: str,
    timestamp_str: str,
    state: dict,
    score: Optional[float],
    extra_reasons: Optional[List[str]] = None,
) -> str:
    motivo_texto = _translate_reason(reason_code)
    hora_bonita = html.escape(format_timestamp_friendly(timestamp_str))

    text = "⚠️ <b>ALERTA DOMÓTICA</b> ⚠️\n\n"
    text += f"<b>Motivo:</b> {motivo_texto}\n"

    if extra_reasons:
        otros = [_translate_reason(r) for r in extra_reasons if r != reason_code]
        if otros:
            text += "<b>Alertas simultáneas:</b>\n"
            for item in otros:
                text += f"  • {item}\n"

    text += f"<b>Hora:</b> {hora_bonita}\n\n"

    # 1. Extraer datos principales
    water = state.get("water", {}).get("water_L_per_min", 0.0)
    power = state.get("energy", {}).get("global_power_W", 0.0)
    presence = state.get("health", {}).get("presence", 0)
    motion = state.get("health", {}).get("motion", 0)

    # 2. Resumen según el tipo de alerta
    all_codes = [reason_code] + (extra_reasons or [])
    if len(all_codes) > 1 or not any(
        k in reason_code for k in ("water", "power", "presence", "motion")
    ):
        text += f"💧 <b>Agua:</b> {water} L/min | ⚡ <b>Luz:</b> {power} W\n"
        text += f"👤 <b>Estado:</b> {'🏠 En casa' if presence == 1 else '❌ Fuera'} | 🏃 <b>Mov:</b> {'Sí' if motion == 1 else 'No'}\n"
    elif "water" in reason_code:
        text += f"💧 <b>Agua:</b> {water} L/min\n"
    elif "power" in reason_code:
        text += f"⚡ <b>Luz (Global):</b> {power} W\n"
    elif "presence" in reason_code or "motion" in reason_code:
        text += f"👤 <b>Estado:</b> {'🏠 En casa' if presence == 1 else '❌ Fuera'}\n"
        text += f"🏃 <b>Movimiento:</b> {'Sí' if motion == 1 else 'No'}\n"

    # 3. Contexto (escapado para HTML seguro en Telegram)
    text += "\n<b>📋 Contexto de la casa:</b>\n"

    security = state.get("security", {})
    open_doors = [html.escape(str(k)) for k, v in security.get("doors_open", {}).items() if v]
    open_windows = [html.escape(str(k)) for k, v in security.get("windows_open", {}).items() if v]

    if open_doors:
        text += f"🚪 <b>Puertas abiertas:</b> {', '.join(open_doors)}\n"
    if open_windows:
        text += f"🪟 <b>Ventanas abiertas:</b> {', '.join(open_windows)}\n"

    comfort = state.get("comfort", {})
    heating_on = [html.escape(str(k)) for k, v in comfort.get("heating_on", {}).items() if v]
    if heating_on:
        text += f"🔥 <b>Calefacción ON en:</b> {', '.join(heating_on)}\n"

    fp2_zones = state.get("health", {}).get("fp2_zones", {})
    active_zones = [html.escape(str(k)) for k, v in fp2_zones.items() if v]
    if active_zones:
        text += f"🛋️ <b>Presencia estática (FP2):</b> {', '.join(active_zones)}\n"

    # 4. Nivel de gravedad IA
    if reason_code.startswith("ml_") and score is not None:
        # Normalizamos el score crudo de scikit-learn (rango típico hasta -0,22) a [0, 1]
        max_anomaly_score = 0.22
        gravedad = min(1.0, max(0.0, abs(min(score, 0.0)) / max_anomaly_score))

        etiqueta = "🔴 CRÍTICO" if gravedad > 0.75 else "🟠 ALTO" if gravedad > 0.60 else "🟡 MODERADO"
        text += f"\n<b>Nivel de Anomalía IA:</b> {gravedad:.2f} ({etiqueta})"
    return text


async def send_telegram_message(text: str) -> bool:
    bot_token = config.BOT_TOKEN
    chat_id = config.CHAT_ID

    if not bot_token or not chat_id:
        logger.warning("Telegram Bot Token o Chat ID no configurados. Imprimiendo alerta en consola:")
        print(text)
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok", True):
                logger.error(f"Telegram rechazó el mensaje: {payload}")
                logger.warning("Alerta no entregada (fallback local):\n%s", text)
                return False
        return True
    except httpx.HTTPStatusError as exc:
        logger.error(f"Error HTTP de Telegram ({exc.response.status_code}): {exc.response.text}")
        logger.warning("Alerta no entregada (fallback local):\n%s", text)
        return False
    except Exception as exc:
        logger.error(f"Error enviando mensaje a Telegram: {exc}")
        logger.warning("Alerta no entregada (fallback local):\n%s", text)
        return False
