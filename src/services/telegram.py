import httpx
import logging
from datetime import datetime
from src.core.config import BOT_TOKEN, CHAT_ID

logger = logging.getLogger("tfg_domotica")

REASON_TRANSLATIONS = {
    "water_critical_high": "🌊 Consumo de agua excesivo (Posible fuga)",
    "power_high": "⚡ Consumo Eléctrico Crítico",
    "presence_no_motion": "⚠️ Alerta: Persona inmóvil en casa",
    "security_element_open_away": "🚪 Casa vacía con puerta/ventana abierta",
    "efficiency_heating_window_open": "💸 Derroche: Calefacción y ventana abiertos",
    "ml_score_threshold": "🤖 Rutina inusual detectada (IA)",
    "ml_default_predict": "🤖 Anomalía de comportamiento general",
    "unknown": "❓ Anomalía desconocida"
}


def format_timestamp_friendly(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y - %H:%M:%S")
    except ValueError:
        return iso_str


def build_alert_text(reason_code: str, timestamp_str: str, state: dict, score: float) -> str:
    motivo_texto = REASON_TRANSLATIONS.get(reason_code, f"Anomalía: {reason_code}")
    hora_bonita = format_timestamp_friendly(timestamp_str)

    text = f"⚠️ <b>ALERTA DOMÓTICA</b> ⚠️\n\n"
    text += f"<b>Motivo:</b> {motivo_texto}\n"
    text += f"<b>Hora:</b> {hora_bonita}\n\n"

    # 1. Extraer datos principales
    water = state.get("water", {}).get("water_L_per_min", 0.0)
    power = state.get("energy", {}).get("global_power_W", 0.0)
    presence = state.get("health", {}).get("presence", 0)
    motion = state.get("health", {}).get("motion", 0)

    # 2. Resumen según el tipo de alerta
    if "water" in reason_code:
        text += f"💧 <b>Agua:</b> {water} L/min\n"
    elif "power" in reason_code:
        text += f"⚡ <b>Luz (Global):</b> {power} W\n"
    elif "presence" in reason_code or "motion" in reason_code:
        text += f"👤 <b>Estado:</b> {'🏠 En casa' if presence == 1 else '❌ Fuera'}\n"
        text += f"🏃 <b>Movimiento:</b> {'Sí' if motion == 1 else 'No'}\n"
    else:
        text += f"💧 Agua: {water} L/min | ⚡ Luz: {power} W\n"
        text += f"🏃 Mov: {motion} | 🏠 Pres: {presence}\n"

    # 3. Contexto
    text += "\n<b>📋 Contexto de la casa:</b>\n"

    # Seguridad
    security = state.get("security", {})
    open_doors = [k for k, v in security.get("doors_open", {}).items() if v]
    open_windows = [k for k, v in security.get("windows_open", {}).items() if v]

    if open_doors:
        text += f"🚪 <b>Puertas abiertas:</b> {', '.join(open_doors)}\n"
    if open_windows:
        text += f"🪟 <b>Ventanas abiertas:</b> {', '.join(open_windows)}\n"

    # Confort
    comfort = state.get("comfort", {})
    heating_on = [k for k, v in comfort.get("heating_on", {}).items() if v]
    if heating_on:
        text += f"🔥 <b>Calefacción ON en:</b> {', '.join(heating_on)}\n"

    # Salud
    fp2_zones = state.get("health", {}).get("fp2_zones", {})
    active_zones = [k for k, v in fp2_zones.items() if v]
    if active_zones:
        text += f"🛋️ <b>Presencia estática (FP2):</b> {', '.join(active_zones)}\n"

    # 4. Nivel de gravedad IA
    if score is not None and score < 0:  # Las anomalías en Isolation Forest son negativas
        # Normalizamos el score crudo de scikit-learn (típicamente hasta -0.35) a un rango [0, 1]
        MAX_ANOMALY_SCORE = 0.35
        gravedad = min(1.0, abs(score) / MAX_ANOMALY_SCORE)

        etiqueta = "🔴 CRÍTICO" if gravedad > 0.75 else "🟠 ALTO" if gravedad > 0.60 else "🟡 MODERADO"
        text += f"\n<b>Nivel de Anomalía IA:</b> {gravedad:.2f} ({etiqueta})"
    return text

async def send_telegram_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram Bot Token o Chat ID no configurados. Imprimiendo alerta en consola:")
        print(text)
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10.0)
        return True
    except Exception as e:
        logger.error(f"Error Telegram: {e}")
        return False