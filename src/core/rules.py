from datetime import datetime
import time
from typing import List


def _parse_timestamp(timestamp_str: str) -> tuple[int, float]:
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return ts.hour, ts.timestamp()
    except (ValueError, TypeError, AttributeError):
        now = datetime.now()
        return now.hour, time.time()


def evaluate_all_business_rules(state: dict, current_timestamp_str: str) -> List[str]:
    """Evalúa todas las reglas deterministas y devuelve las disparadas por orden de gravedad."""
    hour, current_unix_time = _parse_timestamp(current_timestamp_str)
    triggered: List[str] = []

    # 1. AGUA (Crítica): Posible fuga
    water = state.get("water", {}).get("water_L_per_min")
    if water is not None and water > 15.0:
        triggered.append("water_critical_high")

    # 2. ENERGÍA (Crítica): Riesgo de sobrecarga / incendio
    power = state.get("energy", {}).get("global_power_W")
    if power is not None and power >= 3500.0:
        triggered.append("power_high")

    # 3. SALUD (Alta): Persona inmóvil mucho tiempo durante el día
    health = state.get("health", {})
    presence = health.get("presence", 0)
    last_motion = health.get("last_motion_timestamp", current_unix_time)

    if presence == 1 and (8 <= hour < 22):
        seconds_without_motion = current_unix_time - last_motion
        if seconds_without_motion > 3600:
            triggered.append("presence_no_motion")

    # 4. SEGURIDAD (Alta): Casa vacía con accesos abiertos (olvido o intrusión)
    security = state.get("security", {})
    doors_open = any(security.get("doors_open", {}).values())
    windows_open = any(security.get("windows_open", {}).values())

    if presence == 0 and (doors_open or windows_open):
        triggered.append("security_element_open_away")

    # 5. CONFORT / EFICIENCIA (Moderada): Derroche térmico
    comfort = state.get("comfort", {})
    heating_on = any(comfort.get("heating_on", {}).values())

    if heating_on and windows_open:
        triggered.append("efficiency_heating_window_open")

    return triggered


def evaluate_business_rules(state: dict, current_timestamp_str: str) -> str:
    """Devuelve la regla determinista más prioritaria disparada, o 'normal'."""
    triggered = evaluate_all_business_rules(state, current_timestamp_str)
    return triggered[0] if triggered else "normal"
