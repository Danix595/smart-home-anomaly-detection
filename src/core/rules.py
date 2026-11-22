from datetime import datetime
import time

def evaluate_business_rules(state: dict, current_timestamp_str: str) -> str:
    try:
        ts = datetime.fromisoformat(current_timestamp_str)
        hour = ts.hour
        current_unix_time = ts.timestamp()
    except (ValueError, TypeError):
        hour = datetime.now().hour
        current_unix_time = time.time()

    # 1. AGUA: Fuga
    water = state.get("water", {}).get("water_L_per_min", 0.0)
    if water > 15.0:
        return "water_critical_high"

    # 2. ENERGÍA: Riesgo de sobrecarga / incendio
    power = state.get("energy", {}).get("global_power_W", 0.0)
    if power > 3500:
        return "power_high"

    # 3. SALUD: Persona inmóvil mucho tiempo (Solo de día)
    health = state.get("health", {})
    presence = health.get("presence", 0)
    motion = health.get("motion", 0)
    last_motion = health.get("last_motion_timestamp", current_unix_time)

    if presence == 1 and (8 <= hour < 22):
        seconds_without_motion = current_unix_time - last_motion
        if seconds_without_motion > 3600 and motion == 0:
            return "presence_no_motion"

    # 4. SEGURIDAD: Casa vacía con accesos abiertos (Olvido o Intrusión)
    security = state.get("security", {})
    doors_open = any(security.get("doors_open", {}).values())
    windows_open = any(security.get("windows_open", {}).values())

    if presence == 0 and (doors_open or windows_open):
        return "security_element_open_away"

    # 5. CONFORT / EFICIENCIA: Derroche energético
    comfort = state.get("comfort", {})
    heating_on = any(comfort.get("heating_on", {}).values())

    if heating_on and windows_open:
        return "efficiency_heating_window_open"

    return "normal"