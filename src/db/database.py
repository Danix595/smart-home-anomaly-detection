import sqlite3
import json
import time
from src.core.config import DB_FILE


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS home_state (id TEXT PRIMARY KEY, state_data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS cooldowns (alert_key TEXT PRIMARY KEY, last_timestamp REAL)''')

    c.execute("SELECT count(*) FROM home_state WHERE id='current'")
    if c.fetchone()[0] == 0:
        # Aquí definimos la estructura base inicial de la casa inteligente
        default_state = {
            "energy": {"global_power_W": 0.0, "plugs_W": {}, "lights_on": {}},
            "security": {"doors_open": {}, "windows_open": {}},
            "comfort": {"heating_on": {}, "target_temp": {}, "blinds_open": {}},
            "health": {
                "presence": 0,
                "motion": 0,
                "last_motion_timestamp": time.time(),
                "fp2_zones": {}
            },
            "water": {"water_L_per_min": 0.0}
        }
        c.execute("INSERT INTO home_state (id, state_data) VALUES ('current', ?)", (json.dumps(default_state),))

    conn.commit()
    conn.close()


def get_state() -> dict:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT state_data FROM home_state WHERE id='current'")
    state = json.loads(c.fetchone()[0])
    conn.close()
    return state


def save_state(state: dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE home_state SET state_data=? WHERE id='current'", (json.dumps(state),))
    conn.commit()
    conn.close()


def get_last_alert_time(alert_key: str) -> float:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT last_timestamp FROM cooldowns WHERE alert_key=?", (alert_key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0.0


def save_last_alert_time(alert_key: str, timestamp: float):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO cooldowns (alert_key, last_timestamp) VALUES (?, ?)", (alert_key, timestamp))
    conn.commit()
    conn.close()