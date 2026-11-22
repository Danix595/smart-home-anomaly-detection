from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from src.core import config


def default_home_state() -> dict:
    """Estructura base inicial del estado de la vivienda."""
    return {
        "energy": {"global_power_W": 0.0, "plugs_W": {}, "lights_on": {}},
        "security": {"doors_open": {}, "windows_open": {}},
        "comfort": {"heating_on": {}, "target_temp": {}, "blinds_open": {}},
        "health": {
            "presence": 0,
            "motion": 0,
            "last_motion_timestamp": time.time(),
            "fp2_zones": {},
        },
        "water": {"water_L_per_min": 0.0},
    }


def _get_connection() -> sqlite3.Connection:
    db_file = config.DB_FILE
    if db_file != ":memory:":
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS home_state (id TEXT PRIMARY KEY, state_data TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cooldowns (alert_key TEXT PRIMARY KEY, last_timestamp REAL)"
            )
            row = conn.execute(
                "SELECT 1 FROM home_state WHERE id='current'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO home_state (id, state_data) VALUES ('current', ?)",
                    (json.dumps(default_home_state()),),
                )


def get_state() -> dict:
    with closing(_get_connection()) as conn:
        row = conn.execute(
            "SELECT state_data FROM home_state WHERE id='current'"
        ).fetchone()
        if row is None:
            state = default_home_state()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO home_state (id, state_data) VALUES ('current', ?)",
                    (json.dumps(state),),
                )
            return state
        return json.loads(row[0])


def save_state(state: dict) -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO home_state (id, state_data) VALUES ('current', ?)",
                (json.dumps(state),),
            )


def get_last_alert_time(alert_key: str) -> float:
    with closing(_get_connection()) as conn:
        row = conn.execute(
            "SELECT last_timestamp FROM cooldowns WHERE alert_key=?",
            (alert_key,),
        ).fetchone()
        return float(row[0]) if row else 0.0


def save_last_alert_time(alert_key: str, timestamp: float) -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO cooldowns (alert_key, last_timestamp) VALUES (?, ?)",
                (alert_key, timestamp),
            )
