from datetime import datetime

from src.api.routes import deep_update
from src.core.rules import evaluate_all_business_rules, evaluate_business_rules


def test_deep_update_preserves_existing_keys_and_ignores_empty_dicts():
    state = {
        "energy": {"global_power_W": 120.0, "plugs_W": {"tv": 40.0}},
        "security": {"doors_open": {"puerta": True}, "windows_open": {"salon": True}},
    }

    # Un diccionario vacío no debe borrar el estado almacenado
    deep_update(state, {"security": {}, "energy": {"plugs_W": {}}})
    assert state["security"]["doors_open"] == {"puerta": True}
    assert state["security"]["windows_open"] == {"salon": True}
    assert state["energy"]["plugs_W"] == {"tv": 40.0}

    # Actualización parcial mantiene los demás sensores del mismo subdiccionario
    deep_update(state, {"security": {"windows_open": {"bano": False}}})
    assert state["security"]["windows_open"] == {"salon": True, "bano": False}


def test_rules_power_boundary_and_water_leak():
    base = {
        "water": {"water_L_per_min": 15.0},
        "energy": {"global_power_W": 3499.9},
        "health": {"presence": 1, "motion": 1},
        "security": {"doors_open": {}, "windows_open": {}},
        "comfort": {"heating_on": {}},
    }
    assert evaluate_all_business_rules(base, "2026-07-02T12:00:00") == []
    assert evaluate_business_rules(base, "2026-07-02T12:00:00") == "normal"

    # 3500.0 W exactos sí debe disparar sobrecarga
    base["energy"]["global_power_W"] = 3500.0
    assert evaluate_all_business_rules(base, "2026-07-02T12:00:00Z") == ["power_high"]

    # Fuga crítica (> 15.0 L/min)
    base["water"]["water_L_per_min"] = 15.1
    assert evaluate_all_business_rules(base, "2026-07-02T12:00:00") == [
        "water_critical_high",
        "power_high",
    ]


def test_rules_simultaneous_emergencies_are_not_hidden():
    ts = "2026-07-02T14:00:00"
    unix_ts = datetime.fromisoformat(ts).timestamp()
    state = {
        "water": {"water_L_per_min": 120.0},
        "energy": {"global_power_W": 3500.0},
        "health": {"presence": 1, "motion": 0, "last_motion_timestamp": unix_ts - 4000},
        "security": {"doors_open": {}, "windows_open": {"ventana_bano": True}},
        "comfort": {"heating_on": {"termostato_bano": True}},
    }
    triggered = evaluate_all_business_rules(state, ts)
    assert triggered == [
        "water_critical_high",
        "power_high",
        "presence_no_motion",
        "efficiency_heating_window_open",
    ]
    assert evaluate_business_rules(state, ts) == "water_critical_high"


def test_rules_immobility_only_during_daytime():
    day_ts = "2026-07-02T15:00:00"
    day_unix = datetime.fromisoformat(day_ts).timestamp()
    state = {
        "water": {"water_L_per_min": 0.0},
        "energy": {"global_power_W": 80.0},
        "health": {"presence": 1, "motion": 0, "last_motion_timestamp": day_unix - 3700},
        "security": {},
        "comfort": {},
    }
    assert evaluate_all_business_rules(state, day_ts) == ["presence_no_motion"]

    # De noche (03:00) no debe dispararse la alerta de inmovilidad diurna
    night_ts = "2026-07-02T03:00:00"
    night_unix = datetime.fromisoformat(night_ts).timestamp()
    state["health"]["last_motion_timestamp"] = night_unix - 7200
    assert evaluate_all_business_rules(state, night_ts) == []


def test_rules_security_away_open_access():
    state = {
        "water": {"water_L_per_min": 0.0},
        "energy": {"global_power_W": 50.0},
        "health": {"presence": 0, "motion": 0},
        "security": {"doors_open": {"puerta_principal": True}, "windows_open": {}},
        "comfort": {},
    }
    assert evaluate_all_business_rules(state, "2026-07-02T11:00:00") == ["security_element_open_away"]
