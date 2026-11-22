from unittest.mock import AsyncMock

import pytest

from src.api import routes
from src.core import config

AUTH_HEADERS = {"X-API-Key": "test-secret-api-key"}


def test_healthcheck_does_not_leak_home_state(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "online"}


def test_state_endpoint_requires_valid_api_key(api_client, monkeypatch):
    # Sin cabecera
    resp_missing = api_client.get("/state")
    assert resp_missing.status_code in (401, 403)

    # Clave incorrecta
    resp_bad = api_client.get("/state", headers={"X-API-Key": "clave-falsa"})
    assert resp_bad.status_code == 403

    # Clave válida
    resp_ok = api_client.get("/state", headers=AUTH_HEADERS)
    assert resp_ok.status_code == 200
    assert "state" in resp_ok.json()

    # Servidor sin API_KEY configurada -> fail-closed (503), jamás acepta "CLAVE_NO_CONFIGURADA"
    monkeypatch.setattr(config, "SECRET_API_KEY", "")
    resp_unconfigured = api_client.get("/state", headers={"X-API-Key": "CLAVE_NO_CONFIGURADA"})
    assert resp_unconfigured.status_code == 503


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"water": {"water_L_per_min": -1.0}},
        {"energy": {"global_power_W": -10.0}},
        {"energy": {"plugs_W": {"tv": -5.0}}},
        {"comfort": {"target_temp": {"salon": 99.0}}},
        {"health": {"presence": 2}},
        {"health": {"motion": -1}},
        {"timestamp": "fecha-no-valida"},
    ],
)
def test_detect_rejects_invalid_schema_values(api_client, bad_payload):
    resp = api_client.post("/detect", json=bad_payload, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_partial_updates_preserve_state_and_do_not_reset_last_motion(api_client):
    # 1. Residente en casa con movimiento a las 10:00
    r1 = api_client.post(
        "/detect",
        json={
            "timestamp": "2026-07-02T10:00:00",
            "energy": {"global_power_W": 110.0},
            "health": {"presence": 1, "motion": 1},
            "water": {"water_L_per_min": 0.0},
            "security": {"windows_open": {"ventana_salon": False}},
        },
        headers=AUTH_HEADERS,
    )
    assert r1.status_code == 200
    ts_motion_10 = r1.json()["current_state"]["health"]["last_motion_timestamp"]

    # 2. A las 11:15 llega una actualización parcial solo de energía (sin campo health)
    #    No debe resetear last_motion_timestamp, y al superar 3600s debe saltar inmovilidad.
    r2 = api_client.post(
        "/detect",
        json={
            "timestamp": "2026-07-02T11:15:00",
            "energy": {"global_power_W": 115.0},
            "security": {},
        },
        headers=AUTH_HEADERS,
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["current_state"]["health"]["last_motion_timestamp"] == ts_motion_10
    assert body["current_state"]["security"]["windows_open"] == {"ventana_salon": False}
    assert body["anomaly"] is True
    assert "presence_no_motion" in body["reasons"]


def test_detect_deterministic_emergency_and_cooldown_behavior(api_client, monkeypatch):
    mock_send = AsyncMock(return_value=True)
    monkeypatch.setattr(routes, "send_telegram_message", mock_send)

    payload = {
        "timestamp": "2026-07-02T14:00:00",
        "energy": {"global_power_W": 3500.0},
        "water": {"water_L_per_min": 120.0},
        "health": {"presence": 1, "motion": 1},
        "security": {"windows_open": {"ventana_bano": True}},
        "comfort": {"heating_on": {"termostato_bano": True}},
    }

    r1 = api_client.post("/detect", json=payload, headers=AUTH_HEADERS)
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["anomaly"] is True
    assert b1["reason"] == "water_critical_high"
    assert b1["reasons"] == ["water_critical_high", "power_high", "efficiency_heating_window_open"]
    assert mock_send.await_count == 1

    # Segunda petición inmediata: bloqueada por cooldown (no vuelve a enviar a Telegram)
    r2 = api_client.post("/detect", json=payload, headers=AUTH_HEADERS)
    assert r2.status_code == 200
    assert mock_send.await_count == 1


def test_failed_telegram_delivery_does_not_consume_cooldown(api_client, monkeypatch):
    mock_send = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(routes, "send_telegram_message", mock_send)

    payload = {
        "timestamp": "2026-07-02T14:00:00",
        "water": {"water_L_per_min": 50.0},
        "health": {"presence": 1, "motion": 1},
    }

    # Primer intento falla -> no debe grabar cooldown
    api_client.post("/detect", json=payload, headers=AUTH_HEADERS)
    # Segundo intento vuelve a intentar y tiene éxito
    api_client.post("/detect", json=payload, headers=AUTH_HEADERS)
    assert mock_send.await_count == 2


def test_detect_behavioral_ai_anomaly_and_normal_reading(api_client, monkeypatch):
    mock_send = AsyncMock(return_value=True)
    monkeypatch.setattr(routes, "send_telegram_message", mock_send)

    # Lectura normal de madrugada durmiendo (50 W, sin agua, presencia=1, fp2 cama=True)
    normal_payload = {
        "timestamp": "2026-07-02T03:00:00",
        "energy": {"global_power_W": 50.0},
        "water": {"water_L_per_min": 0.0},
        "health": {"presence": 1, "motion": 0, "fp2_zones": {"cama": True}},
        "security": {"doors_open": {"puerta_principal": False}, "windows_open": {"ventana_salon": False}},
        "comfort": {"heating_on": {"termostato_salon": False}},
    }
    r_norm = api_client.post("/detect", json=normal_payload, headers=AUTH_HEADERS)
    assert r_norm.status_code == 200
    assert r_norm.json()["anomaly"] is False
    assert r_norm.json()["reason"] == "normal"

    # Anomalía conductual a las 03:15 AM (debajo de umbrales de reglas, capturada por IA)
    ai_payload = {
        "timestamp": "2026-07-02T03:15:00",
        "energy": {"global_power_W": 1800.0},
        "water": {"water_L_per_min": 6.5},
        "health": {"presence": 1, "motion": 1, "fp2_zones": {"cama": False}},
        "security": {"doors_open": {"puerta_principal": True}, "windows_open": {"ventana_salon": False}},
        "comfort": {"heating_on": {"termostato_salon": False}},
    }
    r_ai = api_client.post("/detect", json=ai_payload, headers=AUTH_HEADERS)
    assert r_ai.status_code == 200
    body = r_ai.json()
    assert body["anomaly"] is True
    assert body["reason"] == "ml_score_threshold"
    assert body["ai_score"] < 0.0
    assert mock_send.await_count == 1


def test_internal_error_hides_exception_details(api_client, monkeypatch):
    monkeypatch.setattr(routes, "predict_from_state", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("ruta/secreta/interna")))
    resp = api_client.post(
        "/detect",
        json={
            "timestamp": "2026-07-02T12:00:00",
            "energy": {"global_power_W": 80.0},
            "water": {"water_L_per_min": 0.0},
            "health": {"presence": 1, "motion": 1},
        },
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 500
    assert "ruta/secreta/interna" not in resp.text
