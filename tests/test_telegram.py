import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from src.core import config
from src.services import telegram


def test_build_alert_text_escapes_html_and_lists_extra_reasons():
    state = {
        "water": {"water_L_per_min": 120.0},
        "energy": {"global_power_W": 3500.0},
        "health": {"presence": 1, "motion": 0, "fp2_zones": {"suelo <bano> & 1": True}},
        "security": {"doors_open": {"puerta<1>": True}, "windows_open": {"ventana&2": True}},
        "comfort": {"heating_on": {"tado<x>": True}},
    }
    msg = telegram.build_alert_text(
        "water_critical_high",
        "2026-07-02T22:45:45Z",
        state,
        0.0,
        extra_reasons=["power_high", "efficiency_heating_window_open"],
    )
    assert "&lt;bano&gt; &amp; 1" in msg
    assert "puerta&lt;1&gt;" in msg
    assert "ventana&amp;2" in msg
    assert "tado&lt;x&gt;" in msg
    assert "Alertas simultáneas" in msg
    assert "Consumo Eléctrico Crítico" in msg
    assert "Nivel de Anomalía IA" not in msg


def test_build_alert_text_includes_ai_severity_for_ml_alerts():
    state = {"water": {"water_L_per_min": 5.0}, "energy": {"global_power_W": 1800.0}, "health": {"presence": 1, "motion": 1}}
    msg = telegram.build_alert_text("ml_score_threshold", "2026-07-02T03:15:00", state, -0.20)
    assert "Nivel de Anomalía IA" in msg
    assert "CRÍTICO" in msg


def test_send_telegram_message_checks_http_status_and_ok_flag(monkeypatch):
    # Sin configuración -> False
    monkeypatch.setattr(config, "BOT_TOKEN", None)
    monkeypatch.setattr(config, "CHAT_ID", None)
    assert asyncio.run(telegram.send_telegram_message("test")) is False

    monkeypatch.setattr(config, "BOT_TOKEN", "fake-token")
    monkeypatch.setattr(config, "CHAT_ID", "12345")
    req = httpx.Request("POST", "https://api.telegram.org/botfake-token/sendMessage")

    # HTTP 400 -> False
    resp_400 = httpx.Response(400, json={"ok": False, "description": "Bad Request"}, request=req)
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp_400)):
        assert asyncio.run(telegram.send_telegram_message("test")) is False

    # HTTP 200 pero ok=False -> False
    resp_not_ok = httpx.Response(200, json={"ok": False}, request=req)
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp_not_ok)):
        assert asyncio.run(telegram.send_telegram_message("test")) is False

    # HTTP 200 y ok=True -> True
    resp_ok = httpx.Response(200, json={"ok": True, "result": {"message_id": 1}}, request=req)
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp_ok)):
        assert asyncio.run(telegram.send_telegram_message("test")) is True
