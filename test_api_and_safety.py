import pathlib

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app

client = TestClient(app)


def test_health_demo_mode_without_any_credentials():
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["provider"]["data_mode"] == "SIMULATED"
    assert r["live_order_execution_enabled"] is False


def test_quote_is_labelled_simulated():
    r = client.get("/api/v1/market/quote/TCS").json()
    assert r["data_mode"] == "SIMULATED" and r["timezone"] == "Asia/Kolkata" and "+05:30" in r["as_of"]


def test_candles_and_indicators_endpoints():
    c = client.get("/api/v1/market/candles/INFY").json()
    assert c["data_mode"] == "SIMULATED" and c["count"] > 200
    i = client.get("/api/v1/market/indicators/INFY").json()
    assert i["computed_by"] == "python" and i["indicators"]["rsi_14"] is not None
    assert i["unavailable"] == {} or "sma_200" in i["unavailable"]


def test_consistent_error_shapes():
    assert client.get("/api/v1/market/quote/NOPE").json()["error"]["code"] == "instrument_not_found"
    r = client.get("/api/v1/market/candles/TCS?interval=5m")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_interval"
    r = client.get("/api/v1/market/candles/TCS?start=2025-02-01&end=2025-01-01")
    assert r.status_code == 400
    assert client.get("/api/v1/market/instruments?exchange=XYZ").status_code == 422


def test_cors_rejects_unknown_origin():
    r = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_live_order_flag_cannot_be_enabled():
    with pytest.raises(ValueError, match="never places real orders"):
        Settings(live_order_execution_enabled=True, _env_file=None)


def test_no_broker_order_placement_code_exists():
    banned = ("order/place", "place_order", "api-hft.upstox.com")
    for f in pathlib.Path("app").rglob("*.py"):
        if f.name == "test_api_and_safety.py":
            continue
        text = f.read_text(encoding="utf-8")
        assert not any(b in text for b in banned), f"{f} references broker order placement"
