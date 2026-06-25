"""FastAPI endpoint tests using TestClient."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def _seasonal(n: int = 96, m: int = 12) -> list[float]:
    t = np.arange(n, dtype=float)
    return list(50.0 + 0.3 * t + 10.0 * np.sin(2 * np.pi * t / m))


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_info() -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "timeseries-forecaster"
    assert "ensemble" in body["models"]


def test_forecast_happy_path() -> None:
    payload = {
        "series": _seasonal(),
        "horizon": 6,
        "season_length": 12,
        "model": "ensemble",
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["point"]) == 6
    assert len(body["lower"]) == 6
    assert len(body["upper"]) == 6
    assert body["model"] == "ensemble"
    for lo, pt, hi in zip(body["lower"], body["point"], body["upper"]):
        assert lo <= pt <= hi


def test_forecast_holtwinters_model() -> None:
    payload = {
        "series": _seasonal(),
        "horizon": 4,
        "season_length": 12,
        "model": "holtwinters",
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "holtwinters"


def test_forecast_rejects_nonpositive_horizon() -> None:
    payload = {"series": _seasonal(), "horizon": 0, "season_length": 12}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422  # Pydantic gt=0 validation


def test_forecast_rejects_short_seasonal_series() -> None:
    # season_length=12 but only 10 points -> fails 2*m rule.
    payload = {"series": [1.0] * 10, "horizon": 3, "season_length": 12}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422
