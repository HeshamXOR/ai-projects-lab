"""FastAPI TestClient checks for the anomaly-sentinel service."""

import numpy as np
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_and_root():
    assert client.get("/health").status_code == 200
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "anomaly-sentinel"


def test_ingest_then_alerts():
    # Stream a stable series then a big spike; expect at least one alert.
    rng = np.random.default_rng(0)
    stable = list(rng.normal(10.0, 0.3, size=60))
    client.post("/ingest", json={"metric": "cpu", "values": stable})
    client.post("/ingest", json={"metric": "cpu", "values": [200.0]})

    r = client.get("/alerts", params={"metric": "cpu"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert all(a["metric"] == "cpu" for a in body["alerts"])


def test_reference_and_drift_check_detects_shift():
    rng = np.random.default_rng(1)
    ref = list(rng.normal(0.0, 1.0, size=1000))
    shifted = list(rng.normal(4.0, 1.0, size=1000))

    assert client.post("/reference", json={"metric": "score", "values": ref}).status_code == 200
    r = client.post("/drift/check", json={"metric": "score", "sample": shifted})
    assert r.status_code == 200
    body = r.json()
    assert body["drift"] is True
    assert body["psi"]["drift"] is True


def test_drift_check_without_reference_404():
    r = client.post("/drift/check", json={"metric": "nope", "sample": [1.0, 2.0, 3.0]})
    assert r.status_code == 404


def test_ingest_validation_error():
    # Empty values list -> 422 from Pydantic validator.
    r = client.post("/ingest", json={"metric": "x", "values": []})
    assert r.status_code == 422


def test_alerts_bad_severity_400():
    r = client.get("/alerts", params={"severity": "BOGUS"})
    assert r.status_code == 400
