"""FastAPI integration tests using TestClient. Resets in-memory state per test."""

from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)

CORPUS = [
    "Acme Corp acquired Beta Inc in 2022 for 12 million dollars.",
    "Beta Inc owns Gamma Labs, a robotics subsidiary.",
    "Gamma Labs manufactures autonomous warehouse robots.",
    "The office cafeteria introduced a new spring menu.",
]


def setup_function(_func):
    """Fresh state before each test."""
    app_module.STATE.reset()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "knowledge-graph-rag"


def test_ingest_then_ask():
    r = client.post("/ingest", json={"documents": CORPUS})
    assert r.status_code == 200
    body = r.json()
    assert body["passages_added"] == len(CORPUS)
    assert body["entities"] >= 3

    r = client.post("/ask", json={"question": "What did Acme Corp acquire?", "k": 3})
    assert r.status_code == 200
    ans = r.json()
    assert ans["passages"]
    assert ans["answer"]
    assert ans["used_graph"] is True


def test_graph_endpoint_shows_nodes():
    client.post("/ingest", json={"documents": CORPUS})
    r = client.get("/graph")
    assert r.status_code == 200
    g = r.json()
    assert g["num_nodes"] >= 3
    assert "Acme Corp" in g["nodes"]
    assert g["edges"]


def test_graph_path():
    client.post("/ingest", json={"documents": CORPUS})
    r = client.get("/graph/path", params={"src": "Acme Corp", "dst": "Gamma Labs"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["path"][0] == "Acme Corp"
    assert body["path"][-1] == "Gamma Labs"


def test_graph_path_unknown_entity_404():
    client.post("/ingest", json={"documents": CORPUS})
    r = client.get("/graph/path", params={"src": "Nonexistent", "dst": "Gamma Labs"})
    assert r.status_code == 404


def test_ingest_validation_error():
    # neither text nor documents -> validation error
    r = client.post("/ingest", json={})
    assert r.status_code == 422


def test_ask_before_ingest_errors():
    r = client.post("/ask", json={"question": "anything?"})
    assert r.status_code == 400
