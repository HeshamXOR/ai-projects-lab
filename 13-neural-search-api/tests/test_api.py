"""API smoke tests using FastAPI's TestClient."""

from __future__ import annotations

import importlib

import pytest

# Skip the whole module gracefully if FastAPI/httpx are not installed.
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A fresh TestClient backed by an empty, isolated index dir."""
    monkeypatch.setenv("NEURAL_SEARCH_INDEX_DIR", str(tmp_path / "idx"))
    import app as app_module

    importlib.reload(app_module)  # rebuild retriever against the temp dir
    return TestClient(app_module.app)


def test_health(client):
    """Health endpoint reports an empty, ok service."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["num_docs"] == 0


def test_index_then_search(client):
    """Index a batch, then search and get a sensible top hit."""
    docs = {
        "documents": [
            {"id": "1", "text": "vector databases enable semantic search", "metadata": {"c": "tech"}},
            {"id": "2", "text": "the cat sat on the mat", "metadata": {"c": "animal"}},
            {"id": "3", "text": "transformers power modern nlp models", "metadata": {"c": "tech"}},
        ]
    }
    r = client.post("/index", json=docs)
    assert r.status_code == 200
    assert r.json()["indexed"] == 3

    r = client.post("/search", json={"query": "semantic vector search", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert body["results"][0]["id"] == "1"


def test_search_with_filter_and_pagination(client):
    """Metadata filter narrows results; limit/offset paginate."""
    client.post(
        "/index",
        json={
            "documents": [
                {"id": "a", "text": "tech one machine learning", "metadata": {"c": "tech"}},
                {"id": "b", "text": "tech two deep learning", "metadata": {"c": "tech"}},
                {"id": "c", "text": "animal cat dog", "metadata": {"c": "animal"}},
            ]
        },
    )
    r = client.post(
        "/search",
        json={"query": "learning tech", "k": 10, "filter": {"c": "tech"}, "limit": 1, "offset": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert all(hit["metadata"]["c"] == "tech" for hit in body["results"])
    assert len(body["results"]) == 1


def test_delete_and_404(client):
    """Delete removes a doc; deleting a missing doc returns 404."""
    client.post("/index", json={"documents": [{"id": "z", "text": "hello world", "metadata": {}}]})

    r = client.get("/doc/z")
    assert r.status_code == 200

    r = client.delete("/doc/z")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = client.get("/doc/z")
    assert r.status_code == 404

    r = client.delete("/doc/does-not-exist")
    assert r.status_code == 404


def test_validation_422(client):
    """Pydantic rejects an empty document batch with 422."""
    r = client.post("/index", json={"documents": []})
    assert r.status_code == 422

    r = client.post("/search", json={"query": "", "k": 5})
    assert r.status_code == 422
