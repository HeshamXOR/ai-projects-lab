"""FastAPI TestClient checks for the /turn endpoint and validation."""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_turn_text_returns_intent_and_response():
    resp = client.post("/turn", json={"text": "what is the weather in london", "session_id": "s1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "check_weather"
    assert body["transcript"] == "what is the weather in london"
    assert body["response"]
    assert "london" in body["response"].lower()


def test_turn_missing_slot_flow():
    # destination only -> awaiting the date slot
    resp = client.post(
        "/turn", json={"text": "book a flight to paris", "session_id": "s2"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "book_flight"
    assert body["state"] == "AWAITING_SLOT"
    assert "date" in body["missing_slots"]


def test_turn_validation_error_on_empty_payload():
    resp = client.post("/turn", json={})
    assert resp.status_code == 422  # pydantic validation: no text or audio


def test_turn_validation_error_on_blank_text():
    resp = client.post("/turn", json={"text": "   "})
    # empty/whitespace text fails the model validator (no usable input)
    assert resp.status_code in (400, 422)


def test_health_and_root():
    assert client.get("/health").json()["status"] == "ok"
    root = client.get("/").json()
    assert root["service"] == "speech-agent"
    assert "POST /turn" in root["endpoints"]


def test_ws_turn_streams_result():
    with client.websocket_connect("/ws/turn") as ws:
        ws.send_json({"text": "hello there", "session_id": "ws1"})
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["event"] == "result":
                break
        result = events[-1]
        assert result["intent"] == "greet"
        assert result["response"]
        assert any(e["event"] == "stage" for e in events)
