"""speech-agent — FastAPI voice-assistant pipeline (ASR -> intent -> dialog -> TTS).

POST a turn as ``{text}`` or ``{audio_b64}``; the service runs ASR (when audio is
given), classifies the intent, fills slots, advances the dialog FSM, and returns
a structured turn result. A WebSocket endpoint ``/ws/turn`` streams the same
pipeline turn-by-turn for an interactive session.

By default the app uses the deterministic MockASR/MockTTS so it runs with no
pretrained models. Swap in WhisperASR / Pyttsx3TTS (see ``core/interfaces.py``)
for real speech.
"""

from __future__ import annotations

import base64
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, model_validator

from core.dialog import DialogManager
from core.interfaces import ASR, TTS, MockASR, MockTTS
from core.intent import IntentClassifier

# --------------------------------------------------------------------------
# Training data for the bundled intent classifier (small, illustrative set).
# --------------------------------------------------------------------------

TRAIN_TEXTS: List[str] = [
    # greet
    "hello there", "hi", "hey good morning", "good evening", "hello assistant",
    # goodbye
    "goodbye", "bye now", "see you later", "that is all thanks bye", "goodbye assistant",
    # book_flight
    "book a flight to paris", "i want to fly to london tomorrow",
    "reserve a flight to tokyo", "can you book me a flight to berlin",
    "i need a flight to madrid on june 25",
    # check_weather
    "what is the weather in london", "weather forecast for paris",
    "is it going to rain in berlin", "how is the weather in tokyo today",
    "tell me the weather in madrid",
    # set_alarm
    "set an alarm for 7am", "wake me up at 6:30",
    "set alarm for tomorrow at 8am", "remind me with an alarm at 9pm",
    "set an alarm for noon",
]

TRAIN_LABELS: List[str] = (
    ["greet"] * 5
    + ["goodbye"] * 5
    + ["book_flight"] * 5
    + ["check_weather"] * 5
    + ["set_alarm"] * 5
)


def build_classifier() -> IntentClassifier:
    """Train the bundled intent classifier on the demo data."""
    clf = IntentClassifier(n_iters=1000, learning_rate=0.5)
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return clf


# --------------------------------------------------------------------------
# Pydantic request/response models.
# --------------------------------------------------------------------------


class TurnRequest(BaseModel):
    """A single conversational turn: provide ``text`` or ``audio_b64``."""

    text: Optional[str] = Field(default=None, description="User utterance text.")
    audio_b64: Optional[str] = Field(
        default=None, description="Base64-encoded audio for ASR."
    )
    session_id: str = Field(default="default", description="Conversation session key.")

    @model_validator(mode="after")
    def _require_input(self) -> "TurnRequest":
        if not self.text and not self.audio_b64:
            raise ValueError("Provide either 'text' or 'audio_b64'.")
        return self


class TurnResponse(BaseModel):
    """Structured result of one pipeline turn."""

    transcript: str
    intent: str
    slots: Dict[str, object]
    missing_slots: List[str]
    state: str
    response: str
    session_id: str


# --------------------------------------------------------------------------
# Application + per-session state.
# --------------------------------------------------------------------------

app = FastAPI(
    title="speech-agent",
    description="Voice-assistant pipeline: ASR -> intent -> slots -> dialog -> TTS.",
    version="1.0.0",
)

_classifier: IntentClassifier = build_classifier()
_asr: ASR = MockASR()
_tts: TTS = MockTTS()
_sessions: Dict[str, DialogManager] = {}


def _manager(session_id: str) -> DialogManager:
    """Return (creating if needed) the dialog manager for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = DialogManager()
    return _sessions[session_id]


def _run_turn(text: Optional[str], audio_b64: Optional[str], session_id: str) -> TurnResponse:
    """Execute the full pipeline for one turn and build the response."""
    transcript = text or ""
    if audio_b64:
        try:
            audio = base64.b64decode(audio_b64)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 audio.") from exc
        transcript = _asr.transcribe(audio)

    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Empty transcript after ASR.")

    intent = _classifier.predict(transcript)
    manager = _manager(session_id)
    result = manager.handle(intent, transcript)

    return TurnResponse(
        transcript=transcript,
        intent=result.intent or intent,
        slots=result.slots,
        missing_slots=result.missing,
        state=result.state.value,
        response=result.response,
        session_id=session_id,
    )


# --------------------------------------------------------------------------
# Routes.
# --------------------------------------------------------------------------


@app.get("/")
def root() -> Dict[str, object]:
    """Service info and available endpoints."""
    return {
        "service": "speech-agent",
        "version": "1.0.0",
        "pipeline": ["asr", "intent", "slots", "dialog", "tts"],
        "endpoints": ["POST /turn", "WS /ws/turn", "GET /health", "GET /"],
        "intents": sorted(set(TRAIN_LABELS)),
        "note": "Uses MockASR/MockTTS by default; runs with no pretrained models.",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/turn", response_model=TurnResponse)
def turn(req: TurnRequest) -> TurnResponse:
    """Run one conversational turn through the full pipeline."""
    return _run_turn(req.text, req.audio_b64, req.session_id)


@app.websocket("/ws/turn")
async def ws_turn(websocket: WebSocket) -> None:
    """Streaming turn endpoint.

    Accepts JSON messages of the same shape as :class:`TurnRequest` and streams
    back the response. Streaming is simulated by emitting the pipeline stages
    (``stage`` events) before the final ``result`` event, so a client sees
    incremental progress.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text")
            audio_b64 = data.get("audio_b64")
            session_id = data.get("session_id", "default")

            if not text and not audio_b64:
                await websocket.send_json(
                    {"event": "error", "detail": "Provide 'text' or 'audio_b64'."}
                )
                continue

            await websocket.send_json({"event": "stage", "stage": "asr"})
            await websocket.send_json({"event": "stage", "stage": "intent"})
            try:
                result = _run_turn(text, audio_b64, session_id)
            except HTTPException as exc:
                await websocket.send_json({"event": "error", "detail": exc.detail})
                continue
            await websocket.send_json({"event": "stage", "stage": "dialog"})
            await websocket.send_json({"event": "result", **result.model_dump()})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
