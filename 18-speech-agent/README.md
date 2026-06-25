# 🎙️ speech-agent — a voice assistant pipeline with the hard parts built from scratch

## What I implemented from scratch

- **Intent classifier** — a hand-rolled **TF-IDF vectorizer** (vocab build, term
  frequency, smoothed `idf = log((1+N)/(1+df)) + 1`, L2 row normalization) feeding
  a **multinomial logistic regression** trained by batch gradient descent in NumPy
  (softmax, cross-entropy, analytic gradient, weight updates). No sklearn for the
  model — `core/intent.py`.
- **Slot filling** — regex + keyword/gazetteer extractors for dates, times,
  numbers, and cities, with per-intent required/optional slot schemas and a
  missing-slot report — `core/slots.py`.
- **Dialog FSM** — an explicit finite state machine (`IDLE → LISTENING →
  AWAITING_SLOT → CONFIRMING → RESPONDING → DONE`) with a transition table,
  per-session context, slot-driven follow-ups, and confirmation handling —
  `core/dialog.py`.
- **Endpointing & barge-in** — pure numeric logic over audio energy frames:
  end-of-utterance after a silence timeout, and barge-in detection from a
  sustained voiced run while the system is speaking — `core/endpointing.py`.

The pretrained pieces (Whisper ASR, pyttsx3 TTS) are **pluggable** components
behind Protocols in `core/interfaces.py` and are **mocked** in tests, so the whole
service runs with no models. See [EXPLAINER.md](EXPLAINER.md).

## Why it's here

A voice assistant is mostly glue around two big models — until you look at the
*orchestration*: turning text into an intent, knowing which fields are still
missing, deciding when the user stopped talking, and handling interruptions. That
orchestration layer is what this project implements by hand; the ASR/TTS models
are interchangeable backends.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload          # http://localhost:8000

# one text turn
curl -s -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"text": "what is the weather in london"}'
# -> {"intent":"check_weather","response":"Here is the weather for london.", ...}

# a turn that needs a follow-up slot
curl -s -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"text": "book a flight to paris", "session_id": "demo"}'
# -> {"state":"AWAITING_SLOT","missing_slots":["date"], "response":"What date should I use?", ...}
```

Audio turns work the same way — send `{"audio_b64": "..."}`; the default MockASR
decodes UTF-8 audio so you can test the pipeline without Whisper.

## API

- `POST /turn` — body `{text? , audio_b64?, session_id?}` (one of text/audio
  required). Runs ASR (if audio) → intent → slots → dialog. Returns
  `{transcript, intent, slots, missing_slots, state, response, session_id}`.
- `WS /ws/turn` — send JSON turns; receive streamed `stage` events followed by a
  `result` event with the same fields as `/turn`. Keeps session state across
  messages.
- `GET /health` — `{"status":"ok"}`.
- `GET /` — service info, pipeline stages, and known intents.

## Verify

```bash
pytest -q
# core: intent training + held-out classification, TF-IDF idf/L2 properties,
# dialog FSM follow-up + confirmation flow, endpointing + barge-in, slot extraction.
# api:  /turn text turn, missing-slot flow, validation errors, websocket streaming.
```

## Limitations

- **ASR/TTS are pluggable and mocked.** `WhisperASR` / `Pyttsx3TTS` adapters exist
  in `core/interfaces.py` but import lazily and are optional; tests use the mocks.
- The intent classifier is trained on a small illustrative set; it generalizes to
  nearby phrasings but is not a production NLU model.
- Slot extraction uses a small city gazetteer and regex date/time patterns — clear
  upgrade path to a real NER model behind the same interface.
- Dialog state is in-memory per `session_id`; a real deployment would persist it.
