# EXPLAINER — speech-agent: the orchestration layer, by hand

## What I implemented from scratch

- **TF-IDF + logistic-regression intent classifier** (`core/intent.py`)
- **Slot filling** with per-intent schemas (`core/slots.py`)
- **Dialog finite state machine** (`core/dialog.py`)
- **Endpointing & barge-in** over an energy stream (`core/endpointing.py`)

ASR (Whisper) and TTS (pyttsx3) are *optional injected backends* behind Protocols
(`core/interfaces.py`); the orchestration above is the project.

## Intent classifier

### TF-IDF (`TfidfVectorizer`)
1. **Vocabulary** — every distinct token across the corpus gets an index.
2. **Term frequency** — raw count of each vocab term in a document.
3. **Inverse document frequency** — smoothed, sklearn convention:
   `idf(t) = log((1 + N) / (1 + df(t))) + 1`, where `N` is the document count and
   `df(t)` the number of documents containing `t`. Rare terms get larger weights;
   the `+1` smoothing keeps a term that appears everywhere from going to zero.
4. **Weighting + normalization** — multiply `tf * idf`, then divide each row by
   its L2 norm so document length doesn't dominate the dot products.

### Logistic regression (`LogisticRegression`)
Multinomial (softmax) classifier with weights `W ∈ ℝ^{F×C}` and bias `b ∈ ℝ^C`.

- **Forward**: `logits = X·W + b`, `P = softmax(logits)` (max-subtracted for
  numerical stability).
- **Loss**: mean cross-entropy `−(1/n) Σ log P[i, y_i]`, plus an L2 penalty.
- **Gradient**: the clean result for softmax + cross-entropy is
  `∂L/∂logits = (P − Y) / n` with one-hot targets `Y`. Then
  `∂L/∂W = Xᵀ·(P−Y)/n + λW` and `∂L/∂b = Σ(P−Y)/n`.
- **Update**: `W ← W − η·∂L/∂W`, `b ← b − η·∂L/∂b`, repeated for `n_iters`.

A `sigmoid` is also provided for the binary / one-vs-rest framing, but the
multiclass case is handled directly by softmax. `IntentClassifier` ties the
vectorizer and model together and maps integer class ids back to string labels.

## Slot filling (`core/slots.py`)

Each intent has an `IntentSchema` listing `required` and `optional` slots and the
extractor for each. Extractors are deterministic rules:
- **date** — ISO (`2026-06-25`), numeric (`12/05`), `month day` / `day month`, and
  relative words (`today`, `tomorrow`, weekdays).
- **time** — `14:30`, `3pm`, `7am`.
- **number** — digits or number words (`one`…`twelve`).
- **city** — a small gazetteer, multi-word cities matched first.

`fill_slots(intent, text, known=...)` merges newly extracted slots with ones
carried from earlier turns and returns the still-missing required slots — exactly
the signal the FSM needs.

## Dialog FSM (`core/dialog.py`)

States and the legal transitions (transition table in `DialogManager.TRANSITIONS`):

```
IDLE ──────────────► LISTENING
LISTENING ─────────► AWAITING_SLOT | CONFIRMING | RESPONDING | DONE
AWAITING_SLOT ─────► AWAITING_SLOT | CONFIRMING | RESPONDING
CONFIRMING ────────► RESPONDING | LISTENING (cancel) | AWAITING_SLOT
RESPONDING ────────► LISTENING | DONE
DONE ──────────────► LISTENING
```

Per turn, `handle(intent, text)`:
1. From IDLE/DONE, enter LISTENING.
2. If a confirmation is pending, interpret yes/no.
3. Otherwise extract slots (carrying context from the active frame).
4. **Missing required slot** → `AWAITING_SLOT`, ask the slot's follow-up prompt.
5. **All slots filled** → if the intent needs confirmation (`book_flight`) go to
   `CONFIRMING`; else go straight to `RESPONDING` with the filled template.
6. After responding, return to LISTENING and clear the frame for the next turn.

Context (`intent`, accumulated `slots`, `pending_confirmation`) lives per session
so a follow-up that supplies only the missing value completes the frame.

## Endpointing & barge-in (`core/endpointing.py`)

Input is a stream of `Frame(timestamp, energy)`. A frame is **voiced** when its
energy ≥ `energy_threshold`.

- **Endpointing** — once at least `min_speech_frames` voiced frames have been
  seen (speech has started), the detector watches trailing silence. When
  `now − last_voiced_ts ≥ silence_timeout`, the utterance has ended. Requiring
  speech-first prevents an initial pause from triggering an end.
- **Barge-in** — `detect_barge_in` scans for the first run of
  `barge_in_min_frames` *consecutive* voiced frames. Requiring a sustained run
  rejects single-frame noise spikes (flapping), so the system only stops playback
  on a genuine interruption.

Default thresholds: `energy_threshold=0.02`, `silence_timeout=0.8s`,
`min_speech_frames=3`, `barge_in_min_frames=3` — all tunable via `EndpointConfig`.

## Proof it works

`tests/test_core.py` trains the classifier and checks held-out phrasings, verifies
TF-IDF idf ordering and L2 normalization, drives the FSM through a missing-slot →
follow-up → confirm → respond flow, and confirms endpointing fires only after
enough silence while barge-in needs a sustained voiced run. `tests/test_api.py`
exercises `/turn`, the missing-slot response, validation errors, and the
`/ws/turn` streaming events.
