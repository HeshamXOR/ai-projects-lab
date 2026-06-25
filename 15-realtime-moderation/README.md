# realtime-moderation

A streaming content-moderation API that flags text and **explains why** it was flagged — combining a configurable policy rule engine with a from-scratch classifier.

## What I implemented from scratch

- **Multinomial Naive Bayes** text classifier (`core/naive_bayes.py`) — full log-space implementation with Laplace smoothing, trained on a bundled labeled dataset. No scikit-learn, no NumPy: only the standard library.
- **A tokenizer** (`core/tokenizer.py`) — normalization, casefolding, and n-gram features for the classifier.
- **A policy rule-DSL** (`core/policy.py`, `core/rules.py`) — declarative rules for categories (toxicity, PII, spam, self-harm) that fire deterministically and combine with the classifier.
- **Severity scoring + explanations** (`core/scoring.py`, `core/explain.py`) — every decision carries a severity and a human-readable reason for *why* content was flagged.

## What it does

`POST /moderate` takes text (single or batch) and returns: a decision (allow / flag / block), the categories triggered, a severity score, and an explanation tracing which rules and which classifier signal drove the call. The policy engine and the learned classifier are blended so precise rules (exact PII patterns) and fuzzy signals (learned toxicity) both contribute.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload          # http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t realtime-moderation . && docker run -p 8000:8000 realtime-moderation
```

## API

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/moderate` | `{ "text": "...", }` or `{ "texts": [...] }` | decision, categories, severity, explanation |
| `GET`  | `/health`   | — | service status |

## Verify

```bash
pytest -q     # proves the Naive Bayes learns, the tokenizer is stable, and rules fire
```

## How it works

See [EXPLAINER.md](EXPLAINER.md) for the Naive Bayes derivation, the smoothing math, and the policy-vs-classifier blending strategy.

## Limitations

- The bundled training set is small (illustrative); swap in a larger labeled corpus for production recall.
- English-oriented tokenizer; multilingual moderation would need a wider feature set.
