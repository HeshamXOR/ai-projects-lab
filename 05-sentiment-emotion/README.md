# 😊 Sentiment & Emotion Analyzer

Classify text as **positive / negative / neutral** and detect **fine-grained emotions** (joy, anger, sadness, fear, love, surprise). Works on single text or a whole CSV.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **MLP classifier with hand-derived backprop** (NumPy, no autograd) — `core/mlp.py`
- **Metrics**: confusion matrix, precision/recall/F1, macro-F1 — `core/metrics.py`
- **Calibration**: temperature scaling + Expected Calibration Error — `core/metrics.py`

The pretrained classifiers stay as the production path; the from-scratch MLP proves the fundamentals (it learns XOR). See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## What it does

- **Sentiment** — RoBERTa model tuned for sentiment (`twitter-roberta-base-sentiment-latest`).
- **Emotion** — DistilBERT emotion classifier with a full probability breakdown.
- **Batch mode** — upload a CSV with a `text` column and download scored results.

## Why it's real

Sentiment/emotion analysis drives product reviews, support-ticket triage, brand monitoring, and survey analysis. The batch CSV mode makes it immediately useful on real datasets.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

Runs on **CPU** — these are small models. Pre-filled example makes the preview work on first click.

## How it works (files)

- `analyzer.py` — `analyze()` / `analyze_batch()`, lazy-loaded pipelines, GPU/CPU auto-detect.
- `app.py` — two-tab Gradio UI (single text + batch CSV).

## Extend it

- Add aspect-based sentiment (sentiment per topic).
- Swap in a multilingual model for non-English text.
- Add a bar-chart visualization of the emotion distribution.
