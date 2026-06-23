"""Sentiment + emotion analysis.

Two lightweight transformer classifiers:
  * sentiment — positive / negative / neutral
  * emotion   — joy, anger, sadness, fear, love, surprise (fine-grained)

Models load lazily and cache. CPU-friendly (these are small DistilBERT-class
models). Batch mode runs the same models over a CSV column.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import torch

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"


def _device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def _sentiment():
    from transformers import pipeline

    return pipeline("text-classification", model=SENTIMENT_MODEL, device=_device_index())


@lru_cache(maxsize=1)
def _emotion():
    from transformers import pipeline

    return pipeline(
        "text-classification", model=EMOTION_MODEL, top_k=None, device=_device_index()
    )


@dataclass
class Analysis:
    sentiment: str
    sentiment_score: float
    emotions: Dict[str, float]  # label -> probability, sorted desc


def analyze(text: str) -> Analysis:
    text = (text or "").strip()
    if not text:
        return Analysis("—", 0.0, {})
    s = _sentiment()(text[:512])[0]
    e_raw = _emotion()(text[:512])[0]  # list of {label, score}
    emotions = {d["label"]: round(float(d["score"]), 3) for d in e_raw}
    emotions = dict(sorted(emotions.items(), key=lambda kv: -kv[1]))
    return Analysis(
        sentiment=s["label"].capitalize(),
        sentiment_score=round(float(s["score"]), 3),
        emotions=emotions,
    )


def analyze_batch(texts: List[str]) -> List[Analysis]:
    return [analyze(t) for t in texts]


def device_label() -> str:
    return "Running on **GPU**" if torch.cuda.is_available() else "Running on **CPU** (fine for these small models)"
