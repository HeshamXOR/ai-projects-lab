"""Long-text summarization with action-item extraction.

Uses a DistilBART summarization model. Long inputs are split into token-bounded
chunks, each chunk summarized, then the chunk summaries are summarized again
(map-reduce) so arbitrarily long articles/transcripts fit. Action items are
pulled out heuristically (imperative / commitment phrasing) — handy for meeting
notes. GPU if available, else CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import torch

MODEL = "sshleifer/distilbart-cnn-12-6"

# Cues that a sentence is an action item / commitment.
ACTION_CUES = [
    "will ", "need to", "needs to", "should ", "must ", "let's ", "lets ",
    "action item", "to do", "todo", "follow up", "follow-up", "by next",
    "i'll ", "we'll ", "assign", "responsible for", "deadline", "due ",
]


@dataclass
class Summary:
    summary: str
    action_items: List[str]
    compression: str


def _device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def _summarizer():
    from transformers import pipeline

    return pipeline("summarization", model=MODEL, device=_device_index())


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _chunk(text: str, max_words: int = 700) -> List[str]:
    """Split text into word-bounded chunks at sentence boundaries."""
    sentences = _split_sentences(text)
    chunks, current, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if count + w > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += w
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _summarize_one(text: str, max_len: int = 130, min_len: int = 30) -> str:
    summ = _summarizer()
    # The model has a ~1024 token limit; word chunking keeps us safely under it.
    out = summ(text, max_length=max_len, min_length=min_len, do_sample=False)
    return out[0]["summary_text"].strip()


def extract_action_items(text: str) -> List[str]:
    items = []
    for s in _split_sentences(text):
        low = s.lower()
        if any(cue in low for cue in ACTION_CUES):
            items.append(s)
    # de-dup while preserving order, cap to keep it tidy
    seen, unique = set(), []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique[:8]


def summarize(text: str) -> Summary:
    text = text.strip()
    if not text:
        return Summary("Paste some text to summarize.", [], "")
    if len(text.split()) < 40:
        return Summary(text, extract_action_items(text), "input too short to compress")

    chunks = _chunk(text)
    partials = [_summarize_one(c) for c in chunks]
    combined = " ".join(partials)

    # Map-reduce: if there were multiple chunks, summarize the combination again.
    if len(chunks) > 1 and len(combined.split()) > 60:
        final = _summarize_one(combined, max_len=160, min_len=40)
    else:
        final = combined

    in_words = len(text.split())
    out_words = len(final.split())
    ratio = f"{in_words} → {out_words} words ({100 * out_words / in_words:.0f}% of original)"
    return Summary(final, extract_action_items(text), ratio)


def device_label() -> str:
    return "Running on **GPU**" if torch.cuda.is_available() else "Running on **CPU** (works fine; long inputs take a little longer)"
