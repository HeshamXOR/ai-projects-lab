"""Speech-to-text (Whisper) with an optional summary.

Transcribes an uploaded audio file using OpenAI's Whisper via the HF
transformers ASR pipeline, then optionally condenses long transcripts with a
summarization model. Whisper-base is small enough to run on CPU for a demo;
GPU makes it much faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

ASR_MODEL = "openai/whisper-base"
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"


def _device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def _asr():
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        device=_device_index(),
        chunk_length_s=30,  # enables transcription of long audio
    )


@lru_cache(maxsize=1)
def _summarizer():
    from transformers import pipeline

    return pipeline("summarization", model=SUMMARY_MODEL, device=_device_index())


@dataclass
class TranscriptResult:
    transcript: str
    summary: Optional[str]


def transcribe(audio_path: str, do_summary: bool = True) -> TranscriptResult:
    if not audio_path:
        return TranscriptResult("Upload or record audio first.", None)
    text = _asr()(audio_path)["text"].strip()
    summary = None
    if do_summary and len(text.split()) > 60:
        # cap input length for the summarizer's token window
        chunk = " ".join(text.split()[:700])
        summary = _summarizer()(chunk, max_length=130, min_length=30, do_sample=False)[0][
            "summary_text"
        ].strip()
    return TranscriptResult(text, summary)


def device_label() -> str:
    return (
        "Running on **GPU** — fast transcription."
        if torch.cuda.is_available()
        else "Running on **CPU** — works, but long audio takes a while (first run also downloads the model)."
    )
