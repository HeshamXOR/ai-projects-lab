"""speech-agent core: intent, slots, dialog FSM, endpointing, ASR/TTS interfaces."""

from .intent import IntentClassifier, TfidfVectorizer, LogisticRegression
from .slots import fill_slots, SlotResult, IntentSchema, DEFAULT_SCHEMAS
from .dialog import DialogManager, State, DialogResult
from .endpointing import (
    Endpointer,
    EndpointConfig,
    Frame,
    detect_barge_in,
    frames_from_energies,
    segment_speech,
)
from .interfaces import ASR, TTS, MockASR, MockTTS, WhisperASR, Pyttsx3TTS

__all__ = [
    "IntentClassifier", "TfidfVectorizer", "LogisticRegression",
    "fill_slots", "SlotResult", "IntentSchema", "DEFAULT_SCHEMAS",
    "DialogManager", "State", "DialogResult",
    "Endpointer", "EndpointConfig", "Frame", "detect_barge_in",
    "frames_from_energies", "segment_speech",
    "ASR", "TTS", "MockASR", "MockTTS", "WhisperASR", "Pyttsx3TTS",
]
