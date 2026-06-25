"""ASR and TTS interfaces with mock and optional real adapters.

The pipeline depends only on the ``ASR`` and ``TTS`` Protocols, so the heavy
pretrained models are pluggable. Tests use the deterministic mocks; production
can drop in the Whisper / pyttsx3 adapters, which import lazily and are *not*
required to be installed.
"""

from __future__ import annotations

import base64
from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ASR(Protocol):
    """Automatic speech recognition: audio bytes -> transcript text."""

    def transcribe(self, audio: bytes) -> str:
        """Return the transcript of ``audio``."""
        ...


@runtime_checkable
class TTS(Protocol):
    """Text to speech: text -> audio bytes."""

    def synthesize(self, text: str) -> bytes:
        """Return synthesized audio for ``text``."""
        ...


class MockASR:
    """Deterministic ASR for tests and offline runs.

    Decodes audio bytes that were produced as UTF-8 (or base64-of-UTF-8) text so
    tests can round-trip a known transcript without a model. A lookup table of
    canned phrases can also be supplied.
    """

    def __init__(self, table: Optional[Dict[bytes, str]] = None) -> None:
        self.table = table or {}

    def transcribe(self, audio: bytes) -> str:
        """Return a transcript for ``audio`` (table lookup, then decode)."""
        if audio in self.table:
            return self.table[audio]
        # try plain utf-8
        try:
            return audio.decode("utf-8")
        except UnicodeDecodeError:
            pass
        # try base64-encoded utf-8
        try:
            return base64.b64decode(audio).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return ""


class MockTTS:
    """Deterministic TTS for tests: encodes the text as UTF-8 bytes."""

    def synthesize(self, text: str) -> bytes:
        """Return ``text`` encoded as bytes (round-trips with MockASR)."""
        return text.encode("utf-8")


class WhisperASR:
    """Optional real ASR backed by openai-whisper. Imports lazily.

    Not required to be installed; constructing this raises a clear error if the
    package is missing, so the rest of the app keeps importing fine.
    """

    def __init__(self, model_name: str = "base") -> None:
        try:
            import whisper  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only with the dep
            raise ImportError(
                "openai-whisper is not installed; install it to use WhisperASR, "
                "or use MockASR for offline runs."
            ) from exc
        self._whisper = whisper
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio: bytes) -> str:  # pragma: no cover - needs model
        """Transcribe raw audio bytes via Whisper (writes a temp file)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(audio)
            path = fh.name
        result = self.model.transcribe(path)
        return str(result.get("text", "")).strip()


class Pyttsx3TTS:
    """Optional real TTS backed by pyttsx3. Imports lazily."""

    def __init__(self) -> None:
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only with the dep
            raise ImportError(
                "pyttsx3 is not installed; install it to use Pyttsx3TTS, "
                "or use MockTTS for offline runs."
            ) from exc
        self.engine = pyttsx3.init()

    def synthesize(self, text: str) -> bytes:  # pragma: no cover - needs engine
        """Render ``text`` to a wav file and return its bytes."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            path = fh.name
        self.engine.save_to_file(text, path)
        self.engine.runAndWait()
        with open(path, "rb") as fh:
            return fh.read()
