"""Endpointing and barge-in logic — pure numeric stream processing.

Given a stream of audio frames (each with an energy value and a timestamp), this
module decides:

- **End of utterance (endpointing)** — the user has stopped speaking once the
  trailing silence exceeds ``silence_timeout`` seconds *after* speech was seen.
- **Barge-in** — the user starts speaking (energy above threshold for a minimum
  run) while the system itself is "speaking", so we should stop playback.

Speech vs. silence is a simple energy gate: a frame is "voiced" when its energy
exceeds ``energy_threshold``. To avoid flapping on a single noisy frame, barge-in
requires ``barge_in_min_frames`` consecutive voiced frames.

From scratch: no VAD library, just thresholds over the numeric stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Frame:
    """One audio analysis frame."""

    timestamp: float  # seconds since stream start
    energy: float     # short-time energy (>= 0)


@dataclass
class EndpointConfig:
    """Tunable thresholds for endpointing and barge-in."""

    energy_threshold: float = 0.02
    silence_timeout: float = 0.8          # seconds of trailing silence -> end
    min_speech_frames: int = 3            # frames of speech before endpointing arms
    barge_in_min_frames: int = 3          # consecutive voiced frames -> barge-in


@dataclass
class EndpointState:
    """Running state of the endpointer as frames arrive."""

    speech_frames: int = 0
    last_voiced_ts: Optional[float] = None
    started_speaking: bool = False
    ended: bool = False
    end_ts: Optional[float] = None


class Endpointer:
    """Streaming endpoint detector over energy frames."""

    def __init__(self, config: Optional[EndpointConfig] = None) -> None:
        self.config = config or EndpointConfig()
        self.state = EndpointState()

    def reset(self) -> None:
        """Clear state for a new utterance."""
        self.state = EndpointState()

    def is_voiced(self, frame: Frame) -> bool:
        """Whether ``frame`` is above the energy gate."""
        return frame.energy >= self.config.energy_threshold

    def update(self, frame: Frame) -> bool:
        """Feed one frame; return True if end-of-utterance is now detected.

        End of utterance requires that speech has been observed (at least
        ``min_speech_frames`` voiced frames) and then silence has persisted for
        ``silence_timeout`` seconds since the last voiced frame.
        """
        cfg = self.config
        st = self.state
        if st.ended:
            return True

        if self.is_voiced(frame):
            st.speech_frames += 1
            st.last_voiced_ts = frame.timestamp
            if st.speech_frames >= cfg.min_speech_frames:
                st.started_speaking = True
        elif st.started_speaking and st.last_voiced_ts is not None:
            silence = frame.timestamp - st.last_voiced_ts
            if silence >= cfg.silence_timeout:
                st.ended = True
                st.end_ts = frame.timestamp
                return True
        return False

    def process(self, frames: Sequence[Frame]) -> Optional[float]:
        """Run a whole frame sequence; return the end timestamp or None."""
        self.reset()
        for frame in frames:
            if self.update(frame):
                return self.state.end_ts
        return None


@dataclass
class BargeInResult:
    """Outcome of a barge-in scan over frames while the system was speaking."""

    detected: bool
    timestamp: Optional[float] = None
    run_frames: List[Frame] = field(default_factory=list)


def detect_barge_in(
    frames: Sequence[Frame],
    config: Optional[EndpointConfig] = None,
) -> BargeInResult:
    """Detect the first sustained voiced run (the user interrupting).

    Returns the timestamp of the frame that completes the first run of
    ``barge_in_min_frames`` consecutive voiced frames. Used while the system is
    playing TTS to decide whether to stop and listen.
    """
    cfg = config or EndpointConfig()
    run: List[Frame] = []
    for frame in frames:
        if frame.energy >= cfg.energy_threshold:
            run.append(frame)
            if len(run) >= cfg.barge_in_min_frames:
                return BargeInResult(True, run[-1].timestamp, list(run))
        else:
            run = []
    return BargeInResult(False)


def frames_from_energies(
    energies: Sequence[float],
    frame_period: float = 0.1,
    start: float = 0.0,
) -> List[Frame]:
    """Helper: build evenly-spaced frames from a list of energy values."""
    return [
        Frame(timestamp=start + i * frame_period, energy=float(e))
        for i, e in enumerate(energies)
    ]


def segment_speech(
    frames: Sequence[Frame],
    config: Optional[EndpointConfig] = None,
) -> List[Tuple[float, float]]:
    """Return (start_ts, end_ts) spans of contiguous voiced frames."""
    cfg = config or EndpointConfig()
    spans: List[Tuple[float, float]] = []
    seg_start: Optional[float] = None
    prev_ts = 0.0
    for frame in frames:
        if frame.energy >= cfg.energy_threshold:
            if seg_start is None:
                seg_start = frame.timestamp
            prev_ts = frame.timestamp
        else:
            if seg_start is not None:
                spans.append((seg_start, prev_ts))
                seg_start = None
    if seg_start is not None:
        spans.append((seg_start, prev_ts))
    return spans
