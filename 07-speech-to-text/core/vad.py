"""Voice Activity Detection from scratch: energy + zero-crossing rate.

Splits audio into speech vs. silence using two classic cheap features:
  * short-time energy — speech is louder than silence,
  * zero-crossing rate — distinguishes voiced speech from noise/fricatives.

Used to trim silence and segment an utterance before transcription. No webrtcvad
— the framing, the features, and the thresholding are all here.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .dsp import frame_signal


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    return np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)


def detect_speech(
    signal: np.ndarray, sr: int = 16000, frame_ms: int = 25, hop_ms: int = 10,
    energy_percentile: float = 60.0,
) -> List[Tuple[float, float]]:
    """Return list of (start_sec, end_sec) speech segments.

    A frame is 'speech' if its energy exceeds an adaptive threshold (a
    percentile of the clip's energy). Adjacent speech frames are merged into
    segments.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames = frame_signal(signal.astype(np.float64), frame_len, hop)
    energy = short_time_energy(frames)
    if energy.max() <= 0:
        return []

    threshold = np.percentile(energy, energy_percentile)
    is_speech = energy > threshold

    segments = []
    start = None
    for i, s in enumerate(is_speech):
        t = i * hop / sr
        if s and start is None:
            start = t
        elif not s and start is not None:
            segments.append((start, t))
            start = None
    if start is not None:
        segments.append((start, len(is_speech) * hop / sr))
    # merge segments separated by < 0.2s
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < 0.2:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return merged
