"""Audio DSP front-end from scratch: framing, windowing, STFT, mel-spectrogram.

This is the signal-processing pipeline that turns a raw waveform into the
mel-spectrogram features a speech model (like Whisper) actually consumes. Each
step is implemented directly:

  waveform → pre-emphasis → frame → Hann window → FFT (power spectrum)
           → mel filterbank → log → log-mel spectrogram

Only numpy.fft is used for the FFT itself (writing a fast FFT from scratch is a
separate exercise); everything else — framing, the window, the mel filter
construction, the dB conversion — is hand-built.
"""

from __future__ import annotations

import numpy as np


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Boost high frequencies: y[n] = x[n] - coeff*x[n-1]. Standard speech step."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice the signal into overlapping frames -> (n_frames, frame_len)."""
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    n_frames = 1 + (len(signal) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return signal[idx]


def hann_window(n: int) -> np.ndarray:
    """Hann window, written out: 0.5 - 0.5*cos(2πk/(n-1)). Reduces spectral leakage."""
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """|FFT|^2 of each frame -> (n_frames, n_fft//2 + 1)."""
    spec = np.fft.rfft(frames, n=n_fft)
    return (np.abs(spec) ** 2) / n_fft


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, n_fft: int, sr: int, fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """Triangular mel filters -> (n_filters, n_fft//2 + 1).

    The mel scale spaces filters by perceived pitch (linear below ~1kHz,
    logarithmic above), matching human hearing — which is why speech models use
    it instead of a raw spectrum.
    """
    fmax = fmax or sr / 2
    # equally-spaced points on the mel scale, mapped back to Hz
    mel_pts = np.linspace(hz_to_mel(np.array([fmin])), hz_to_mel(np.array([fmax])), n_filters + 2).ravel()
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        for j in range(left, center):
            if center > left:
                fb[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i - 1, j] = (right - j) / (right - center)
    return fb


def log_mel_spectrogram(
    signal: np.ndarray, sr: int = 16000, n_fft: int = 512, hop: int = 160,
    frame_len: int = 400, n_mels: int = 40,
) -> np.ndarray:
    """The full pipeline: waveform -> log-mel spectrogram (n_mels, n_frames)."""
    sig = pre_emphasis(signal.astype(np.float64))
    frames = frame_signal(sig, frame_len, hop)
    frames = frames * hann_window(frame_len)[None, :]
    pow_spec = power_spectrum(frames, n_fft)            # (n_frames, n_fft//2+1)
    fb = mel_filterbank(n_mels, n_fft, sr)              # (n_mels, n_fft//2+1)
    mel = pow_spec @ fb.T                               # (n_frames, n_mels)
    log_mel = 10 * np.log10(mel + 1e-10)
    return log_mel.T                                    # (n_mels, n_frames)
