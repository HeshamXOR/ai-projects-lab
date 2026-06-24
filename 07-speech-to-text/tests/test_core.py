"""Proofs for the from-scratch DSP front-end and VAD."""

import numpy as np

from core.dsp import (
    hz_to_mel, mel_to_hz, hann_window, frame_signal, log_mel_spectrogram, mel_filterbank,
)
from core.vad import detect_speech, short_time_energy


def test_mel_hz_roundtrip():
    f = np.array([100.0, 440.0, 1000.0, 4000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(f)), f, rtol=1e-6)


def test_hann_window_endpoints():
    w = hann_window(400)
    assert abs(w[0]) < 1e-9 and abs(w[-1]) < 1e-9   # zero at the edges
    assert abs(w[len(w) // 2] - 1.0) < 1e-2          # ~1 in the middle


def test_framing_shape():
    sig = np.arange(1000.0)
    frames = frame_signal(sig, frame_len=400, hop=160)
    assert frames.shape[1] == 400
    assert frames.shape[0] == 1 + (1000 - 400) // 160


def test_mel_filterbank_shape_and_nonneg():
    fb = mel_filterbank(40, 512, 16000)
    assert fb.shape == (40, 512 // 2 + 1)
    assert (fb >= 0).all()


def test_spectrogram_detects_tone_frequency():
    # a 1 kHz sine should put energy in the mel band covering 1 kHz
    sr = 16000
    t = np.arange(sr) / sr
    sig = np.sin(2 * np.pi * 1000 * t)
    spec = log_mel_spectrogram(sig, sr=sr, n_mels=40)
    assert spec.shape[0] == 40
    # the loudest mel band should be in the lower-middle of the range (1 kHz)
    loud_band = np.argmax(spec.mean(axis=1))
    assert 3 < loud_band < 30


def test_vad_finds_speech_region():
    sr = 16000
    silence = np.zeros(sr // 2)
    tone = 0.5 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)  # 1s "speech"
    sig = np.concatenate([silence, tone, silence])
    segs = detect_speech(sig, sr=sr)
    assert len(segs) >= 1
    # the detected speech should start after the initial silence (~0.5s)
    assert segs[0][0] > 0.2
