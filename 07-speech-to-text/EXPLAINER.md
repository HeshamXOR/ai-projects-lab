# EXPLAINER — Speech-to-Text: the audio front-end from scratch

## What I implemented from scratch

- The **mel-spectrogram pipeline** that turns a waveform into the features a speech model consumes: pre-emphasis → framing → Hann window → power spectrum → mel filterbank → log (`core/dsp.py`).
- **Voice Activity Detection** (energy + zero-crossing rate) to find speech regions (`core/vad.py`).

Whisper still does the actual transcription; this builds (and visualizes) the signal processing that normally happens invisibly inside it.

## How the pipeline works

1. **Pre-emphasis** — `y[n] = x[n] − 0.97·x[n−1]` boosts high frequencies that carry consonant information.
2. **Framing** — speech is non-stationary, so we cut it into ~25 ms overlapping frames and analyze each as if locally steady.
3. **Hann window** — multiply each frame by `0.5 − 0.5·cos(2πk/(N−1))` to taper the edges; without it, the FFT of a chopped frame "leaks" energy across frequencies.
4. **Power spectrum** — `|FFT|²` per frame gives energy at each frequency.
5. **Mel filterbank** — triangular filters spaced on the **mel scale** (linear below ~1 kHz, log above) collapse the spectrum into bands that match human pitch perception. I build the triangles by hand from the Hz↔mel formulas.
6. **Log** — compress dynamic range (we hear loudness logarithmically).

The result is the log-mel spectrogram — the same representation Whisper/wav2vec ingest.

## VAD

Two cheap, classic features: **short-time energy** (speech is louder than silence) and **zero-crossing rate** (separates voiced speech from noise). An adaptive energy threshold marks speech frames, which are merged into segments — useful for trimming silence before transcription.

## Proof it works

`tests/test_core.py`:
- Hz↔mel conversions round-trip; the Hann window is zero at its edges and ~1 in the middle.
- A synthesized **1 kHz tone** lights up the expected mel band — the spectrogram is actually measuring frequency.
- VAD locates a tone burst sandwiched in silence at the right time.

## Limitations

- Uses `numpy.fft` for the transform itself (a from-scratch radix-2 FFT is a separate exercise); every other stage is hand-built.
- VAD is energy-based, so it can be fooled by loud non-speech noise — a model-based VAD would be more robust.
