# 🎙️ Speech-to-Text + Summary

Upload or record audio → get an accurate **transcript** (OpenAI Whisper) and, for longer recordings, an automatic **summary**. Built for meetings, lectures, interviews, and voice notes.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What it does

- **Transcription** — `whisper-base` via the transformers ASR pipeline, with 30-second chunking so long audio works.
- **Summary** — condenses long transcripts with `distilbart-cnn` (skipped automatically for short clips).
- **Record or upload** — the Gradio audio component supports both.

## Why it's real

Audio → text → summary is the backbone of meeting-notes tools, podcast indexing, and accessibility captioning. End-to-end audio understanding is a strong, demoable capability.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

- **GPU (L4) recommended** — Whisper is much faster on GPU. CPU works for short clips (first run downloads the model).
- Audio decoding needs `ffmpeg`. On Lightning it's usually present; if not: `apt-get install -y ffmpeg`.

## How it works (files)

- `transcribe.py` — `transcribe()` runs Whisper, then optionally summarizes. Lazy-loaded, GPU/CPU auto-detect.
- `app.py` — Gradio UI: audio in → transcript + summary out.

## Extend it

- Swap `whisper-base` → `whisper-small`/`medium` on a bigger GPU for higher accuracy.
- Add speaker diarization (who spoke when).
- Add timestamped segments and export to SRT subtitles.
