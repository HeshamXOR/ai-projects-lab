# 🖼️ Image Q&A (Multimodal)

Upload an image → get an automatic **caption**, then ask **visual questions** about it. A multimodal app built on the BLIP vision-language model.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **Attention rollout** — composes per-layer transformer attention into a saliency map — `core/attention_rollout.py`
- **Grad-CAM** — gradient-weighted activation heatmaps (where the model looked) — `core/gradcam.py`

BLIP answers the questions; these add explainability you can *see*. See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## What it does

- **Captioning** — describes the image in a sentence (`blip-image-captioning-base`).
- **Visual Q&A** — answers questions about the image content (`blip-vqa-base`), e.g. "What color is the car?", "How many people are there?"

## Why it's real

Multimodal understanding powers accessibility tools (alt-text generation), content moderation, visual search, and product tagging. This shows you can work across vision *and* language — a standout portfolio capability.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

- **GPU (L4) recommended** for snappy responses. The app auto-detects CUDA.
- **CPU works** for a demo — the first run downloads the models and each answer takes a few seconds.

## How it works (files)

- `vision.py` — lazy-loaded BLIP pipelines; `caption()` and `answer()`; auto GPU/CPU placement.
- `app.py` — Gradio UI: upload → caption + visual chat.

## Notes & limitations

- BLIP-base is small and fast but not as capable as large VLMs (LLaVA, etc.). For a stronger demo on a bigger GPU, swap the model id in `vision.py`.
- Models download on first use (a few hundred MB) and are cached afterward.
