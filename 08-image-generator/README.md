# 🎨 AI Image Generator

Generate images from text prompts with **Stable Diffusion**. Control steps, guidance scale, negative prompts, and seed for reproducible results.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **Diffusion math**: noise schedule, forward process, **DDPM + DDIM** reverse steps, classifier-free guidance — `core/ddim.py`
- **Toy diffusion model**: a NumPy noise-predictor (hand-written backprop) trained on 2D data, sampled with the DDIM loop — `core/toy.py`

The app's second tab trains this live so you can watch noise become a shape. Stable Diffusion remains the high-quality image path. See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## What it does

- **Text-to-image** — SD 1.5 via the `diffusers` library.
- **Full controls** — inference steps, guidance scale, negative prompt, and a seed for reproducibility.
- **Hardware-aware** — uses fp16 + attention slicing on GPU; falls back to CPU with reduced steps so a demo image still appears.

## Why it's real

Text-to-image generation powers design tools, marketing asset creation, and concept art. It's the most visually striking project in the lab — a great portfolio centerpiece.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

- **GPU (L4) strongly recommended.** A few seconds per image on GPU.
- **CPU works but is slow** (a minute+ per image, reduced quality) — fine to verify it runs, but record the preview on a GPU Studio.

## How it works (files)

- `generate.py` — loads the SD pipeline (fp16 on GPU), `generate()` with all parameters, GPU/CPU detection.
- `app.py` — Gradio UI with prompt, negative prompt, and sliders.

## Extend it

- Switch `MODEL_ID` in `generate.py` to an **SDXL** checkpoint on a bigger GPU.
- Add image-to-image or inpainting.
- Add a LoRA to specialize the model on a style (ties into fine-tuning).
