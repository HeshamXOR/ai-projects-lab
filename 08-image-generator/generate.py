"""Text-to-image generation with Stable Diffusion (diffusers).

Generates images from a text prompt. This is the one project that genuinely
wants a GPU: SD on CPU is very slow (minutes per image). The code runs anywhere,
but the UI warns clearly when no GPU is present.

Defaults to SD 1.5 (small, fast, widely available). On a bigger GPU you can
switch MODEL_ID to an SDXL checkpoint.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def has_gpu() -> bool:
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def _pipe():
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if has_gpu() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to("cuda" if has_gpu() else "cpu")
    # Memory-friendly attention on GPU.
    if has_gpu():
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe


def generate(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    guidance: float = 7.5,
    seed: Optional[int] = None,
):
    if not prompt.strip():
        return None, "Enter a prompt."
    pipe = _pipe()
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda" if has_gpu() else "cpu").manual_seed(int(seed))
    # On CPU, force few steps so a demo image still appears in reasonable time.
    if not has_gpu():
        steps = min(steps, 12)
    result = pipe(
        prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    )
    note = "" if has_gpu() else " (CPU mode — reduced steps; use a GPU for quality + speed)"
    return result.images[0], f"Done{note}."


def device_label() -> str:
    if has_gpu():
        return "Running on **GPU** — generation takes a few seconds per image."
    return (
        "⚠️ **No GPU detected.** Stable Diffusion runs on CPU but is *slow* "
        "(can take a minute+ per image, with reduced quality). Use an **L4** "
        "Studio on Lightning for a smooth demo."
    )
