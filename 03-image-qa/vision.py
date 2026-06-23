"""Multimodal image understanding with BLIP.

Two capabilities:
  * caption(image)            — describe an image (BLIP captioning model)
  * answer(image, question)   — answer a question about an image (BLIP VQA model)

Models are loaded lazily on first use (so the app starts instantly and only
pays the download/load cost when you actually use a feature) and placed on GPU
if one is available, else CPU.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
VQA_MODEL = "Salesforce/blip-vqa-base"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _caption_pipeline():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(_device())
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _vqa_pipeline():
    from transformers import BlipForQuestionAnswering, BlipProcessor

    processor = BlipProcessor.from_pretrained(VQA_MODEL)
    model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL).to(_device())
    model.eval()
    return processor, model


def caption(image: Image.Image) -> str:
    if image is None:
        return "Upload an image first."
    processor, model = _caption_pipeline()
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(_device())
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True).strip().capitalize()


def answer(image: Image.Image, question: str) -> str:
    if image is None:
        return "Upload an image first."
    if not question.strip():
        return "Ask a question about the image."
    processor, model = _vqa_pipeline()
    image = image.convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(_device())
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def device_label() -> str:
    return f"Running on **{_device().upper()}**" + (
        "" if _device() == "cuda" else " (CPU — first run downloads the model; answers take a few seconds)"
    )
