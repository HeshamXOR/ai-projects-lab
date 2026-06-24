"""Grad-CAM — from scratch.

Grad-CAM (Selvaraju et al., 2017) explains a CNN/transformer decision by
weighting its feature-map activations by the gradient of the target output with
respect to those activations. Channels whose activations most increase the
target score get the most weight; the weighted, ReLU'd sum is a coarse heatmap
over the input showing "what the model looked at."

The math (independent of framework):
    weights_c = global_average_pool( d(score) / d(A_c) )      # per channel
    cam = ReLU( Σ_c  weights_c * A_c )

This module implements that math over activation/gradient arrays so it can be
unit-tested without a model; the app wires in real hooks on BLIP's vision tower.
"""

from __future__ import annotations

import numpy as np


def grad_cam(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Compute a Grad-CAM heatmap.

    activations, gradients: (C, H, W) — feature maps and the gradient of the
        target score w.r.t. those feature maps.
    returns: (H, W) heatmap normalized to [0, 1].
    """
    assert activations.shape == gradients.shape
    # 1) channel weights = global-average-pooled gradients
    weights = gradients.mean(axis=(1, 2))            # (C,)
    # 2) weighted combination of activation maps
    cam = np.tensordot(weights, activations, axes=([0], [0]))  # (H, W)
    # 3) ReLU — we only care about features that *increase* the target score
    cam = np.maximum(cam, 0)
    # 4) normalize for display
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def upscale_nearest(cam: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor upscaling of a small CAM to image size — from scratch."""
    h, w = cam.shape
    ys = (np.arange(out_h) * h // max(out_h, 1)).clip(0, h - 1)
    xs = (np.arange(out_w) * w // max(out_w, 1)).clip(0, w - 1)
    return cam[ys][:, xs]
