"""Attention rollout — from scratch.

A Vision/Language Transformer produces an attention matrix per layer (how much
each token attends to each other token). A single layer's attention is a poor
explanation because information mixes across *all* layers. **Attention rollout**
(Abnar & Zuidema, 2020) composes attention across layers to estimate how much
each input token ultimately influences each output position.

The recipe:
  1. Add the identity to each layer's attention (residual connections let a
     token keep its own information): A' = 0.5*A + 0.5*I.
  2. Re-normalize rows to sum to 1.
  3. Multiply the layers together: rollout = A'_L @ ... @ A'_2 @ A'_1.

The result's row for a query position tells you which input tokens it draws
from — for an image transformer, that becomes a saliency map over patches.

Implemented as pure NumPy over attention arrays, so it's independent of any
model and unit-testable.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _add_residual_and_normalize(att: np.ndarray) -> np.ndarray:
    """A' = normalize_rows(0.5*A + 0.5*I)."""
    n = att.shape[-1]
    a = 0.5 * att + 0.5 * np.eye(n)
    a = a / a.sum(axis=-1, keepdims=True)
    return a


def attention_rollout(attentions: List[np.ndarray]) -> np.ndarray:
    """Compose per-layer attention (each (T, T), already head-averaged) into a
    single (T, T) rollout matrix."""
    assert len(attentions) > 0
    result = _add_residual_and_normalize(attentions[0])
    for att in attentions[1:]:
        a = _add_residual_and_normalize(att)
        result = a @ result
    return result


def average_heads(attn_layer: np.ndarray) -> np.ndarray:
    """(n_heads, T, T) -> (T, T) by averaging over heads."""
    return attn_layer.mean(axis=0)


def saliency_from_rollout(rollout: np.ndarray, query_index: int = 0, drop_special: int = 1) -> np.ndarray:
    """Extract the saliency over input tokens for a given query position.

    For a ViT, query_index 0 is typically the [CLS] token; `drop_special`
    removes leading special tokens so what's left maps to image patches.
    """
    row = rollout[query_index]
    return row[drop_special:]


def saliency_to_grid(saliency: np.ndarray) -> np.ndarray:
    """Reshape a per-patch saliency vector to a square grid for display."""
    n = len(saliency)
    side = int(round(np.sqrt(n)))
    side = max(side, 1)
    grid = np.zeros(side * side)
    grid[: min(n, side * side)] = saliency[: side * side]
    grid = grid.reshape(side, side)
    # normalize to [0,1] for an overlay
    if grid.max() > grid.min():
        grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid
