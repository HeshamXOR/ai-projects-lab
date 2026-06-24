"""Connected-components labeling — from scratch.

Given a binary mask, label each connected blob with a unique id (4-connectivity)
via iterative flood fill. This is the building block for counting objects and
turning a thresholded image into discrete segments — implemented with an
explicit stack, no scipy.label.
"""

from __future__ import annotations

import numpy as np


def connected_components(mask: np.ndarray) -> np.ndarray:
    """Label connected True-regions of a boolean mask. Returns int label map
    (0 = background, 1..n = components)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    current = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                current += 1
                stack = [(i, j)]
                labels[i, j] = current
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels


def count_objects(mask: np.ndarray, min_size: int = 1) -> int:
    labels = connected_components(mask)
    if labels.max() == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]  # skip background
    return int((sizes >= min_size).sum())
