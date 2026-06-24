"""Region growing — interactive click-to-segment, from scratch.

Starting from a seed pixel (where the user clicks), flood outward to neighboring
pixels whose color is within a threshold of the growing region's mean. A
breadth-first traversal over the pixel grid — the classic interactive
segmentation primitive, implemented with an explicit queue (no scikit-image).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def region_grow(image: np.ndarray, seed_xy, threshold: float = 25.0) -> np.ndarray:
    """Grow a region from seed (x, y). Returns a boolean mask (H, W).

    `threshold` is the max Euclidean color distance (0-255 scale) a pixel may be
    from the region's running mean color to be included.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float64)
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    sx = min(max(sx, 0), w - 1)
    sy = min(max(sy, 0), h - 1)

    mask = np.zeros((h, w), dtype=bool)
    mask[sy, sx] = True
    region_sum = img[sy, sx].copy()
    region_count = 1

    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        mean = region_sum / region_count
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx]:
                dist = np.linalg.norm(img[ny, nx] - mean)
                if dist <= threshold:
                    mask[ny, nx] = True
                    region_sum += img[ny, nx]
                    region_count += 1
                    q.append((nx, ny))
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(124, 92, 255), alpha=0.5) -> np.ndarray:
    """Blend a colored highlight over the masked region for display."""
    out = image.copy().astype(np.float64)
    color = np.array(color, dtype=np.float64)
    out[mask] = (1 - alpha) * out[mask] + alpha * color
    return out.clip(0, 255).astype(np.uint8)
