"""K-means image segmentation — from scratch.

Cluster an image's pixels in (R,G,B) — or (R,G,B,x,y) — space into k groups and
recolor each pixel by its cluster mean. This is the classic unsupervised
segmentation: no labels, no model, just Lloyd's algorithm implemented directly
(init → assign → update → repeat).
"""

from __future__ import annotations

import numpy as np


def kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0):
    """Lloyd's algorithm. X: (n_samples, n_features). Returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    # k-means++-ish init: pick the first center at random, then spread out
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / max(d2.sum(), 1e-12)
        centers.append(X[rng.choice(len(X), p=probs)])
    centers = np.array(centers, dtype=np.float64)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        # assign each point to the nearest center
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        # update centers to cluster means
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels, centers


def segment_image(image: np.ndarray, k: int = 5, use_xy: bool = True, seed: int = 0):
    """Segment an (H,W,3) uint8 image into k color regions.

    Returns (segmented_image_uint8, label_map). Including (x,y) in the feature
    vector biases clusters toward spatially-compact regions.
    """
    h, w = image.shape[:2]
    rgb = image.reshape(-1, 3).astype(np.float64) / 255.0
    if use_xy:
        ys, xs = np.mgrid[0:h, 0:w]
        coords = np.stack([xs.ravel() / w, ys.ravel() / h], axis=1)
        X = np.concatenate([rgb, 0.5 * coords], axis=1)  # weight color > position
    else:
        X = rgb
    labels, centers = kmeans(X, k, seed=seed)
    # recolor by the RGB part of each center
    seg = (centers[labels][:, :3] * 255).clip(0, 255).astype(np.uint8)
    return seg.reshape(h, w, 3), labels.reshape(h, w)
