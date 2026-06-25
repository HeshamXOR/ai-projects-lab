"""Threshold segmentation: global, Otsu, and adaptive -- all from scratch.

WHY THIS MODULE EXISTS
----------------------
Once we have a defect-salience map (the local-contrast z-score from
``features.py``), we must binarize it into a defect mask. Three strategies
are provided because real inspection lines need all three:

* **Global threshold** -- fastest, used when lighting is controlled.
* **Otsu's method** -- chooses the global threshold automatically by maximizing
  between-class variance of a 1D histogram. Implemented here from the
  histogram up (no ``skimage.filters.threshold_otsu``).
* **Adaptive (local-mean) threshold** -- compares each pixel to the mean of its
  neighborhood minus an offset, so it survives illumination gradients. Reuses
  the integral-image machinery from ``features.py``.

We segment on the *absolute* contrast by default, because defects can be
either darker or brighter than the surface, but the sign is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .features import FeatureConfig, integral_image, _window_sums

__all__ = [
    "SegmentationConfig",
    "otsu_threshold",
    "global_threshold",
    "adaptive_threshold",
    "segment",
]

Polarity = Literal["abs", "bright", "dark"]


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters controlling how a salience map is binarized.

    Attributes
    ----------
    method:
        ``"otsu"``, ``"global"`` or ``"adaptive"``.
    threshold:
        Used by the ``"global"`` method (z-score units).
    block:
        Neighborhood side length (odd) for the ``"adaptive"`` method.
    offset:
        Subtracted from the local mean in the ``"adaptive"`` method; raising it
        makes detection stricter.
    polarity:
        Whether defects are darker (``"dark"``), brighter (``"bright"``), or
        either (``"abs"``) than their surroundings.
    nbins:
        Histogram resolution for Otsu.
    """

    method: Literal["otsu", "global", "adaptive"] = "otsu"
    threshold: float = 3.0
    block: int = 31
    offset: float = 0.5
    polarity: Polarity = "abs"
    nbins: int = 256

    def __post_init__(self) -> None:
        if self.method not in ("otsu", "global", "adaptive"):
            raise ValueError(f"unknown method {self.method!r}")
        if self.block < 1 or self.block % 2 == 0:
            raise ValueError("block must be a positive odd integer")
        if self.nbins < 2:
            raise ValueError("nbins must be >= 2")


def _apply_polarity(salience: np.ndarray, polarity: Polarity) -> np.ndarray:
    """Reduce a signed salience map to a non-negative defect strength map."""
    if polarity == "abs":
        return np.abs(salience)
    if polarity == "bright":
        return np.clip(salience, 0.0, None)
    if polarity == "dark":
        return np.clip(-salience, 0.0, None)
    raise ValueError(f"unknown polarity {polarity!r}")


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Compute Otsu's optimal threshold on a 1D set of values.

    Otsu picks the threshold ``t`` that maximizes the between-class variance::

        sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2

    where ``w0,w1`` are the mass of the two classes split at ``t`` and
    ``mu0,mu1`` their means. Maximizing between-class variance is equivalent to
    minimizing within-class variance but is cheaper to evaluate from a
    histogram via cumulative sums. We compute all candidate thresholds at once
    with vectorized prefix sums.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    vmin = float(flat.min())
    vmax = float(flat.max())
    if vmax <= vmin:
        return vmin  # degenerate flat input

    hist, edges = np.histogram(flat, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    total = hist.sum()
    w0 = np.cumsum(hist)                  # mass at or below each bin
    w1 = total - w0                       # mass above
    # Cumulative weighted sums for the class means.
    csum = np.cumsum(hist * centers)
    grand = csum[-1]

    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.zeros_like(w0)
    mu1 = np.zeros_like(w1)
    mu0[valid] = csum[valid] / w0[valid]
    mu1[valid] = (grand - csum[valid]) / w1[valid]

    between = np.zeros_like(w0)
    between[valid] = w0[valid] * w1[valid] * (mu0[valid] - mu1[valid]) ** 2

    best = int(np.argmax(between))
    return float(centers[best])


def global_threshold(salience: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a salience map with a single global cut-off."""
    return (np.asarray(salience) >= threshold)


def adaptive_threshold(
    strength: np.ndarray, block: int, offset: float
) -> np.ndarray:
    """Local-mean adaptive threshold using an integral image.

    A pixel is foreground when ``strength > local_mean(block) + offset``. The
    local mean is computed over a clipped ``block x block`` window in O(N) via
    the summed-area table, mirroring ``features.local_mean_std``.
    """
    strength = np.asarray(strength, dtype=np.float64)
    radius = block // 2
    sat = integral_image(strength)
    sums, counts = _window_sums(sat, strength.shape, radius)
    local_mean = sums / counts
    return strength > (local_mean + offset)


def segment(salience: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Binarize a signed salience map into a boolean defect mask.

    The polarity is applied first to turn the signed z-score into a
    non-negative defect strength, then the configured thresholding method runs.
    Returns a boolean array (True = defect pixel).
    """
    strength = _apply_polarity(np.asarray(salience, dtype=np.float64),
                               config.polarity)
    if config.method == "global":
        return global_threshold(strength, config.threshold)
    if config.method == "otsu":
        t = otsu_threshold(strength, config.nbins)
        return strength >= t
    if config.method == "adaptive":
        return adaptive_threshold(strength, config.block, config.offset)
    raise ValueError(f"unknown method {config.method!r}")
