"""Local-contrast feature maps in pure NumPy.

WHY THIS MODULE EXISTS
----------------------
Defects are almost always *local contrast* anomalies: a scratch is darker than
its neighborhood, a bright pit reflects more than the surrounding surface.
A globally-thresholded image fails under uneven lighting, so we compute
*local* statistics -- a per-pixel local mean and local standard deviation over
a sliding window -- and express each pixel as how many local standard
deviations it sits away from its neighborhood mean (a local z-score). That
contrast map is far more robust to illumination gradients than raw intensity.

The expensive part is computing a windowed mean/variance at every pixel. Doing
that naively is O(N * k^2). We instead use *integral images* (summed-area
tables) so each window sum is four array lookups, giving O(N) regardless of
window size. We implement the integral image and the windowed moments from
scratch with NumPy cumulative sums -- no OpenCV ``boxFilter``, no
``scipy.ndimage``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FeatureConfig",
    "to_grayscale",
    "integral_image",
    "local_mean_std",
    "local_contrast",
    "gradient_magnitude",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Tunable parameters for local-contrast feature extraction.

    Attributes
    ----------
    window:
        Side length (pixels) of the square neighborhood. Must be odd so the
        window is centered on the pixel.
    eps:
        Numerical floor added to the local std to avoid divide-by-zero in flat
        regions (and to suppress noise amplification there).
    """

    window: int = 15
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse an image to a float64 2D grayscale array in [0, 1]-ish range.

    Accepts (H, W) or (H, W, C). For 3+ channels we use the Rec. 601 luma
    weights for the first three channels; extra channels are ignored. The
    output is float64 but *not* forcibly normalized -- callers that need [0,1]
    should pre-scale. We do divide uint inputs by their dtype max so 8-bit
    images land in [0,1].
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float64) / float(info.max)
    else:
        arr = arr.astype(np.float64)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"image must be 2D or 3D, got shape {arr.shape}")


def integral_image(img: np.ndarray) -> np.ndarray:
    """Return the summed-area table of ``img`` with a zero-padded top/left row.

    ``out[y, x]`` holds the sum of ``img[:y, :x]`` so that the sum over any
    rectangle ``[y0:y1, x0:x1]`` is::

        out[y1, x1] - out[y0, x1] - out[y1, x0] + out[y0, x0]

    The zero border removes special-casing at the image edges.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    out = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(np.cumsum(img, axis=0), axis=1, out=out[1:, 1:])
    return out


def _window_sums(sat: np.ndarray, shape, radius: int) -> np.ndarray:
    """Sum over a (2*radius+1) square window centered on each pixel.

    Windows are clipped at the image border (so border pixels see a smaller
    window). ``sat`` is the integral image from :func:`integral_image`.
    """
    h, w = shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h)
    y1 = np.clip(ys + radius + 1, 0, h)
    x0 = np.clip(xs - radius, 0, w)
    x1 = np.clip(xs + radius + 1, 0, w)

    # Broadcast the four corner lookups across the full grid.
    Y0 = y0[:, None]
    Y1 = y1[:, None]
    X0 = x0[None, :]
    X1 = x1[None, :]
    sums = sat[Y1, X1] - sat[Y0, X1] - sat[Y1, X0] + sat[Y0, X0]
    counts = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return sums, counts.astype(np.float64)


def local_mean_std(gray: np.ndarray, config: FeatureConfig):
    """Compute per-pixel local mean and (population) std via integral images.

    Returns ``(mean, std)`` arrays the same shape as ``gray``. Variance is
    computed as ``E[x^2] - E[x]^2`` from two summed-area tables, then clipped
    at zero before the square root to absorb tiny negative values from
    floating-point cancellation.
    """
    gray = np.asarray(gray, dtype=np.float64)
    radius = config.window // 2
    sat = integral_image(gray)
    sat_sq = integral_image(gray * gray)

    sums, counts = _window_sums(sat, gray.shape, radius)
    sums_sq, _ = _window_sums(sat_sq, gray.shape, radius)

    mean = sums / counts
    mean_sq = sums_sq / counts
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    return mean, std


def local_contrast(gray: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Per-pixel local z-score: (pixel - local_mean) / (local_std + eps).

    The sign encodes whether the pixel is brighter (+) or darker (-) than its
    surroundings; the magnitude is "how many local sigmas away". This is the
    primary defect-salience map fed to segmentation.
    """
    gray = np.asarray(gray, dtype=np.float64)
    mean, std = local_mean_std(gray, config)
    return (gray - mean) / (std + config.eps)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, convolved manually (no OpenCV/scipy).

    Edge density is a useful texture cue: scratches and cracks have high local
    gradient energy. We implement the 3x3 Sobel convolution by shifting the
    padded array -- nine multiply-adds, fully vectorized.
    """
    gray = np.asarray(gray, dtype=np.float64)
    p = np.pad(gray, 1, mode="edge")

    # Sobel-x and Sobel-y as explicit shifted sums.
    gx = (
        (p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:])
        - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2])
    )
    gy = (
        (p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
        - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:])
    )
    return np.sqrt(gx * gx + gy * gy)
