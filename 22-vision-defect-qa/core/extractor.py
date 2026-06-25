"""Feature extractors behind a clean, swappable interface.

WHY THIS MODULE EXISTS
----------------------
The anomaly scorer and the active-learning sampler both operate on a *feature
vector* per image, not on raw pixels. How that vector is produced should be a
pluggable decision:

* By default we use a **pure-NumPy handcrafted extractor** so the entire system
  runs with zero deep-learning dependencies. It summarizes an image with
  illumination-robust statistics: histogram of local-contrast z-scores,
  gradient-energy statistics, and global intensity moments.
* Optionally a **pretrained CNN** (e.g. torchvision ResNet penultimate
  activations) can be dropped in. We define a ``FeatureExtractor`` Protocol and
  document the exact plug-in point; nothing in the rest of the codebase imports
  torch.

This is dependency injection: callers pass *an object implementing the
protocol*. The service constructs the handcrafted one by default but accepts
any conforming extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .features import (
    FeatureConfig,
    to_grayscale,
    local_contrast,
    gradient_magnitude,
)

__all__ = [
    "FeatureExtractor",
    "HandcraftedExtractor",
    "TorchCNNExtractor",
]


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol every feature extractor must satisfy.

    Implementations map a single image (2D grayscale or 3D color array) to a
    fixed-length 1D float feature vector. The vector length must be constant
    for a given extractor instance (``dim``) so downstream covariance math and
    distance computations are well-defined.
    """

    @property
    def dim(self) -> int:
        """Length of the feature vector this extractor produces."""
        ...

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Return a 1D float64 feature vector for ``image``."""
        ...


@dataclass
class HandcraftedExtractor:
    """Pure-NumPy feature extractor (the zero-dependency default).

    The feature vector concatenates, in order:

    1. A normalized histogram of the *absolute* local-contrast z-score
       (``contrast_bins`` values) -- captures how much anomalous local contrast
       the image contains and at what magnitudes.
    2. Global intensity moments: mean, std, skew-ish (third standardized
       moment), min, max (5 values).
    3. Gradient-energy statistics: mean and std of the Sobel magnitude, plus
       the fraction of pixels whose gradient exceeds the global mean gradient
       (3 values) -- a texture/edge-density summary.

    All features are illumination-aware (the contrast histogram is computed on
    a local z-score, not raw intensity), which is what makes them useful for
    distinguishing "good" from "defective" surfaces.
    """

    contrast_bins: int = 16
    contrast_clip: float = 8.0
    config: FeatureConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = FeatureConfig()
        if self.contrast_bins < 1:
            raise ValueError("contrast_bins must be >= 1")

    @property
    def dim(self) -> int:
        return self.contrast_bins + 5 + 3

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = to_grayscale(image)

        # (1) Local-contrast histogram.
        contrast = np.abs(local_contrast(gray, self.config))
        clipped = np.clip(contrast, 0.0, self.contrast_clip)
        hist, _ = np.histogram(
            clipped, bins=self.contrast_bins, range=(0.0, self.contrast_clip)
        )
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total

        # (2) Global intensity moments.
        mean = float(gray.mean())
        std = float(gray.std())
        if std > 1e-12:
            skew = float(np.mean(((gray - mean) / std) ** 3))
        else:
            skew = 0.0
        gmin = float(gray.min())
        gmax = float(gray.max())
        moments = np.array([mean, std, skew, gmin, gmax], dtype=np.float64)

        # (3) Gradient-energy statistics.
        grad = gradient_magnitude(gray)
        gmean = float(grad.mean())
        gstd = float(grad.std())
        edge_frac = float(np.mean(grad > gmean)) if grad.size else 0.0
        grad_stats = np.array([gmean, gstd, edge_frac], dtype=np.float64)

        vec = np.concatenate([hist, moments, grad_stats])
        return vec.astype(np.float64)


@dataclass
class TorchCNNExtractor:
    """Optional pretrained-CNN extractor (penultimate-layer activations).

    This class is the documented plug-in point for a deep extractor. It is
    *lazy*: torch/torchvision are imported only inside ``__post_init__`` so the
    rest of the project has no hard dependency on them. To use it::

        from core.extractor import TorchCNNExtractor
        ext = TorchCNNExtractor(model_name="resnet18")
        service = InspectionService(extractor=ext)

    The default forward pass:
      * converts the image to a 3-channel float tensor in ImageNet
        normalization,
      * runs it through the backbone up to the global-average-pool,
      * returns the pooled activation vector (512-d for resnet18).

    Everything downstream (anomaly scoring, active learning) is agnostic to
    which extractor produced the vector, so swapping this in changes only the
    feature space, not the algorithms.
    """

    model_name: str = "resnet18"
    device: str = "cpu"
    _model: object = None
    _dim: int = 512

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TorchCNNExtractor requires torch and torchvision. Install "
                "them or use HandcraftedExtractor (the default)."
            ) from exc

        import torch
        from torchvision import models

        builder = getattr(models, self.model_name)
        net = builder(weights="DEFAULT")
        # Drop the classification head; keep everything up to the pooled vector.
        modules = list(net.children())[:-1]
        self._model = torch.nn.Sequential(*modules).eval().to(self.device)
        # Infer output dim with a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, device=self.device)
            out = self._model(dummy)
        self._dim = int(out.reshape(1, -1).shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def extract(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover
        import torch

        gray = to_grayscale(image)
        # Expand grayscale to 3 channels and apply ImageNet normalization.
        x = np.stack([gray, gray, gray], axis=0)[None]  # (1,3,H,W)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        x = (x - mean) / std
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self._model(tensor)
        return out.reshape(-1).cpu().numpy().astype(np.float64)
