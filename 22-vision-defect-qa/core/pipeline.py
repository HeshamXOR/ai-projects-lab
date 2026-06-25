"""End-to-end inspection pipeline + run-length mask encoding.

WHY THIS MODULE EXISTS
----------------------
The individual algorithms in ``core`` are deliberately decoupled, but the API
needs a single object that owns the configured pipeline and the fitted anomaly
model. :class:`InspectionService` is that object. It:

* extracts a per-image feature vector (via an injected
  :class:`~core.extractor.FeatureExtractor`),
* builds the local-contrast salience map and segments it into a defect mask,
* labels connected components and computes blob descriptors,
* scores the image's global anomaly via the fitted Mahalanobis model (if fit),
* maintains an unlabeled pool so the active-learning sampler has data to choose
  from.

The mask is returned run-length encoded by default because defect masks are
sparse (mostly background), so RLE is dramatically smaller over the wire than a
nested list -- but a nested-list option is provided for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .active import ActiveConfig, boundary_uncertainty, select_samples
from .anomaly import AnomalyConfig, MahalanobisAnomalyModel
from .components import BlobDescriptor, describe_blobs, label_components
from .extractor import FeatureExtractor, HandcraftedExtractor
from .features import FeatureConfig, local_contrast, to_grayscale
from .segmentation import SegmentationConfig, segment

__all__ = ["InspectionResult", "InspectionService", "rle_encode", "rle_decode"]


def rle_encode(mask: np.ndarray) -> Dict[str, object]:
    """Row-major run-length encode a boolean mask.

    Returns ``{"shape": [h, w], "runs": [[start, length], ...]}`` where each run
    marks a contiguous span of True pixels in flattened row-major order. This
    is compact for the sparse masks defect detection produces and trivially
    decodable on any client.
    """
    mask = np.asarray(mask).astype(bool).ravel()
    h, w = (np.asarray(mask).shape if mask.ndim == 2 else (0, 0))
    runs: List[List[int]] = []
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append([int(i), int(j - i)])
            i = j
        else:
            i += 1
    return {"runs": runs}


def rle_decode(shape, runs) -> np.ndarray:
    """Inverse of :func:`rle_encode`. ``shape`` is ``(h, w)``."""
    h, w = shape
    flat = np.zeros(h * w, dtype=bool)
    for start, length in runs:
        flat[start : start + length] = True
    return flat.reshape(h, w)


@dataclass
class InspectionResult:
    """Structured outcome of inspecting a single image."""

    shape: tuple
    mask_rle: Dict[str, object]
    n_components: int
    blobs: List[BlobDescriptor]
    anomaly_score: Optional[float]
    feature_vector: np.ndarray

    def as_dict(self, include_features: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "mask": {
                "encoding": "rle_row_major",
                "shape": [int(self.shape[0]), int(self.shape[1])],
                "runs": self.mask_rle["runs"],
            },
            "n_components": int(self.n_components),
            "blobs": [b.as_dict() for b in self.blobs],
            "anomaly_score": (
                None if self.anomaly_score is None
                else float(self.anomaly_score)
            ),
        }
        if include_features:
            out["feature_vector"] = [float(v) for v in self.feature_vector]
        return out


@dataclass
class InspectionService:
    """Owns the configured pipeline, fitted anomaly model and unlabeled pool.

    Parameters are injected so the service is testable and the extractor is
    swappable (NumPy default or a CNN). The anomaly model starts unfitted;
    call :meth:`fit_good` with defect-free images (or :meth:`fit_good_features`
    with precomputed vectors) before anomaly scores are produced.
    """

    extractor: FeatureExtractor = field(default_factory=HandcraftedExtractor)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    seg_config: SegmentationConfig = field(default_factory=SegmentationConfig)
    anomaly_config: AnomalyConfig = field(default_factory=AnomalyConfig)
    active_config: ActiveConfig = field(default_factory=ActiveConfig)
    anomaly_threshold: float = 0.0  # set by fit_good; boundary for active learn

    _model: MahalanobisAnomalyModel = field(init=False, default=None)  # type: ignore
    _pool_features: List[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._model = MahalanobisAnomalyModel(self.anomaly_config)

    # ---- Fitting the "good" distribution ----------------------------------
    def fit_good_features(self, features: np.ndarray) -> None:
        """Fit the anomaly model from precomputed good-sample feature vectors.

        Also sets a default ``anomaly_threshold`` at the 95th percentile of the
        in-distribution scores -- a reasonable boundary for both flagging and
        active-learning uncertainty.
        """
        X = np.asarray(features, dtype=np.float64)
        self._model.fit(X)
        scores = self._model.score(X)
        self.anomaly_threshold = float(np.percentile(scores, 95.0))

    def fit_good(self, images: List[np.ndarray]) -> None:
        """Extract features from defect-free ``images`` and fit the model."""
        feats = np.stack([self.extractor.extract(im) for im in images], axis=0)
        self.fit_good_features(feats)

    @property
    def is_fitted(self) -> bool:
        return self._model.is_fitted

    # ---- Inspection -------------------------------------------------------
    def inspect(self, image: np.ndarray) -> InspectionResult:
        """Run the full pipeline on one image and return structured results."""
        gray = to_grayscale(image)
        salience = local_contrast(gray, self.feature_config)
        mask = segment(salience, self.seg_config)
        labels, count = label_components(mask, connectivity=8)
        blobs = describe_blobs(labels, count, min_area=1)

        feat = self.extractor.extract(image)
        score: Optional[float] = None
        if self._model.is_fitted:
            score = float(self._model.score(feat))

        return InspectionResult(
            shape=tuple(gray.shape),
            mask_rle=rle_encode(mask),
            n_components=count,
            blobs=blobs,
            anomaly_score=score,
            feature_vector=feat,
        )

    # ---- Active learning --------------------------------------------------
    def set_pool(self, features: np.ndarray) -> None:
        """Replace the unlabeled pool with precomputed feature vectors."""
        X = np.asarray(features, dtype=np.float64)
        self._pool_features = [row for row in X]

    def add_to_pool(self, image: np.ndarray) -> int:
        """Extract features from ``image`` and append to the pool. Returns idx."""
        self._pool_features.append(self.extractor.extract(image))
        return len(self._pool_features) - 1

    def pool_size(self) -> int:
        return len(self._pool_features)

    def suggest_labels(self, n_select: int) -> List[int]:
        """Return pool indices the active learner recommends labeling.

        Uncertainty is derived from each pooled sample's anomaly score relative
        to ``anomaly_threshold`` (closeness to the decision boundary). The
        diversity term operates in feature space. Requires a fitted model.
        """
        if not self._pool_features:
            return []
        feats = np.stack(self._pool_features, axis=0)
        if self._model.is_fitted:
            scores = self._model.score(feats)
            unc = boundary_uncertainty(scores, self.anomaly_threshold)
        else:
            # Without a model, treat everything as equally uncertain so the
            # selection reduces to pure diversity (k-center) coverage.
            unc = np.ones(feats.shape[0], dtype=np.float64)
        return select_samples(feats, unc, n_select, self.active_config)
