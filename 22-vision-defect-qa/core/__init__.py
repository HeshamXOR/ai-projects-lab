"""core -- from-scratch industrial defect-inspection pipeline.

Public surface:

* :mod:`core.features`     -- local-contrast feature maps (integral images).
* :mod:`core.segmentation` -- global / Otsu / adaptive thresholding.
* :mod:`core.components`   -- connected-components labeling + blob descriptors.
* :mod:`core.anomaly`      -- Mahalanobis one-class anomaly scorer.
* :mod:`core.active`       -- uncertainty + diversity active-learning sampler.
* :mod:`core.extractor`    -- pluggable feature extractors (NumPy default,
                              optional CNN).
* :mod:`core.pipeline`     -- glue: image -> mask + blobs + anomaly score.
"""

from __future__ import annotations

from .active import (
    ActiveConfig,
    boundary_uncertainty,
    entropy_uncertainty,
    select_samples,
)
from .anomaly import AnomalyConfig, MahalanobisAnomalyModel
from .components import (
    BlobDescriptor,
    UnionFind,
    describe_blobs,
    flood_fill_label,
    label_components,
)
from .extractor import (
    FeatureExtractor,
    HandcraftedExtractor,
    TorchCNNExtractor,
)
from .features import (
    FeatureConfig,
    gradient_magnitude,
    integral_image,
    local_contrast,
    local_mean_std,
    to_grayscale,
)
from .pipeline import InspectionResult, InspectionService
from .segmentation import (
    SegmentationConfig,
    adaptive_threshold,
    global_threshold,
    otsu_threshold,
    segment,
)

__all__ = [
    "ActiveConfig",
    "boundary_uncertainty",
    "entropy_uncertainty",
    "select_samples",
    "AnomalyConfig",
    "MahalanobisAnomalyModel",
    "BlobDescriptor",
    "UnionFind",
    "describe_blobs",
    "flood_fill_label",
    "label_components",
    "FeatureExtractor",
    "HandcraftedExtractor",
    "TorchCNNExtractor",
    "FeatureConfig",
    "gradient_magnitude",
    "integral_image",
    "local_contrast",
    "local_mean_std",
    "to_grayscale",
    "InspectionResult",
    "InspectionService",
    "SegmentationConfig",
    "adaptive_threshold",
    "global_threshold",
    "otsu_threshold",
    "segment",
]
