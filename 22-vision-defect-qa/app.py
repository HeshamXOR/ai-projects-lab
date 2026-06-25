"""FastAPI service for industrial defect inspection with active learning.

WHY THIS MODULE EXISTS
----------------------
This is the network boundary around the :class:`core.InspectionService`. It
exposes the pipeline over HTTP with strict Pydantic validation, structured JSON
responses, and explicit error handling. A single process-global service holds
the fitted anomaly model and the unlabeled pool across requests (industrial
inspectors are long-lived single-tenant services, so in-memory state is
appropriate; swap for a store if you need horizontal scaling).

Endpoints
---------
* ``GET  /health``  -- liveness + whether the anomaly model is fitted.
* ``POST /fit``     -- fit the "good" feature distribution from defect-free
                       samples (raw images OR precomputed feature vectors).
* ``POST /inspect`` -- inspect one image; returns RLE defect mask, anomaly
                       score and per-blob descriptors.
* ``POST /sample``  -- active learning; given an unlabeled pool, return the
                       indices most worth labeling.

Images are accepted as nested JSON lists (2D grayscale or 3D HxWxC) or as a
base64-encoded PNG/JPEG (decoded with Pillow if available).

Run::

    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from core import (
    ActiveConfig,
    AnomalyConfig,
    FeatureConfig,
    HandcraftedExtractor,
    InspectionService,
    SegmentationConfig,
    select_samples,
)

app = FastAPI(
    title="Vision Defect QA",
    version="1.0.0",
    description=(
        "Classical industrial defect detection (local contrast + from-scratch "
        "connected components), Mahalanobis anomaly scoring, and an "
        "uncertainty+diversity active-learning loop."
    ),
)

# --- Process-global service. Holds fitted model + unlabeled pool. ----------
SERVICE = InspectionService(extractor=HandcraftedExtractor())


# ===========================================================================
# Request / response schemas
# ===========================================================================
class ImagePayload(BaseModel):
    """An image supplied either as nested lists or a base64-encoded file.

    Exactly one of ``array`` or ``base64`` must be provided.
    """

    array: Optional[list] = Field(
        default=None,
        description="2D (HxW) or 3D (HxWxC) nested list of numbers.",
    )
    base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG/JPEG (requires Pillow on the server).",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "ImagePayload":
        if (self.array is None) == (self.base64 is None):
            raise ValueError("provide exactly one of 'array' or 'base64'")
        return self

    def to_ndarray(self) -> np.ndarray:
        if self.array is not None:
            try:
                arr = np.asarray(self.array, dtype=np.float64)
            except Exception as exc:  # malformed nesting
                raise ValueError(f"invalid array payload: {exc}") from exc
            if arr.ndim not in (2, 3):
                raise ValueError("array must be 2D or 3D")
            return arr
        # base64 branch
        try:
            raw = base64.b64decode(self.base64, validate=True)
        except Exception as exc:
            raise ValueError(f"invalid base64: {exc}") from exc
        try:
            from PIL import Image  # optional dependency
        except Exception as exc:  # pragma: no cover - optional
            raise ValueError(
                "base64 image decoding requires Pillow on the server"
            ) from exc
        try:
            img = Image.open(io.BytesIO(raw))
            return np.asarray(img)
        except Exception as exc:
            raise ValueError(f"could not decode image: {exc}") from exc


class SegConfigModel(BaseModel):
    """Optional per-request override of the segmentation configuration."""

    method: str = "otsu"
    threshold: float = 3.0
    block: int = 31
    offset: float = 0.5
    polarity: str = "abs"
    nbins: int = 256

    def to_config(self) -> SegmentationConfig:
        return SegmentationConfig(
            method=self.method,  # type: ignore[arg-type]
            threshold=self.threshold,
            block=self.block,
            offset=self.offset,
            polarity=self.polarity,  # type: ignore[arg-type]
            nbins=self.nbins,
        )


class InspectRequest(BaseModel):
    image: ImagePayload
    segmentation: Optional[SegConfigModel] = None
    include_features: bool = False


class FitRequest(BaseModel):
    """Fit the good distribution from images OR precomputed feature vectors."""

    images: Optional[List[ImagePayload]] = None
    features: Optional[List[List[float]]] = Field(
        default=None,
        description="Precomputed (n_samples x n_features) good-sample vectors.",
    )
    ridge: float = 1e-2
    standardize: bool = True

    @model_validator(mode="after")
    def _one_source(self) -> "FitRequest":
        if (self.images is None) == (self.features is None):
            raise ValueError("provide exactly one of 'images' or 'features'")
        if self.images is not None and len(self.images) < 2:
            raise ValueError("need at least 2 good images to fit")
        if self.features is not None and len(self.features) < 2:
            raise ValueError("need at least 2 feature vectors to fit")
        return self


class SampleRequest(BaseModel):
    """Active-learning query.

    Provide a pool of unlabeled ``features`` (and optional ``uncertainty``). If
    ``uncertainty`` is omitted and a model is fitted, boundary uncertainty is
    derived from anomaly scores; otherwise selection is pure diversity.
    """

    features: List[List[float]] = Field(..., min_length=1)
    uncertainty: Optional[List[float]] = None
    n_select: int = Field(..., ge=1)
    alpha: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> "SampleRequest":
        if self.uncertainty is not None and len(self.uncertainty) != len(
            self.features
        ):
            raise ValueError("uncertainty length must match features")
        return self


# ===========================================================================
# Endpoints
# ===========================================================================
@app.get("/health")
def health() -> dict:
    """Liveness probe and model status."""
    return {
        "status": "ok",
        "model_fitted": SERVICE.is_fitted,
        "feature_dim": SERVICE.extractor.dim,
        "pool_size": SERVICE.pool_size(),
        "anomaly_threshold": SERVICE.anomaly_threshold,
    }


@app.post("/fit")
def fit(req: FitRequest) -> dict:
    """Fit the defect-free feature distribution for anomaly scoring."""
    SERVICE.anomaly_config = AnomalyConfig(
        ridge=req.ridge, standardize=req.standardize
    )
    SERVICE._model = type(SERVICE._model)(SERVICE.anomaly_config)
    try:
        if req.features is not None:
            feats = np.asarray(req.features, dtype=np.float64)
            SERVICE.fit_good_features(feats)
        else:
            imgs = [p.to_ndarray() for p in (req.images or [])]
            SERVICE.fit_good(imgs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "fitted": True,
        "n_samples": SERVICE._model.n_samples_,
        "feature_dim": SERVICE._model.dim_,
        "anomaly_threshold": SERVICE.anomaly_threshold,
    }


@app.post("/inspect")
def inspect(req: InspectRequest) -> dict:
    """Inspect one image; return defect mask, anomaly score, blob descriptors."""
    try:
        image = req.image.to_ndarray()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if req.segmentation is not None:
        try:
            SERVICE.seg_config = req.segmentation.to_config()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = SERVICE.inspect(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.as_dict(include_features=req.include_features)


@app.post("/sample")
def sample(req: SampleRequest) -> dict:
    """Return the unlabeled-pool indices most worth labeling next."""
    feats = np.asarray(req.features, dtype=np.float64)
    if feats.ndim != 2:
        raise HTTPException(status_code=422, detail="features must be 2D")

    if req.uncertainty is not None:
        unc = np.asarray(req.uncertainty, dtype=np.float64)
    elif SERVICE.is_fitted and feats.shape[1] == SERVICE._model.dim_:
        from core import boundary_uncertainty

        scores = SERVICE._model.score(feats)
        unc = boundary_uncertainty(scores, SERVICE.anomaly_threshold)
    else:
        # No supervision signal -> equal uncertainty -> pure diversity.
        unc = np.ones(feats.shape[0], dtype=np.float64)

    cfg = ActiveConfig(alpha=req.alpha)
    try:
        selected = select_samples(feats, unc, req.n_select, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "selected_indices": selected,
        "n_selected": len(selected),
        "uncertainty_used": unc.tolist(),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
