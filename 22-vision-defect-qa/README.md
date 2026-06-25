# 22 — Vision Defect QA

Industrial surface-defect detection with a classical, illumination-robust
computer-vision core and an active-learning loop — served over a FastAPI HTTP
API. The detection math is written from scratch in NumPy (no OpenCV/skimage for
the core algorithms); a pretrained CNN is an optional, pluggable feature
extractor.

## What I implemented from scratch

Everything below is pure NumPy — no OpenCV, no scikit-image, no scipy for the
core algorithms:

- **Local-contrast feature maps via integral images** (`core/features.py`).
  Per-pixel local mean/std over a sliding window computed in O(N) with
  summed-area tables (implemented from cumulative sums), turned into a local
  z-score salience map. Also a hand-convolved Sobel gradient-magnitude map.
- **Threshold segmentation** (`core/segmentation.py`): a global threshold, an
  **Otsu** threshold computed from the histogram by maximizing between-class
  variance, and an **adaptive local-mean** threshold (reusing the integral
  image). Polarity-aware (dark / bright / absolute defects).
- **Connected-components labeling, from scratch** (`core/components.py`): the
  classic **two-pass algorithm backed by a union-find** (path compression +
  union by rank), with both 4- and 8-connectivity. An independent **iterative
  flood-fill** labeler is also implemented and cross-checked in tests. Then
  per-component **blob/shape descriptors**: area, bounding box, centroid,
  edge-counting perimeter estimate, aspect ratio, extent/fill ratio, and an
  image-moment **eccentricity** + orientation (covariance-ellipse of the blob).
- **Mahalanobis anomaly scorer** (`core/anomaly.py`): learns a "good" feature
  Gaussian (mean + covariance) from defect-free samples and scores new samples
  by Mahalanobis distance. The covariance is **regularized** (relative ridge)
  for invertibility and inverted via **Cholesky** with a pseudo-inverse
  fallback — the distance math is hand-written.
- **Active-learning sampler** (`core/active.py`): a greedy selector that
  combines **uncertainty** (closeness to the anomaly decision boundary, or
  classifier entropy) with **diversity** (farthest-point / k-center coverage in
  feature space) so it never queries near-duplicate images.

A pretrained CNN extractor (`core/extractor.py`, `TorchCNNExtractor`) is an
**optional** drop-in behind a `FeatureExtractor` protocol; the default
`HandcraftedExtractor` is pure NumPy so the whole system runs without torch.

## Run it

```bash
# (optional) create a venv, then:
pip install -r requirements.txt

# run the API
uvicorn app:app --reload --port 8000
# or
python app.py

# run the tests that prove the core
pytest -q
```

Docker:

```bash
docker build -t vision-defect-qa .
docker run -p 8000:8000 vision-defect-qa
```

## API

Base URL `http://localhost:8000`.

### `GET /health`
Liveness + model status.
```json
{ "status": "ok", "model_fitted": false, "feature_dim": 24,
  "pool_size": 0, "anomaly_threshold": 0.0 }
```

### `POST /fit`
Fit the defect-free distribution. Provide **either** `images` (list of image
payloads) **or** precomputed `features` (n_samples × n_features). Needs ≥ 2.
```json
{ "features": [[0.1, 0.2, ...], [0.13, 0.19, ...]], "ridge": 0.01,
  "standardize": true }
```
Response: `{ "fitted": true, "n_samples": 2, "feature_dim": 24,
"anomaly_threshold": 4.7 }`

### `POST /inspect`
Inspect one image. The image is a nested list (2D `HxW` grayscale or 3D
`HxWxC`) or a base64-encoded PNG/JPEG (`{"base64": "..."}`, requires Pillow).
```json
{
  "image": { "array": [[0,0,0],[0,255,0],[0,0,0]] },
  "segmentation": { "method": "otsu", "polarity": "abs" },
  "include_features": false
}
```
Response:
```json
{
  "shape": [3, 3],
  "mask": { "encoding": "rle_row_major", "shape": [3,3],
            "runs": [[4, 1]] },
  "n_components": 1,
  "blobs": [ { "label": 1, "area": 1, "bbox": [1,1,1,1],
               "centroid": [1.0,1.0], "perimeter": 4.0,
               "aspect_ratio": 1.0, "extent": 1.0,
               "eccentricity": 0.0, "orientation": 0.0 } ],
  "anomaly_score": null
}
```
The mask is **run-length encoded** (row-major) because defect masks are sparse;
`runs` is a list of `[start, length]` spans over the flattened image.

### `POST /sample`
Active learning. Given an unlabeled pool of `features`, return the indices most
worth labeling. Optionally pass per-sample `uncertainty`; otherwise it is
derived from anomaly scores (if a model is fitted) or selection falls back to
pure diversity. `alpha` ∈ [0,1] trades uncertainty (1.0) vs diversity (0.0).
```json
{ "features": [[0,0],[0.01,0],[9,0],[0,9]], "n_select": 3, "alpha": 0.5 }
```
Response: `{ "selected_indices": [...], "n_selected": 3,
"uncertainty_used": [...] }`

## Plugging in a CNN extractor

```python
from core.extractor import TorchCNNExtractor
from core import InspectionService
service = InspectionService(extractor=TorchCNNExtractor("resnet18"))
```
This requires `torch`/`torchvision` (commented out in `requirements.txt`).
Nothing else in the codebase changes — the anomaly scorer and active learner
operate on whatever feature vectors the extractor produces.

See **EXPLAINER.md** for the algorithm internals (connected components,
Mahalanobis math, uncertainty + diversity sampling).
