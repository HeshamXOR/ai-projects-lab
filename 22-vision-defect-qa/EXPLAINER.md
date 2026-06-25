# EXPLAINER — the algorithms, in depth

This document explains the three load-bearing pieces of the from-scratch core:
connected-components labeling, the Mahalanobis anomaly score, and the
uncertainty + diversity active-learning sampler. It also covers the supporting
integral-image and Otsu machinery they rely on.

---

## 0. Pipeline overview

```
image ──► grayscale ──► local-contrast salience (z-score)
                              │
                              ▼
                     threshold segmentation  ──►  binary defect mask
                              │
                              ▼
              connected-components labeling (union-find)
                              │
                              ▼
                 per-blob shape descriptors  ──►  defect objects

image ──► FeatureExtractor ──► feature vector ──► Mahalanobis score (anomaly)
                                              └──► active-learning pool
```

The salience map is a **local z-score**: for each pixel,
`(pixel - local_mean) / (local_std + eps)`. Computing a windowed mean and
variance at *every* pixel is the expensive step, so we use **integral images**.

### Integral images (summed-area tables)

`integral_image(img)[y, x]` holds the sum of `img[:y, :x]`, with a zero top/left
border. Then the sum over any rectangle `[y0:y1, x0:x1]` is four lookups:

```
S = SAT[y1, x1] - SAT[y0, x1] - SAT[y1, x0] + SAT[y0, x0]
```

Building two SATs — one of the image, one of the image squared — gives the
windowed mean `E[x]` and second moment `E[x^2]`, and the local variance is
`E[x^2] - E[x]^2` (clipped at 0 to absorb floating-point cancellation). This is
O(N) total regardless of window size, versus O(N·k²) for naive box filtering.

### Otsu's threshold

To binarize automatically, Otsu's method scans a 1D histogram of the defect
strength and picks the cut `t` maximizing the **between-class variance**:

```
σ_b²(t) = w0(t) · w1(t) · (μ0(t) - μ1(t))²
```

where `w0, w1` are the probability masses of the two classes split at `t` and
`μ0, μ1` their means. Maximizing between-class variance is algebraically
equivalent to minimizing within-class variance, but cheaper to evaluate from
cumulative histogram sums — we compute every candidate `t` at once with prefix
sums and take the argmax.

---

## 1. Connected-components labeling (two-pass + union-find)

**Goal:** turn a binary mask into integer labels where each maximal connected
region of foreground pixels gets a unique id.

### Why two-pass + union-find

A single raster pass can't assign final labels, because a region shaped like a
`U` looks like two separate strips until you reach the bottom that joins them.
The two-pass algorithm handles this by:

1. **Pass 1 — provisional labels + equivalences.** Scan top-to-bottom,
   left-to-right. For each foreground pixel, look only at *already-visited*
   neighbors (north and west for 4-connectivity; add the two upper diagonals
   for 8-connectivity). 
   - No labeled neighbor → start a **new provisional label** (a new union-find
     set).
   - One or more labeled neighbors → take the **smallest** neighbor label, and
     `union()` all the neighbor labels together, recording that they are the
     same component.
2. **Resolve equivalences.** Each provisional label maps to a union-find
   element; `find()` gives its canonical root. We assign final, contiguous ids
   to roots in ascending provisional-label order (so components are numbered in
   raster order of first appearance).
3. **Pass 2 — relabel.** A lookup table maps every provisional label to its
   final id; we apply it to the whole label image vectorized.

### Union-find (disjoint sets)

The equivalence bookkeeping is a **disjoint-set forest** with:

- **Path compression** in `find()` — every node on the lookup path is pointed
  straight at the root, flattening the tree.
- **Union by rank** in `union()` — the shorter tree is hung under the taller,
  keeping trees shallow.

Together these give near-constant amortized time per operation (inverse
Ackermann, α(N)), so the whole labeler is effectively O(N).

### 4- vs 8-connectivity

4-connectivity joins only orthogonal neighbors; 8-connectivity also joins
diagonal touches. A diagonal chain of pixels `(0,0),(1,1),(2,2)` is **one**
component under 8-connectivity but **three** under 4-connectivity — exactly the
case asserted in `tests/test_components.py`.

### Cross-check: flood fill

`flood_fill_label` implements the same labeling independently via iterative
BFS/DFS with an explicit stack (no recursion, so big blobs can't overflow the
stack). The tests assert it produces the same component count as the two-pass
labeler — a strong correctness check on both.

### Blob descriptors

For each labeled component we compute classical shape features:

- **area** = pixel count; **bbox**, **centroid** = mean (y, x).
- **perimeter** = number of pixel *edges* facing background or the image border
  (counting exposed sides, not boundary pixels — this avoids the systematic
  undercount of pixel-counting and is resolution-stable).
- **aspect_ratio** = bbox width / height; **extent** = area / bbox area.
- **eccentricity / orientation** from the **second central moments**: we build
  the 2×2 covariance of the blob's pixel coordinates (the moment ellipse), take
  its eigenvalues λ_min ≤ λ_max, and set
  `eccentricity = sqrt(1 - λ_min/λ_max)` (0 = round, →1 = elongated), with
  orientation `0.5·atan2(2·μ_xy, μ_yy - μ_xx)`.

---

## 2. Mahalanobis anomaly score

**Goal:** one-class detection — learn what "good" looks like in feature space,
flag anything far from it.

We model defect-free feature vectors as a single multivariate Gaussian
`N(μ, Σ)` and score a new sample `x` by its **squared Mahalanobis distance**:

```
D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

### Why Mahalanobis, not Euclidean

Euclidean distance treats every feature as independent and equally scaled.
Mahalanobis whitens the space by `Σ⁻¹`: a deviation along a **low-variance,
tightly-correlated** direction counts for much more than the same-sized
deviation along a noisy, high-variance direction. For defects this is exactly
right — a small but *consistent* shift is more suspicious than a large swing in
an already-jittery feature. `tests/test_anomaly_segmentation.py` asserts that
moving "against" a strong correlation scores higher than moving "along" it by
the same Euclidean amount.

### The numerical crux: regularization + Cholesky

`Σ` is frequently singular or ill-conditioned (correlated features, or fewer
good samples than dimensions). A raw inverse would explode. We:

1. Optionally **standardize** features (per-dimension z-score) so the ridge is
   scale-invariant.
2. Add a **relative ridge**: `Σ_reg = Σ + λ·avg_var·I`, shrinking toward a
   scaled identity. This guarantees positive-definiteness.
3. Invert via **Cholesky** (`Σ_reg = L Lᵀ`, then `Σ⁻¹ = L⁻ᵀ L⁻¹`) — the fast,
   stable route for symmetric PD matrices — falling back to the Moore–Penrose
   **pseudo-inverse** if even the regularized matrix isn't PD.

The score is evaluated as a batched quadratic form
`einsum("ij,ij->i", Z @ Σ⁻¹, Z)` over standardized residuals `Z`, clipped at 0.
After fitting, the 95th percentile of in-distribution scores is stored as a
default **decision threshold** (also reused as the active-learning boundary).

---

## 3. Active learning: uncertainty + diversity

**Goal:** spend a limited labeling budget on the most *informative* unlabeled
images.

Two signals, each insufficient alone:

- **Uncertainty.** For the one-class model, the natural proxy is **closeness to
  the decision boundary**: `1 / (1 + |score - threshold|)`. Samples sitting
  right on the good/bad threshold are maximally informative; obviously-good or
  obviously-bad ones teach little. (For a probabilistic classifier we also
  provide normalized **entropy** uncertainty.)
- **Diversity.** Pure uncertainty sampling *clusters* — it happily queries a
  dozen near-identical borderline images. We counter with a greedy
  **k-center / farthest-point** rule in feature space.

### The greedy combined rule

```
seed     = argmax uncertainty                      # most uncertain point
repeat:
    div(i)   = min distance from i to selected set, renormalized to [0,1]
    gain(i)  = α · uncertainty(i) + (1-α) · div(i)
    pick     = argmax gain(i)  over remaining
```

`α ∈ [0,1]` trades the two objectives (1.0 = pure uncertainty, 0.0 = pure
k-center coverage). Re-normalizing the distance term each round keeps both
terms on a comparable `[0,1]` scale as the selected set grows.

**Why this avoids near-duplicates:** once a point is selected, every point near
it has its `div` term collapse toward 0, so a near-duplicate is only ever picked
if its uncertainty is overwhelming. `tests/test_active.py` constructs two
near-identical points and asserts the sampler never selects both, while still
seeding from the most-uncertain candidate and reaching out to far-apart anchors.

Features are z-scored before the distance computation so no single high-variance
dimension dominates the diversity term.
