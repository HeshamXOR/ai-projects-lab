"""Active-learning sample selection combining uncertainty and diversity.

WHY THIS MODULE EXISTS
----------------------
Labeling industrial defect images is expensive (a human inspector must adjudge
each one). Active learning spends that budget where it helps the model most.
Two signals matter, and using either alone is a known failure mode:

* **Uncertainty** -- query points the model is least sure about. For a
  one-class anomaly model the natural proxy is "how close is this sample's
  anomaly score to the good/bad decision boundary": samples sitting right on
  the threshold are maximally informative; clearly-good or clearly-bad ones
  teach little. We also support a generic entropy input when a probabilistic
  classifier is available.
* **Diversity** -- pure uncertainty sampling clusters: it will happily pick a
  dozen near-duplicate borderline images. We counter this with a greedy
  **k-center / farthest-point** selection in feature space, so each new query
  is far (in feature distance) from everything already chosen.

THE STRATEGY
------------
We combine them with a greedy facility-location-style rule. Starting from the
most-uncertain point, we iteratively add the unlabeled point maximizing::

    gain(i) = alpha * uncertainty(i)
            + (1 - alpha) * normalized_distance_to_selected(i)

``alpha`` trades off the two objectives. The distance term is the min distance
from candidate ``i`` to the already-selected set (the k-center coverage term),
renormalized each round so the two terms stay comparable. This provably avoids
near-duplicates: once a point is chosen, its neighbors' distance term collapses
to ~0, so they only get picked if their uncertainty is overwhelming.

All of this is implemented from scratch with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "ActiveConfig",
    "boundary_uncertainty",
    "entropy_uncertainty",
    "select_samples",
]


@dataclass(frozen=True)
class ActiveConfig:
    """Configuration for active-learning sample selection.

    Attributes
    ----------
    alpha:
        Weight on uncertainty in ``[0, 1]``. ``1.0`` = pure uncertainty
        sampling, ``0.0`` = pure diversity (k-center) sampling. The default
        0.5 balances the two.
    normalize_features:
        Z-score features before computing distances so no single high-variance
        dimension dominates the diversity term.
    metric:
        Distance metric for the diversity term: ``"euclidean"`` only (kept as a
        field for forward compatibility / documentation of intent).
    """

    alpha: float = 0.5
    normalize_features: bool = True
    metric: str = "euclidean"

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


def boundary_uncertainty(
    scores: np.ndarray, threshold: float
) -> np.ndarray:
    """Uncertainty as closeness of an anomaly score to the decision boundary.

    Far from ``threshold`` (very good or very anomalous) -> low uncertainty.
    Right at the threshold -> uncertainty 1.0. We map the absolute gap through
    ``1 / (1 + |score - threshold|)`` so the result is bounded in ``(0, 1]``
    and monotonic. This needs no probability calibration, which is why it suits
    the Mahalanobis one-class model.
    """
    scores = np.asarray(scores, dtype=np.float64)
    gap = np.abs(scores - float(threshold))
    return 1.0 / (1.0 + gap)


def entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """Binary/Multiclass entropy uncertainty from predicted probabilities.

    For a probability vector per sample, entropy ``-sum p log p`` is maximal
    when the model is maximally undecided. Normalized to ``[0, 1]`` by dividing
    by ``log(n_classes)``. Accepts shape ``(n,)`` (binary p) or ``(n, k)``.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    ent = -np.sum(p * np.log(p), axis=1)
    k = p.shape[1]
    return ent / np.log(k) if k > 1 else ent


def _normalize(features: np.ndarray) -> np.ndarray:
    """Per-dimension z-score; constant dims left untouched."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def _pairwise_min_distance(
    candidates: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Min Euclidean distance from each candidate row to any selected row."""
    if selected.shape[0] == 0:
        return np.full(candidates.shape[0], np.inf)
    # (n_cand, n_sel) distance matrix via the (a-b)^2 expansion.
    diff = candidates[:, None, :] - selected[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    return d.min(axis=1)


def select_samples(
    features: np.ndarray,
    uncertainty: np.ndarray,
    n_select: int,
    config: Optional[ActiveConfig] = None,
    candidate_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Greedily select ``n_select`` informative samples.

    Parameters
    ----------
    features:
        (n_pool, n_features) feature matrix for the *unlabeled* pool.
    uncertainty:
        (n_pool,) per-sample uncertainty in ``[0, 1]`` (e.g. from
        :func:`boundary_uncertainty` or :func:`entropy_uncertainty`).
    n_select:
        Number of samples to return (capped at the pool size).
    config:
        :class:`ActiveConfig`; defaults to ``ActiveConfig()``.
    candidate_indices:
        Optional restriction of the selectable pool (indices into ``features``).

    Returns
    -------
    list[int]:
        Indices (into the original ``features`` array) selected for labeling,
        in the order they were chosen (first = most uncertain seed).

    Algorithm
    ---------
    1. Optionally z-score features for the distance term.
    2. Seed with the single most-uncertain candidate.
    3. Repeatedly add the candidate maximizing
       ``alpha * uncertainty + (1 - alpha) * normalized_min_distance`` where the
       distance is to the already-selected set. Re-normalizing the distance
       term every round (by its current max) keeps the two objectives on the
       same ``[0, 1]`` scale as the selected set grows.
    """
    cfg = config or ActiveConfig()
    feats = np.asarray(features, dtype=np.float64)
    unc = np.asarray(uncertainty, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"features must be 2D, got {feats.shape}")
    if unc.shape[0] != feats.shape[0]:
        raise ValueError("uncertainty length must match number of samples")

    pool = (
        list(range(feats.shape[0]))
        if candidate_indices is None
        else list(candidate_indices)
    )
    if not pool:
        return []
    n_select = min(n_select, len(pool))

    work = _normalize(feats) if cfg.normalize_features else feats

    selected: List[int] = []
    remaining = set(pool)

    # --- Seed: the most uncertain candidate (ties -> lowest index). ---------
    seed = max(remaining, key=lambda i: (unc[i], -i))
    selected.append(seed)
    remaining.discard(seed)

    while len(selected) < n_select and remaining:
        rem = np.array(sorted(remaining))
        cand_feats = work[rem]
        sel_feats = work[np.array(selected)]

        dmin = _pairwise_min_distance(cand_feats, sel_feats)
        # Normalize the diversity term to [0, 1] for comparability with unc.
        dmax = float(dmin.max())
        div = dmin / dmax if dmax > 0 else np.zeros_like(dmin)

        gain = cfg.alpha * unc[rem] + (1.0 - cfg.alpha) * div
        best_local = int(np.argmax(gain))
        best = int(rem[best_local])

        selected.append(best)
        remaining.discard(best)

    return selected
