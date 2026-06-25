"""Online and batch anomaly detectors, implemented from scratch in NumPy.

Two detectors with complementary strengths:

1. :class:`EWMADetector` — a streaming, O(1)-per-sample z-score detector. It
   keeps an *exponentially weighted* running mean and variance (an EWMA analogue
   of Welford's online moments) so recent behaviour dominates and the detector
   adapts to slow regime changes. Each new point is scored by its standardised
   deviation ``z = (x - mean) / std``; ``|z|`` over a threshold ⇒ anomaly.

2. :class:`IsolationForestScorer` — a simplified Isolation Forest. The intuition
   is that anomalies are *few and different*, so a random axis-aligned partition
   isolates them in fewer splits. We build several trees over a window, each
   recursively choosing a random feature and a random split value; the anomaly
   score for a point is its average path length across trees, normalised by the
   expected path length ``c(n)`` of an unsuccessful BST search, mapped through
   the canonical ``2^(-E[h]/c(n))`` transform so scores live in ``(0, 1)`` with
   ``> 0.5`` indicating likely anomalies.

Everything — the incremental moment updates, the tree construction, and the
path-length accounting — is written directly against NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Streaming EWMA / z-score detector
# --------------------------------------------------------------------------- #
@dataclass
class EWMADetector:
    """Exponentially weighted streaming z-score anomaly detector.

    Parameters
    ----------
    alpha:
        Smoothing factor in ``(0, 1]``. Larger ⇒ more weight on recent samples
        (faster adaptation, shorter memory).
    z_threshold:
        ``|z|`` above this flags an anomaly.
    warmup:
        Number of initial samples to learn statistics on before scoring (their
        ``z`` is reported but never flagged), avoiding spurious early alerts.

    Notes
    -----
    The update equations (EWMA mean + EWMA variance) are::

        delta   = x - mean
        mean    = mean + alpha * delta
        var     = (1 - alpha) * (var + alpha * delta**2)

    The variance form is the standard exponentially weighted incremental
    estimator; ``std = sqrt(var)`` standardises the next deviation.
    """

    alpha: float = 0.1
    z_threshold: float = 3.0
    warmup: int = 10

    mean: float = field(default=0.0)
    var: float = field(default=0.0)
    count: int = field(default=0)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

    def update(self, x: float) -> "ScorePoint":
        """Ingest one sample, update state, and return its score.

        The point is scored against the statistics learned *before* it was seen,
        so the score is an honest out-of-sample deviation.
        """
        x = float(x)
        self.count += 1

        if self.count == 1:
            # Initialise on the first sample; cannot score it yet.
            self.mean = x
            self.var = 0.0
            return ScorePoint(value=x, z=0.0, is_anomaly=False)

        std = float(np.sqrt(self.var)) if self.var > 0 else 0.0
        z = (x - self.mean) / std if std > 1e-12 else 0.0

        # Incremental EWMA mean + variance update.
        delta = x - self.mean
        self.mean = self.mean + self.alpha * delta
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta)

        is_anomaly = self.count > self.warmup and abs(z) > self.z_threshold
        return ScorePoint(value=x, z=float(z), is_anomaly=bool(is_anomaly))

    def update_many(self, xs: np.ndarray | List[float]) -> List["ScorePoint"]:
        """Stream a batch of samples in order, returning a score per sample."""
        return [self.update(float(x)) for x in np.asarray(xs, dtype=float).ravel()]


@dataclass
class ScorePoint:
    """Per-sample output of the streaming detector."""

    value: float
    z: float
    is_anomaly: bool

    def as_dict(self) -> dict:
        return {"value": self.value, "z": self.z, "is_anomaly": self.is_anomaly}


# --------------------------------------------------------------------------- #
# Simplified Isolation Forest
# --------------------------------------------------------------------------- #
def _harmonic(n: int) -> float:
    """Approximate the ``n``-th harmonic number ``H_n``."""
    if n <= 1:
        return 0.0
    # Euler-Mascheroni based approximation (exact enough for path-length use).
    return float(np.log(n - 1) + 0.5772156649)


def expected_path_length(n: int) -> float:
    """``c(n)`` — average path length of an unsuccessful BST search of ``n`` points.

    This is the normalisation constant in the Isolation Forest score::

        c(n) = 2 * H_{n-1} - 2 * (n - 1) / n
    """
    if n <= 1:
        return 1.0
    return 2.0 * _harmonic(n) - 2.0 * (n - 1) / n


class _ITreeNode:
    """A node in a single isolation tree."""

    __slots__ = ("feature", "split", "left", "right", "size", "is_leaf")

    def __init__(self) -> None:
        self.feature: int = -1
        self.split: float = 0.0
        self.left: Optional["_ITreeNode"] = None
        self.right: Optional["_ITreeNode"] = None
        self.size: int = 0
        self.is_leaf: bool = False


class _IsolationTree:
    """One isolation tree over a (sub)sample, built with random axis splits."""

    def __init__(self, height_limit: int, rng: np.random.Generator) -> None:
        self.height_limit = height_limit
        self.rng = rng
        self.root: Optional[_ITreeNode] = None

    def fit(self, X: np.ndarray) -> "_IsolationTree":
        self.root = self._grow(X, current_height=0)
        return self

    def _grow(self, X: np.ndarray, current_height: int) -> _ITreeNode:
        node = _ITreeNode()
        n = X.shape[0]
        # Stop: reached the height cap, or the node can't be split further.
        if current_height >= self.height_limit or n <= 1:
            node.is_leaf = True
            node.size = n
            return node

        n_features = X.shape[1]
        feature = int(self.rng.integers(0, n_features))
        col = X[:, feature]
        cmin, cmax = float(col.min()), float(col.max())
        if cmin == cmax:
            node.is_leaf = True
            node.size = n
            return node

        split = float(self.rng.uniform(cmin, cmax))
        left_mask = col < split
        node.feature = feature
        node.split = split
        node.left = self._grow(X[left_mask], current_height + 1)
        node.right = self._grow(X[~left_mask], current_height + 1)
        return node

    def path_length(self, x: np.ndarray) -> float:
        """Path length to isolate ``x``, plus the ``c(size)`` leaf correction."""
        node = self.root
        depth = 0
        while node is not None and not node.is_leaf:
            if x[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
            depth += 1
        # Add expected remaining depth for the points that share the leaf.
        leaf_size = node.size if node is not None else 1
        return depth + expected_path_length(leaf_size)


@dataclass
class IsolationForestScorer:
    """A simplified Isolation Forest anomaly scorer (from scratch).

    Parameters
    ----------
    n_trees:
        Number of isolation trees in the ensemble.
    sample_size:
        Sub-sample size per tree (``psi`` in the original paper). The height
        limit is set to ``ceil(log2(sample_size))``.
    random_state:
        Seed for reproducible random splits.
    """

    n_trees: int = 100
    sample_size: int = 256
    random_state: Optional[int] = None

    trees: List[_IsolationTree] = field(default_factory=list)
    _c: float = field(default=1.0)
    _fitted: bool = field(default=False)

    def fit(self, X: np.ndarray | List[float]) -> "IsolationForestScorer":
        """Build the ensemble over the training window ``X``.

        ``X`` may be 1-D (single metric) or 2-D ``(n_samples, n_features)``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        if n == 0:
            raise ValueError("cannot fit on an empty window")

        rng = np.random.default_rng(self.random_state)
        sample_size = min(self.sample_size, n)
        height_limit = max(1, int(np.ceil(np.log2(max(sample_size, 2)))))
        self._c = expected_path_length(sample_size)

        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=sample_size, replace=False)
            tree = _IsolationTree(height_limit, rng).fit(X[idx])
            self.trees.append(tree)
        self._fitted = True
        return self

    def score(self, X: np.ndarray | List[float]) -> np.ndarray:
        """Anomaly score in ``(0, 1)`` per row; higher ⇒ more anomalous.

        ``score = 2^(-E[h(x)] / c(n))`` where ``E[h(x)]`` is the mean path length
        across trees. Scores near 1 isolate quickly (anomalies); near 0.5 are
        normal; near 0 are deep in dense regions.
        """
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scores = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            mean_path = np.mean([t.path_length(x) for t in self.trees])
            scores[i] = 2.0 ** (-mean_path / self._c) if self._c > 0 else 0.5
        return scores
