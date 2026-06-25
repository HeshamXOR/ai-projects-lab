"""Mahalanobis-distance anomaly scoring from defect-free samples.

WHY THIS MODULE EXISTS
----------------------
The cheapest way to get a useful defect detector is *one-class*: learn what
"good" product looks like in feature space, then flag anything far from it. We
model the defect-free feature vectors as a single multivariate Gaussian and
score a new sample by its **Mahalanobis distance** from that distribution::

    D^2(x) = (x - mu)^T * Sigma^{-1} * (x - mu)

Unlike Euclidean distance, Mahalanobis accounts for feature *scale* and
*correlation*: a deviation along a low-variance, tightly-coupled direction
counts for far more than the same deviation along a noisy, high-variance one.
That is exactly what we want -- a small but consistent contrast shift is more
suspicious than a large swing in an already-noisy feature.

THE NUMERICAL CRUX
------------------
Sigma is often singular or ill-conditioned: features can be correlated, and we
may have fewer "good" samples than feature dimensions. A raw inverse would
explode. We therefore **regularize**: ``Sigma_reg = Sigma + lambda * I`` (ridge
/ shrinkage toward a scaled identity). We then invert via Cholesky when
possible (fast, exploits symmetry/PD) and fall back to the pseudo-inverse if
even the regularized matrix is not positive-definite. All implemented with
``numpy.linalg`` primitives -- the distance math itself is ours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["AnomalyConfig", "MahalanobisAnomalyModel"]


@dataclass(frozen=True)
class AnomalyConfig:
    """Configuration for the Gaussian anomaly model.

    Attributes
    ----------
    ridge:
        Diagonal regularization added to the covariance, expressed as a
        fraction of the mean feature variance (relative ridge). Keeps Sigma
        invertible and stable when samples are scarce or features collinear.
    standardize:
        If True, features are z-scored (per-dimension) before fitting. This
        makes the ridge scale-invariant and improves conditioning.
    """

    ridge: float = 1e-2
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")


@dataclass
class MahalanobisAnomalyModel:
    """One-class Gaussian anomaly model scored by Mahalanobis distance.

    Fit on defect-free feature vectors with :meth:`fit`, then call
    :meth:`score` (squared distance) or :meth:`distance` (the square root) on
    new vectors. Higher = more anomalous.
    """

    config: AnomalyConfig = field(default_factory=AnomalyConfig)

    # Learned state (populated by ``fit``).
    mean_: np.ndarray = field(default=None, init=False)       # type: ignore
    scale_: np.ndarray = field(default=None, init=False)      # type: ignore
    cov_inv_: np.ndarray = field(default=None, init=False)    # type: ignore
    dim_: int = field(default=0, init=False)
    n_samples_: int = field(default=0, init=False)

    @property
    def is_fitted(self) -> bool:
        return self.cov_inv_ is not None

    def fit(self, features: np.ndarray) -> "MahalanobisAnomalyModel":
        """Estimate mean and regularized inverse-covariance from good samples.

        Parameters
        ----------
        features:
            (n_samples, n_features) array of defect-free feature vectors.

        Raises
        ------
        ValueError:
            If fewer than 2 samples are supplied (covariance undefined).
        """
        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")
        n, d = X.shape
        if n < 2:
            raise ValueError("need at least 2 samples to estimate covariance")

        self.dim_ = d
        self.n_samples_ = n
        self.mean_ = X.mean(axis=0)

        if self.config.standardize:
            scale = X.std(axis=0)
            scale[scale < 1e-12] = 1.0  # do not amplify constant features
            self.scale_ = scale
        else:
            self.scale_ = np.ones(d, dtype=np.float64)

        Z = (X - self.mean_) / self.scale_
        # Sample covariance with Bessel's correction.
        cov = np.cov(Z, rowvar=False, bias=False)
        cov = np.atleast_2d(cov)

        # Relative ridge: scale the identity by the average diagonal variance
        # so the regularization strength tracks the data scale.
        avg_var = float(np.mean(np.diag(cov))) if d > 0 else 1.0
        if avg_var <= 0:
            avg_var = 1.0
        lam = self.config.ridge * avg_var
        cov_reg = cov + lam * np.eye(d)

        self.cov_inv_ = self._stable_inverse(cov_reg)
        return self

    @staticmethod
    def _stable_inverse(matrix: np.ndarray) -> np.ndarray:
        """Invert a symmetric PD matrix via Cholesky; fall back to pinv.

        Cholesky (``L L^T = A``) is the numerically-preferred route for a
        symmetric positive-definite covariance: it is twice as fast as LU and
        signals non-PD inputs by raising ``LinAlgError``. If the regularized
        covariance is still not PD (extreme collinearity), we fall back to the
        Moore-Penrose pseudo-inverse, which always exists.
        """
        try:
            L = np.linalg.cholesky(matrix)
            inv_L = np.linalg.inv(L)
            return inv_L.T @ inv_L
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return squared Mahalanobis distance for one or many vectors.

        Accepts a single vector (shape ``(d,)``) or a batch
        (``(n, d)``) and returns a scalar array or 1D array respectively.
        """
        if not self.is_fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        X = np.asarray(features, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.shape[1] != self.dim_:
            raise ValueError(
                f"expected {self.dim_} features, got {X.shape[1]}"
            )
        Z = self._standardize(X)
        # Quadratic form per row: sum((Z @ cov_inv) * Z, axis=1).
        left = Z @ self.cov_inv_
        d2 = np.einsum("ij,ij->i", left, Z)
        d2 = np.clip(d2, 0.0, None)  # guard tiny negatives from round-off
        return d2[0] if single else d2

    def distance(self, features: np.ndarray) -> np.ndarray:
        """Mahalanobis distance (the square root of :meth:`score`)."""
        return np.sqrt(self.score(features))
