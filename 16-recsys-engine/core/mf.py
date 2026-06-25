"""Matrix factorization recommender, implemented from scratch in NumPy.

Model
-----
We learn a low-rank latent-factor model. For a user ``u`` and item ``i`` the
predicted rating is::

    r_hat(u, i) = mu + b_u[u] + b_i[i] + P[u] . Q[i]

where ``mu`` is the global mean rating, ``b_u`` / ``b_i`` are user / item bias
vectors, and ``P`` (``n_users x k``) / ``Q`` (``n_items x k``) are the latent
factor matrices.

Training (SGD)
--------------
We minimise the regularised squared error over the *observed* ratings::

    J = sum_{(u,i) in obs} ( r_ui - r_hat(u,i) )^2
        + lambda * ( ||P[u]||^2 + ||Q[i]||^2 + b_u[u]^2 + b_i[i]^2 )

For each observed rating we compute the error ``e = r_ui - r_hat`` and take a
gradient step (the SGD update derivation is in EXPLAINER.md)::

    b_u[u] += lr * (e - lambda * b_u[u])
    b_i[i] += lr * (e - lambda * b_i[i])
    P[u]   += lr * (e * Q[i] - lambda * P[u])
    Q[i]   += lr * (e * P[u] - lambda * Q[i])

We shuffle the observations each epoch and track train / validation RMSE.

An optional ALS variant (:meth:`MatrixFactorization.fit_als`) is provided for
comparison; SGD is the primary, required algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class MFConfig:
    """Hyper-parameters for the matrix factorization model."""

    n_factors: int = 8
    n_epochs: int = 40
    lr: float = 0.02
    reg: float = 0.05
    init_scale: float = 0.1
    seed: int = 0


class MatrixFactorization:
    """Latent-factor recommender trained with SGD (or ALS).

    Parameters
    ----------
    n_factors:
        Latent dimensionality ``k``.
    n_epochs:
        Number of full passes over the training observations (SGD).
    lr:
        SGD learning rate.
    reg:
        L2 regularisation strength ``lambda`` applied to factors and biases.
    init_scale:
        Std-dev of the Gaussian used to initialise ``P`` and ``Q``.
    seed:
        RNG seed for reproducible initialisation / shuffling.
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_epochs: int = 40,
        lr: float = 0.02,
        reg: float = 0.05,
        init_scale: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.config = MFConfig(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr=lr,
            reg=reg,
            init_scale=init_scale,
            seed=seed,
        )
        self.n_users: int = 0
        self.n_items: int = 0
        self.mu: float = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        # RMSE history captured during the last fit.
        self.train_rmse_: List[float] = []
        self.val_rmse_: List[float] = []
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _init_params(self, n_users: int, n_items: int, mu: float) -> None:
        rng = np.random.default_rng(self.config.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.mu = mu
        self.b_u = np.zeros(n_users, dtype=np.float64)
        self.b_i = np.zeros(n_items, dtype=np.float64)
        self.P = rng.normal(0.0, self.config.init_scale, size=(n_users, self.config.n_factors))
        self.Q = rng.normal(0.0, self.config.init_scale, size=(n_items, self.config.n_factors))

    @staticmethod
    def _rmse(errors: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(errors)))) if errors.size else 0.0

    def _predict_array(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised prediction for arrays of internal user/item indices."""
        assert self.P is not None and self.Q is not None
        assert self.b_u is not None and self.b_i is not None
        dot = np.sum(self.P[users] * self.Q[items], axis=1)
        return self.mu + self.b_u[users] + self.b_i[items] + dot

    def _residuals(self, triplets: np.ndarray) -> np.ndarray:
        users = triplets[:, 0].astype(int)
        items = triplets[:, 1].astype(int)
        preds = self._predict_array(users, items)
        return triplets[:, 2] - preds

    # ------------------------------------------------------------------ #
    # Training -- SGD (primary)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train: np.ndarray,
        n_users: int,
        n_items: int,
        val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "MatrixFactorization":
        """Train via stochastic gradient descent over observed ratings.

        Parameters
        ----------
        train:
            ``(n_obs, 3)`` array of ``(user_idx, item_idx, rating)``.
        n_users, n_items:
            Total counts (so factor matrices size correctly even if some
            ids are absent from ``train``).
        val:
            Optional validation triplets for RMSE tracking.
        verbose:
            Print per-epoch RMSE if True.

        Returns
        -------
        self
        """
        if train.ndim != 2 or train.shape[1] != 3:
            raise ValueError("train must be an (n_obs, 3) array")
        if train.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")

        mu = float(np.mean(train[:, 2]))
        self._init_params(n_users, n_items, mu)
        assert self.P is not None and self.Q is not None
        assert self.b_u is not None and self.b_i is not None

        rng = np.random.default_rng(self.config.seed + 1)
        lr = self.config.lr
        reg = self.config.reg
        n_obs = train.shape[0]
        order = np.arange(n_obs)

        self.train_rmse_ = []
        self.val_rmse_ = []

        for epoch in range(self.config.n_epochs):
            rng.shuffle(order)
            for idx in order:
                u = int(train[idx, 0])
                i = int(train[idx, 1])
                r = train[idx, 2]

                pred = (
                    self.mu
                    + self.b_u[u]
                    + self.b_i[i]
                    + float(self.P[u] @ self.Q[i])
                )
                err = r - pred

                # Cache the old factor rows; the P update uses Q (and vice
                # versa), so update simultaneously to keep the math correct.
                p_u = self.P[u].copy()
                q_i = self.Q[i].copy()

                self.b_u[u] += lr * (err - reg * self.b_u[u])
                self.b_i[i] += lr * (err - reg * self.b_i[i])
                self.P[u] += lr * (err * q_i - reg * p_u)
                self.Q[i] += lr * (err * p_u - reg * q_i)

            train_rmse = self._rmse(self._residuals(train))
            self.train_rmse_.append(train_rmse)
            if val is not None and val.shape[0] > 0:
                val_rmse = self._rmse(self._residuals(val))
                self.val_rmse_.append(val_rmse)
                if verbose:
                    print(f"epoch {epoch + 1:3d}  train RMSE {train_rmse:.4f}  val RMSE {val_rmse:.4f}")
            elif verbose:
                print(f"epoch {epoch + 1:3d}  train RMSE {train_rmse:.4f}")

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Training -- ALS (optional variant)
    # ------------------------------------------------------------------ #
    def fit_als(
        self,
        train: np.ndarray,
        n_users: int,
        n_items: int,
        n_iters: int = 15,
        val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "MatrixFactorization":
        """Train via Alternating Least Squares (optional comparison variant).

        Holds item factors fixed and solves the regularised normal equations
        for each user's factor row, then alternates. Biases are folded into
        the global mean only (kept simple); this variant exists mainly to
        contrast with SGD. SGD via :meth:`fit` remains the primary algorithm.
        """
        if train.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")

        mu = float(np.mean(train[:, 2]))
        self._init_params(n_users, n_items, mu)
        assert self.P is not None and self.Q is not None
        assert self.b_u is not None and self.b_i is not None
        k = self.config.n_factors
        reg = self.config.reg
        lam_eye = reg * np.eye(k)

        # Bucket observations by user and by item once.
        users = train[:, 0].astype(int)
        items = train[:, 1].astype(int)
        resid = train[:, 2] - mu  # bias-free target for ALS

        by_user = [[] for _ in range(n_users)]
        by_item = [[] for _ in range(n_items)]
        for row in range(train.shape[0]):
            by_user[users[row]].append((items[row], resid[row]))
            by_item[items[row]].append((users[row], resid[row]))

        self.train_rmse_ = []
        self.val_rmse_ = []

        for it in range(n_iters):
            # Solve for each user's factors with item factors fixed.
            for u in range(n_users):
                if not by_user[u]:
                    continue
                idx = np.array([t[0] for t in by_user[u]], dtype=int)
                tgt = np.array([t[1] for t in by_user[u]], dtype=np.float64)
                Qsub = self.Q[idx]
                A = Qsub.T @ Qsub + lam_eye
                b = Qsub.T @ tgt
                self.P[u] = np.linalg.solve(A, b)

            # Solve for each item's factors with user factors fixed.
            for i in range(n_items):
                if not by_item[i]:
                    continue
                idx = np.array([t[0] for t in by_item[i]], dtype=int)
                tgt = np.array([t[1] for t in by_item[i]], dtype=np.float64)
                Psub = self.P[idx]
                A = Psub.T @ Psub + lam_eye
                b = Psub.T @ tgt
                self.Q[i] = np.linalg.solve(A, b)

            train_rmse = self._rmse(self._residuals(train))
            self.train_rmse_.append(train_rmse)
            if val is not None and val.shape[0] > 0:
                self.val_rmse_.append(self._rmse(self._residuals(val)))
            if verbose:
                print(f"als iter {it + 1:3d}  train RMSE {train_rmse:.4f}")

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predicted rating for a single (internal) user/item pair."""
        self._check_fitted()
        assert self.P is not None and self.Q is not None
        assert self.b_u is not None and self.b_i is not None
        return float(
            self.mu
            + self.b_u[user_idx]
            + self.b_i[item_idx]
            + self.P[user_idx] @ self.Q[item_idx]
        )

    def scores_for_user(self, user_idx: int) -> np.ndarray:
        """Predicted ratings for *all* items for one user (length n_items)."""
        self._check_fitted()
        assert self.P is not None and self.Q is not None
        assert self.b_u is not None and self.b_i is not None
        dots = self.Q @ self.P[user_idx]
        return self.mu + self.b_u[user_idx] + self.b_i + dots

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """Top-``k`` (item_idx, score) recommendations for a user.

        Parameters
        ----------
        user_idx:
            Internal user index.
        k:
            Number of items to return.
        exclude:
            Internal item indices to filter out (e.g. already-rated items).
        """
        scores = self.scores_for_user(user_idx)
        if exclude is not None and len(exclude) > 0:
            scores = scores.copy()
            scores[np.asarray(exclude, dtype=int)] = -np.inf
        k = min(k, int(np.sum(np.isfinite(scores))))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() first")
