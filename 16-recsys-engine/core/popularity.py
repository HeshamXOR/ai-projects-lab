"""Popularity baseline recommender.

A non-personalised baseline that ranks items by a *shrinkage-adjusted* mean
rating. Raw mean ratings are unreliable for items with few observations (an
item rated 5.0 by a single user looks "better" than one rated 4.6 by fifty
users). We correct this with a Bayesian shrinkage prior toward the global
mean::

    score(i) = ( n_i * mean_i + m * mu ) / ( n_i + m )

where ``n_i`` is the rating count for item ``i``, ``mean_i`` its mean rating,
``mu`` the global mean, and ``m`` the shrinkage strength (a pseudo-count of
prior observations). As ``n_i`` grows the score tends to ``mean_i``; for
sparse items it is pulled toward ``mu``. This is the same idea behind IMDb's
weighted-rating formula.

This model is also the cold-start fallback for unknown users in the API.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class PopularityRecommender:
    """Rank items by shrinkage-adjusted mean rating.

    Parameters
    ----------
    shrinkage:
        Pseudo-count ``m`` of prior observations. Larger values pull
        low-count items more strongly toward the global mean.
    """

    def __init__(self, shrinkage: float = 5.0) -> None:
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        self.shrinkage = float(shrinkage)
        self.n_items: int = 0
        self.global_mean: float = 0.0
        self.counts_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, train: np.ndarray, n_items: int) -> "PopularityRecommender":
        """Compute per-item counts, means and shrinkage scores.

        Parameters
        ----------
        train:
            ``(n_obs, 3)`` array of ``(user_idx, item_idx, rating)``.
        n_items:
            Total number of items (so unrated items still get a score
            equal to the global mean prior).
        """
        if train.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")

        self.n_items = n_items
        self.global_mean = float(np.mean(train[:, 2]))

        items = train[:, 1].astype(int)
        ratings = train[:, 2]

        counts = np.zeros(n_items, dtype=np.float64)
        sums = np.zeros(n_items, dtype=np.float64)
        np.add.at(counts, items, 1.0)
        np.add.at(sums, items, ratings)

        means = np.divide(
            sums,
            counts,
            out=np.full(n_items, self.global_mean),
            where=counts > 0,
        )

        m = self.shrinkage
        scores = (counts * means + m * self.global_mean) / (counts + m)

        self.counts_ = counts
        self.means_ = means
        self.scores_ = scores
        self._fitted = True
        return self

    def scores_for_user(self, user_idx: Optional[int] = None) -> np.ndarray:
        """Return the (user-independent) item scores.

        The ``user_idx`` argument is accepted for interface symmetry with the
        personalised models but ignored -- popularity is global.
        """
        self._check_fitted()
        assert self.scores_ is not None
        return self.scores_.copy()

    def recommend(
        self,
        user_idx: Optional[int] = None,
        k: int = 10,
        exclude: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """Top-``k`` (item_idx, score) popular items, optionally excluding some."""
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

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() first")
