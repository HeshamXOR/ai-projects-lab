"""Item-item cosine kNN recommender, implemented from scratch in NumPy.

Idea
----
Build an item representation directly from the user-item rating matrix: each
item is the column vector of ratings it received across users. Two items are
"similar" if the users who rated them rated them in similar ways -- measured by
cosine similarity between the (mean-centred) column vectors::

    sim(i, j) = ( v_i . v_j ) / ( ||v_i|| * ||v_j|| )

To predict how much user ``u`` would like a candidate item ``i`` we take the
``k`` most-similar items that ``u`` has *already rated* and form a
similarity-weighted average of those ratings::

    score(u, i) = sum_{j in N_k(i) & rated by u} sim(i, j) * r_uj
                  / sum_{j in N_k(i) & rated by u} |sim(i, j)|

Mean-centring each item column before computing cosine removes per-item
popularity bias so the similarity reflects co-rating *pattern*, not absolute
level. We optionally fall back to the item's mean rating when a candidate has
no rated neighbours, so cold items still get a sensible (if generic) score.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class ItemKNN:
    """Item-item collaborative filtering via cosine similarity.

    Parameters
    ----------
    k_neighbors:
        Number of nearest item-neighbours to aggregate over when scoring.
    shrinkage:
        Optional significance weighting. Similarities are multiplied by
        ``co / (co + shrinkage)`` where ``co`` is the number of users who
        co-rated the item pair, damping similarities supported by little
        overlap. Set to 0 to disable.
    min_sim:
        Neighbours with similarity at or below this are ignored.
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        shrinkage: float = 10.0,
        min_sim: float = 0.0,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.shrinkage = float(shrinkage)
        self.min_sim = float(min_sim)
        self.n_users: int = 0
        self.n_items: int = 0
        self.sim_: Optional[np.ndarray] = None
        self.item_means_: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None       # raw user-item matrix
        self._rated: Optional[np.ndarray] = None    # boolean mask of observed
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, train: np.ndarray, n_users: int, n_items: int) -> "ItemKNN":
        """Build the item-item similarity matrix from training ratings.

        Parameters
        ----------
        train:
            ``(n_obs, 3)`` array of ``(user_idx, item_idx, rating)``.
        n_users, n_items:
            Total user / item counts.
        """
        if train.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")

        self.n_users = n_users
        self.n_items = n_items

        R = np.zeros((n_users, n_items), dtype=np.float64)
        rated = np.zeros((n_users, n_items), dtype=bool)
        u = train[:, 0].astype(int)
        i = train[:, 1].astype(int)
        R[u, i] = train[:, 2]
        rated[u, i] = True

        # Per-item mean over observed entries only.
        counts = rated.sum(axis=0)
        sums = R.sum(axis=0)
        item_means = np.divide(
            sums, counts, out=np.zeros(n_items), where=counts > 0
        )

        # Mean-centre observed entries; unobserved stay 0 (neutral).
        centred = np.where(rated, R - item_means[np.newaxis, :], 0.0)

        # Cosine similarity between item columns.
        norms = np.linalg.norm(centred, axis=0)
        # Avoid division by zero for items with no variance / no ratings.
        safe_norms = np.where(norms > 0, norms, 1.0)
        normed = centred / safe_norms[np.newaxis, :]
        sim = normed.T @ normed  # (n_items, n_items), cosine in [-1, 1]

        # Significance weighting by co-rating support.
        if self.shrinkage > 0:
            corate = (rated.T.astype(np.float64)) @ rated.astype(np.float64)
            sim = sim * (corate / (corate + self.shrinkage))

        np.fill_diagonal(sim, 0.0)
        # Items with no ratings have undefined similarity -> zero them out.
        zero_items = counts == 0
        if np.any(zero_items):
            sim[zero_items, :] = 0.0
            sim[:, zero_items] = 0.0

        self.sim_ = sim
        self.item_means_ = item_means
        self._R = R
        self._rated = rated
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def neighbors(self, item_idx: int, top: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return the most similar items to ``item_idx`` as (item_idx, sim).

        Used both for scoring and for introspection / tests.
        """
        self._check_fitted()
        assert self.sim_ is not None
        sims = self.sim_[item_idx]
        top = self.k_neighbors if top is None else top
        order = np.argsort(-sims)
        out: List[Tuple[int, float]] = []
        for j in order:
            if len(out) >= top:
                break
            if j == item_idx:
                continue
            if sims[j] <= self.min_sim:
                break
            out.append((int(j), float(sims[j])))
        return out

    def _score_user(self, user_idx: int) -> np.ndarray:
        """Predicted scores for all items for a single user."""
        assert self.sim_ is not None and self._R is not None
        assert self._rated is not None and self.item_means_ is not None

        rated_mask = self._rated[user_idx]
        rated_items = np.where(rated_mask)[0]
        scores = self.item_means_.copy()  # fallback baseline per item

        if rated_items.size == 0:
            return scores

        user_ratings = self._R[user_idx, rated_items]

        for target in range(self.n_items):
            sims_to_rated = self.sim_[target, rated_items]
            # Keep only the top-k positive neighbours among rated items.
            valid = sims_to_rated > self.min_sim
            if not np.any(valid):
                continue
            s = sims_to_rated[valid]
            r = user_ratings[valid]
            if s.size > self.k_neighbors:
                keep = np.argpartition(-s, self.k_neighbors - 1)[: self.k_neighbors]
                s = s[keep]
                r = r[keep]
            denom = np.sum(np.abs(s))
            if denom > 0:
                scores[target] = float(np.dot(s, r) / denom)
        return scores

    def scores_for_user(self, user_idx: int) -> np.ndarray:
        """Predicted scores for all items for the given user."""
        self._check_fitted()
        return self._score_user(user_idx)

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """Top-``k`` (item_idx, score) recommendations for a user."""
        scores = self.scores_for_user(user_idx)
        if exclude is not None and len(exclude) > 0:
            scores = scores.copy()
            scores[np.asarray(exclude, dtype=int)] = -np.inf
        finite = int(np.sum(np.isfinite(scores)))
        k = min(k, finite)
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() first")
