"""Offline evaluation harness, with ranking metrics implemented from scratch.

Metrics (all computed by hand, no library metric functions)
-----------------------------------------------------------
Given a ranked list of recommended item ids and a set of *relevant* (held-out,
liked) items for a user:

- **Precision@K** = (# relevant items in the top K) / K
- **Recall@K**    = (# relevant items in the top K) / (# relevant items)
- **NDCG@K**      = DCG@K / IDCG@K, where

      DCG@K  = sum_{p=1..K} rel_p / log2(p + 1)
      IDCG@K = the DCG of the ideal ordering (all relevant items ranked first)

  with ``rel_p`` the relevance (1 for a held-out liked item, else 0). NDCG
  rewards placing relevant items higher in the ranking.

Protocol
--------
:func:`leave_one_out_split` holds out, for each user, their single
highest-rated observed item (ties broken deterministically) as the relevance
target, leaving the rest for training. :func:`evaluate_model` trains-agnostic:
it just asks a fitted model to rank items for each user, excludes the user's
training items, and averages the metrics over users with a held-out target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def precision_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    """Precision@K -- fraction of the top ``k`` recommendations that are relevant."""
    if k <= 0:
        return 0.0
    topk = list(ranked)[:k]
    if not topk:
        return 0.0
    hits = sum(1 for it in topk if it in relevant)
    return hits / float(k)


def recall_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    """Recall@K -- fraction of relevant items captured in the top ``k``."""
    if not relevant:
        return 0.0
    topk = list(ranked)[:k]
    hits = sum(1 for it in topk if it in relevant)
    return hits / float(len(relevant))


def dcg_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    """Discounted cumulative gain with binary relevance.

    ``DCG@K = sum_{p=1..K} rel_p / log2(p + 1)`` with positions ``p`` 1-indexed.
    """
    topk = list(ranked)[:k]
    dcg = 0.0
    for pos, item in enumerate(topk, start=1):
        rel = 1.0 if item in relevant else 0.0
        if rel:
            dcg += rel / np.log2(pos + 1.0)
    return float(dcg)


def ndcg_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    """Normalised DCG@K = DCG@K / IDCG@K (0 when no relevant items)."""
    if not relevant or k <= 0:
        return 0.0
    dcg = dcg_at_k(ranked, relevant, k)
    # Ideal DCG: as many relevant items as possible packed into the top slots.
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(pos + 1.0) for pos in range(1, n_rel + 1))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def leave_one_out_split(
    ratings: np.ndarray,
    n_users: int,
    like_threshold: Optional[float] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Hold out one item per user as the relevance target.

    For each user we hold out their highest-rated observed item (ties broken
    by a deterministic shuffle), provided the user has at least two ratings so
    something remains for training.

    Parameters
    ----------
    ratings:
        ``(n_obs, 3)`` array of ``(user_idx, item_idx, rating)``.
    n_users:
        Total number of users.
    like_threshold:
        If set, only hold out items with rating >= this value (so the target
        is genuinely "liked"). Users with no qualifying item are skipped.
    seed:
        RNG seed for deterministic tie-breaking.

    Returns
    -------
    train:
        Training triplets (held-out rows removed).
    heldout:
        Map ``user_idx -> held-out item_idx``.
    """
    rng = np.random.default_rng(seed)
    keep_mask = np.ones(ratings.shape[0], dtype=bool)
    heldout: Dict[int, int] = {}

    users = ratings[:, 0].astype(int)
    for u in range(n_users):
        rows = np.where(users == u)[0]
        if rows.size < 2:
            continue  # need >=2 so training keeps at least one item
        cand = rows
        if like_threshold is not None:
            liked = rows[ratings[rows, 2] >= like_threshold]
            if liked.size == 0:
                continue
            cand = liked
        # Shuffle candidate order for deterministic tie-breaking, then pick
        # the highest-rated.
        perm = rng.permutation(cand.size)
        cand = cand[perm]
        best = cand[int(np.argmax(ratings[cand, 2]))]
        heldout[u] = int(ratings[best, 1])
        keep_mask[best] = False

    return ratings[keep_mask], heldout


# --------------------------------------------------------------------------- #
# Model protocol + evaluation
# --------------------------------------------------------------------------- #
class Recommender(Protocol):
    """Minimal interface the harness needs from a model."""

    def recommend(
        self, user_idx: int, k: int, exclude: Optional[np.ndarray]
    ) -> List[Tuple[int, float]]:
        ...


@dataclass
class EvalResult:
    """Averaged ranking metrics across evaluated users."""

    k: int
    precision: float
    recall: float
    ndcg: float
    n_users_evaluated: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "k": self.k,
            "precision": self.precision,
            "recall": self.recall,
            "ndcg": self.ndcg,
            "n_users_evaluated": self.n_users_evaluated,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"K={self.k}  P@K={self.precision:.4f}  R@K={self.recall:.4f}  "
            f"NDCG@K={self.ndcg:.4f}  (n={self.n_users_evaluated})"
        )


def evaluate_model(
    model: Recommender,
    train: np.ndarray,
    heldout: Dict[int, int],
    k: int = 10,
    candidate_pool: Optional[np.ndarray] = None,
) -> EvalResult:
    """Average Precision/Recall/NDCG@K of a fitted model over held-out targets.

    For each user with a held-out item the model ranks all items, the user's
    *training* items are excluded (so we never recommend already-seen items),
    and the held-out item is the sole relevant target.

    Parameters
    ----------
    model:
        A fitted recommender exposing ``recommend(user_idx, k, exclude)``.
    train:
        Training triplets (used to know which items to exclude per user).
    heldout:
        Map ``user_idx -> held-out item_idx`` from :func:`leave_one_out_split`.
    k:
        Cut-off for the metrics.
    candidate_pool:
        Optional restriction of recommendable items (unused by default).

    Returns
    -------
    EvalResult
    """
    users = train[:, 0].astype(int)
    items = train[:, 1].astype(int)

    # Pre-bucket each user's training items for the exclusion list.
    seen: Dict[int, List[int]] = {}
    for row in range(train.shape[0]):
        seen.setdefault(int(users[row]), []).append(int(items[row]))

    p_sum = r_sum = n_sum = 0.0
    evaluated = 0

    for u, target in heldout.items():
        exclude = np.array(seen.get(u, []), dtype=int)
        recs = model.recommend(u, k=k, exclude=exclude)
        ranked = [item for item, _score in recs]
        relevant = {int(target)}

        p_sum += precision_at_k(ranked, relevant, k)
        r_sum += recall_at_k(ranked, relevant, k)
        n_sum += ndcg_at_k(ranked, relevant, k)
        evaluated += 1

    if evaluated == 0:
        return EvalResult(k=k, precision=0.0, recall=0.0, ndcg=0.0, n_users_evaluated=0)

    return EvalResult(
        k=k,
        precision=p_sum / evaluated,
        recall=r_sum / evaluated,
        ndcg=n_sum / evaluated,
        n_users_evaluated=evaluated,
    )
