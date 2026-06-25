"""Reciprocal-rank fusion (RRF) for combining ranked result lists.

RRF is a simple, robust, score-agnostic way to merge several ranked lists
into one. It ignores the raw scores -- which live on incomparable scales
for a cosine-similarity ANN list and a BM25 list -- and instead uses only
the *rank position* of each item in each list.

For an item :math:`d` appearing at rank :math:`r_\\ell(d)` (1-indexed) in
list :math:`\\ell`, the fused score is::

    RRF(d) = sum_ell  weight_ell / (k + r_ell(d))

The constant :math:`k` (commonly 60) damps the influence of very high ranks
so that an item ranked #1 in one list does not completely dominate an item
that appears respectably in several lists. Items missing from a list simply
contribute nothing for that list.

This module implements RRF from scratch with optional per-list weights.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

# A ranked list is an ordered sequence of (id, score) pairs. Only the order
# matters for RRF; the scores are accepted for convenience and ignored.
RankedList = Sequence[Tuple[str, float]]


def reciprocal_rank_fusion(
    result_lists: Iterable[RankedList],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> List[Tuple[str, float]]:
    """Fuse several ranked lists into one using reciprocal-rank fusion.

    Args:
        result_lists: An iterable of ranked lists. Each ranked list is a
            sequence of ``(id, score)`` pairs already sorted best-first.
            Scores are ignored; only positions are used.
        k: The RRF damping constant. Larger values flatten the contribution
            of top ranks. Must be positive.
        weights: Optional per-list weights, one per input list. Defaults to
            all-ones (equal weighting).

    Returns:
        A list of ``(id, fused_score)`` pairs sorted by descending fused
        score. Ties are broken by ``id`` for deterministic output.

    Raises:
        ValueError: If ``k`` is not positive or ``weights`` length mismatches.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    lists = [list(rl) for rl in result_lists]
    if weights is None:
        weights = [1.0] * len(lists)
    elif len(weights) != len(lists):
        raise ValueError(
            f"weights length {len(weights)} != number of lists {len(lists)}"
        )

    fused: Dict[str, float] = {}
    for weight, ranked in zip(weights, lists):
        for position, (doc_id, _score) in enumerate(ranked):
            rank = position + 1  # 1-indexed
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)

    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered


def weighted_score_fusion(
    result_lists: Sequence[RankedList],
    weights: Sequence[float] | None = None,
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """Alternative fusion that blends *normalized scores* instead of ranks.

    This is provided as a comparison baseline to RRF. Each list's scores are
    min-max normalized to ``[0, 1]`` (so the different scales become
    comparable) and then combined with the given weights.

    Args:
        result_lists: Ranked lists of ``(id, score)`` pairs.
        weights: Optional per-list weights. Defaults to all-ones.
        normalize: If True, min-max normalize each list's scores first.

    Returns:
        A list of ``(id, fused_score)`` pairs sorted descending.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    elif len(weights) != len(result_lists):
        raise ValueError("weights length must match number of lists")

    fused: Dict[str, float] = {}
    for weight, ranked in zip(weights, result_lists):
        if not ranked:
            continue
        scores = [s for _, s in ranked]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for doc_id, score in ranked:
            norm = (score - lo) / span if normalize else score
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * norm

    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
