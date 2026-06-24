"""Reciprocal Rank Fusion (RRF) — combine multiple ranked lists into one.

Hybrid search runs two retrievers (semantic vector search + BM25 keyword
search) that return results on different score scales. RRF sidesteps the
scale-mismatch problem by ignoring the raw scores and using only *rank*:

    RRF(d) = sum over retrievers of  1 / (k + rank_i(d))

A document ranked highly by several retrievers rises to the top. It's simple,
parameter-light (just `k`, conventionally 60), and consistently beats either
retriever alone — which is why production RAG systems use it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]], k: int = 60, top_n: int = 5
) -> List[Tuple[int, float]]:
    """Fuse ranked lists of (doc_id, score) into one (doc_id, rrf_score) list.

    Only the *order* within each list is used, not the scores.
    """
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda x: -x[1])
    return out[:top_n]
