"""Recall test: HNSW approximate search vs brute-force exact NN.

This is the load-bearing proof that the from-scratch HNSW graph actually
works. We build an index over random vectors, query it, and compare the
returned neighbor sets against the exact answer from a full scan. We assert
that recall@10 clears a conservative threshold.
"""

from __future__ import annotations

import numpy as np

from core.ann import HNSWIndex, brute_force_search


def _recall_at_k(approx_ids, exact_ids, k):
    """Fraction of the exact top-k that the approximate search recovered."""
    exact_set = set(exact_ids[:k])
    approx_set = set(approx_ids[:k])
    if not exact_set:
        return 1.0
    return len(exact_set & approx_set) / len(exact_set)


def test_hnsw_recall_vs_bruteforce(rng):
    """HNSW recall@10 should be high on random data."""
    n, dim, k = 1000, 48, 10
    data = rng.standard_normal((n, dim)).astype(np.float32)

    index = HNSWIndex(dim=dim, M=16, ef_construction=200, ef_search=128, seed=7)
    for row in data:
        index.add(row)

    assert index.size == n

    n_queries = 40
    recalls = []
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    for q in queries:
        approx = index.search(q, k=k)
        approx_ids = [idx for idx, _sim in approx]
        exact = brute_force_search(data, q, k=k)
        exact_ids = [idx for idx, _sim in exact]
        recalls.append(_recall_at_k(approx_ids, exact_ids, k))

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.7, f"recall@{k} too low: {mean_recall:.3f}"


def test_hnsw_returns_self_as_nearest(rng):
    """A stored vector queried back should rank itself first."""
    dim = 32
    data = rng.standard_normal((200, dim)).astype(np.float32)
    index = HNSWIndex(dim=dim, ef_search=64, seed=3)
    for row in data:
        index.add(row)

    # Query with an exact stored vector; cosine sim to itself is ~1.0.
    target = 57
    results = index.search(data[target], k=5)
    assert results, "search returned nothing"
    top_idx, top_sim = results[0]
    assert top_idx == target
    assert top_sim > 0.99


def test_hnsw_empty_and_small(rng):
    """Edge cases: empty index returns nothing; single element works."""
    index = HNSWIndex(dim=8, seed=1)
    assert index.search(np.ones(8, dtype=np.float32), k=3) == []

    index.add(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    res = index.search(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), k=3)
    assert len(res) == 1
    assert res[0][0] == 0
