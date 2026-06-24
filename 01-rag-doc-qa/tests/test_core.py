"""Correctness proof for the from-scratch retrieval core.

The key claim: HNSW returns *almost* the same neighbors as exact brute-force
search, but far faster. We assert recall@10 >= 0.90 against ground truth on a
fixed random dataset — the standard way ANN indexes are validated.
"""

import numpy as np
import pytest

from core.hnsw import HNSW, brute_force_knn
from core.bm25 import BM25
from core.fusion import reciprocal_rank_fusion


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((500, 32))


def test_hnsw_recall_vs_brute_force(data):
    index = HNSW(dim=32, M=16, ef_construction=200, ef_search=64, seed=1)
    index.add_batch(data)
    assert len(index) == 500

    rng = np.random.default_rng(99)
    k = 10
    recalls = []
    for _ in range(30):
        q = rng.standard_normal(32)
        truth = {i for i, _ in brute_force_knn(data, q, k)}
        approx = {i for i, _ in index.search(q, k)}
        recalls.append(len(truth & approx) / k)
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.90, f"recall too low: {mean_recall:.3f}"


def test_hnsw_finds_exact_self(data):
    # querying with an indexed vector should return itself as the top hit
    index = HNSW(dim=32, ef_search=64, seed=2)
    index.add_batch(data)
    for i in [0, 100, 499]:
        top_id, dist = index.search(data[i], k=1)[0]
        assert top_id == i
        assert dist < 1e-6


def test_bm25_ranks_relevant_doc_first():
    docs = [
        "the cat sat on the mat",
        "supply chain risk and logistics disruption",
        "quarterly revenue grew on strong cloud demand",
        "a dog ran across the field",
    ]
    bm = BM25()
    bm.index(docs)
    res = bm.search("revenue cloud growth", k=2)
    assert res[0][0] == 2  # the revenue/cloud doc ranks first


def test_rrf_rewards_agreement():
    # doc 7 is ranked highly by both lists -> should win after fusion
    vec = [(7, 0.9), (3, 0.8), (1, 0.5)]
    kw = [(7, 12.0), (1, 9.0), (9, 4.0)]
    fused = reciprocal_rank_fusion([vec, kw], k=60, top_n=3)
    assert fused[0][0] == 7
