"""Benchmark: from-scratch HNSW vs. exact brute force (and FAISS if installed).

Run on Lightning to produce the numbers for the README:
    python bench.py
Reports build time, query latency, speedup, and recall@k against ground truth.
"""

from __future__ import annotations

import time

import numpy as np

from core.hnsw import HNSW, brute_force_knn


def run(n=5000, dim=64, k=10, queries=200, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim))
    qs = rng.standard_normal((queries, dim))

    # build
    t0 = time.perf_counter()
    index = HNSW(dim=dim, M=16, ef_construction=200, ef_search=64, seed=1)
    index.add_batch(data)
    build_s = time.perf_counter() - t0

    # ground truth (exact)
    truth = [set(i for i, _ in brute_force_knn(data, q, k)) for q in qs]

    # brute-force timing
    t0 = time.perf_counter()
    for q in qs:
        brute_force_knn(data, q, k)
    bf_s = (time.perf_counter() - t0) / queries

    # HNSW timing + recall
    t0 = time.perf_counter()
    recalls = []
    for q, tset in zip(qs, truth):
        approx = set(i for i, _ in index.search(q, k))
        recalls.append(len(approx & tset) / k)
    hnsw_s = (time.perf_counter() - t0) / queries

    print(f"dataset: {n} vectors x {dim} dims, k={k}, {queries} queries\n")
    print(f"  build time (HNSW):     {build_s:6.2f} s")
    print(f"  brute-force latency:   {bf_s*1000:6.2f} ms/query")
    print(f"  HNSW latency:          {hnsw_s*1000:6.2f} ms/query")
    print(f"  speedup:               {bf_s/hnsw_s:6.1f}x")
    print(f"  recall@{k}:             {np.mean(recalls):6.3f}")

    try:
        import faiss  # noqa

        index_f = faiss.IndexFlatL2(dim)
        index_f.add(data.astype("float32"))
        t0 = time.perf_counter()
        for q in qs:
            index_f.search(q.astype("float32").reshape(1, -1), k)
        faiss_s = (time.perf_counter() - t0) / queries
        print(f"  FAISS (flat) latency:  {faiss_s*1000:6.2f} ms/query")
    except ImportError:
        print("  (install faiss-cpu to compare against FAISS)")


if __name__ == "__main__":
    run()
