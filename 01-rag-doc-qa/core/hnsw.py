"""HNSW — Hierarchical Navigable Small World graph index, from scratch.

This is the approximate-nearest-neighbor algorithm behind production vector
databases (FAISS, Qdrant, Weaviate, pgvector). Instead of comparing a query to
every vector (O(N) brute force), HNSW builds a multi-layer navigable graph and
greedily hops toward the nearest neighbors in roughly O(log N) time.

The structure (Malkov & Yashunin, 2018):
  * A hierarchy of layers. Layer 0 holds every point; each higher layer holds a
    random, exponentially-thinning subset. Think "skip list, but for a graph."
  * Search starts at the top layer from a single entry point, greedily walks to
    the closest node it can find, then descends a layer and repeats — zooming in.
  * Insertion searches for the new node's nearest neighbors at each layer and
    links them, using a heuristic that keeps the graph navigable (not just
    nearest — it prefers diverse neighbors so the graph doesn't fragment).

Implemented here in pure Python + NumPy (only for the distance kernel). The
companion benchmark (bench.py) shows recall and speedup vs. brute force.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class HNSW:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ):
        """
        M               max neighbors per node per layer (graph degree).
        ef_construction size of the dynamic candidate list while inserting
                        (bigger = better graph, slower build).
        ef_search       candidate-list size at query time (bigger = better
                        recall, slower search).
        """
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M  # layer 0 can be denser
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._mL = 1.0 / math.log(M)  # layer-assignment normalizer
        self._rng = np.random.default_rng(seed)

        self._vectors: List[np.ndarray] = []
        # graph[layer][node_id] -> list of neighbor ids
        self._graph: List[Dict[int, List[int]]] = []
        self._entry: Optional[int] = None
        self._top_layer = -1

    # ---- distance kernel ----
    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        # Euclidean distance squared (monotonic with L2; avoids the sqrt).
        d = a - b
        return float(d @ d)

    def _random_level(self) -> int:
        # Exponentially decaying layer assignment: most nodes land on layer 0.
        return int(-math.log(self._rng.random()) * self._mL)

    # ---- core graph search ----
    def _search_layer(
        self, query: np.ndarray, entry_points: List[int], ef: int, layer: int
    ) -> List[Tuple[float, int]]:
        """Greedy best-first search within a single layer.

        Returns up to `ef` (distance, node_id) pairs, the closest found.
        Uses two heaps: a min-heap of candidates to explore, and a max-heap of
        the best results so far (so we can pop the current worst).
        """
        visited = set(entry_points)
        # candidates: min-heap by distance (explore closest first)
        candidates: List[Tuple[float, int]] = []
        # results: max-heap (store negative distance) of the best ef so far
        results: List[Tuple[float, int]] = []

        for ep in entry_points:
            d = self._dist(query, self._vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst = -results[0][0]
            if dist_c > worst and len(results) >= ef:
                break  # everything left is farther than our current worst
            for neighbor in self._graph[layer].get(c, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._dist(query, self._vectors[neighbor])
                worst = -results[0][0]
                if d < worst or len(results) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)  # drop current worst

        return [(-nd, nid) for nd, nid in results]

    def _select_neighbors(
        self, candidates: List[Tuple[float, int]], M: int
    ) -> List[int]:
        """Heuristic neighbor selection (Algorithm 4 in the paper).

        Rather than just taking the M closest, prefer candidates that are closer
        to the new node than to any already-selected neighbor. This keeps edges
        diverse and the graph globally navigable instead of clumping.
        """
        candidates = sorted(candidates, key=lambda x: x[0])
        selected: List[int] = []
        for dist_cand, cand in candidates:
            if len(selected) >= M:
                break
            good = True
            for s in selected:
                d_to_selected = self._dist(self._vectors[cand], self._vectors[s])
                if d_to_selected < dist_cand:
                    # cand is closer to an existing pick than to the query node
                    good = False
                    break
            if good:
                selected.append(cand)
        # top up with nearest if the heuristic was too strict
        if len(selected) < M:
            for _, cand in candidates:
                if cand not in selected:
                    selected.append(cand)
                    if len(selected) >= M:
                        break
        return selected

    # ---- public API ----
    def add(self, vector: np.ndarray) -> int:
        vector = np.asarray(vector, dtype=np.float64)
        node_id = len(self._vectors)
        self._vectors.append(vector)
        level = self._random_level()

        # grow graph structure to accommodate new layers
        while len(self._graph) <= level:
            self._graph.append({})
        for lyr in range(level + 1):
            self._graph[lyr].setdefault(node_id, [])

        # first node ever: it's the entry point
        if self._entry is None:
            self._entry = node_id
            self._top_layer = level
            return node_id

        # 1) descend from the top to `level`+1 using greedy 1-NN search
        ep = [self._entry]
        for lyr in range(self._top_layer, level, -1):
            res = self._search_layer(vector, ep, ef=1, layer=lyr)
            ep = [res[0][1]]

        # 2) from `level` down to 0, find neighbors and link
        for lyr in range(min(level, self._top_layer), -1, -1):
            res = self._search_layer(vector, ep, ef=self.ef_construction, layer=lyr)
            M = self.M_max0 if lyr == 0 else self.M
            neighbors = self._select_neighbors(res, M)
            # link both directions
            self._graph[lyr][node_id] = list(neighbors)
            for n in neighbors:
                self._graph[lyr].setdefault(n, []).append(node_id)
                # prune n's neighbor list if it exceeds the cap
                if len(self._graph[lyr][n]) > M:
                    n_vec = self._vectors[n]
                    cand = [(self._dist(n_vec, self._vectors[x]), x) for x in self._graph[lyr][n]]
                    self._graph[lyr][n] = self._select_neighbors(cand, M)
            ep = [nid for _, nid in res]

        # 3) update the entry point if this node went higher than any before
        if level > self._top_layer:
            self._top_layer = level
            self._entry = node_id
        return node_id

    def add_batch(self, vectors: np.ndarray) -> None:
        for v in vectors:
            self.add(v)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Return the k approximate nearest neighbors as (id, distance)."""
        if self._entry is None:
            return []
        query = np.asarray(query, dtype=np.float64)
        ep = [self._entry]
        # zoom down through the upper layers with ef=1
        for lyr in range(self._top_layer, 0, -1):
            res = self._search_layer(query, ep, ef=1, layer=lyr)
            ep = [res[0][1]]
        # thorough search on layer 0
        res = self._search_layer(query, ep, ef=max(self.ef_search, k), layer=0)
        res.sort(key=lambda x: x[0])
        # convert squared distance back to Euclidean for reporting
        return [(nid, math.sqrt(d)) for d, nid in res[:k]]

    def __len__(self):
        return len(self._vectors)


def brute_force_knn(vectors: np.ndarray, query: np.ndarray, k: int = 5):
    """Exact k-NN by scanning everything — the ground truth for benchmarking."""
    d = vectors - query
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    idx = np.argsort(dist)[:k]
    return [(int(i), float(dist[i])) for i in idx]
