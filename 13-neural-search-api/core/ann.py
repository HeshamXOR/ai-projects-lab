"""An HNSW-style approximate nearest-neighbor index in pure Python/NumPy.

HNSW (Hierarchical Navigable Small World, Malkov & Yashunin 2018) is the
state-of-the-art graph index for approximate nearest-neighbor search. This
module implements the algorithm from scratch -- there is no faiss, no hnswlib,
no nmslib. The only dependency is NumPy for vector arithmetic.

The structure is a multi-layer proximity graph:

* Each inserted point is assigned a maximum layer ``l`` drawn from an
  exponentially-decaying distribution ``l = floor(-ln(U) * mL)`` where
  ``U ~ Uniform(0, 1)``. Most points live only on layer 0; a few reach the
  upper, sparser layers that act as an express skip-list.
* Search starts at a single entry point on the top layer and greedily walks
  toward the query, descending one layer at a time. On the bottom layer it
  runs a best-first beam search of width ``ef`` and returns the ``k`` closest.
* Insertion uses the same descent to find entry points, then on each layer
  ``<= l`` it runs the beam search with width ``ef_construction`` and connects
  the new node to ``M`` neighbors chosen by a heuristic that keeps the graph
  navigable (it prefers diverse neighbors over merely-closest ones).

Distances: vectors are compared with cosine distance ``1 - cos_sim``. Inputs
are L2-normalized on insertion so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class _Node:
    """A single graph node holding its vector and per-layer adjacency.

    Attributes:
        idx: Internal integer index into the vector store.
        layer: Top layer this node participates in (0-indexed).
        neighbors: ``neighbors[l]`` is the list of internal indices this
            node connects to on layer ``l``.
    """

    __slots__ = ("idx", "layer", "neighbors")

    def __init__(self, idx: int, layer: int) -> None:
        self.idx = idx
        self.layer = layer
        self.neighbors: List[List[int]] = [[] for _ in range(layer + 1)]


class HNSWIndex:
    """A from-scratch HNSW approximate nearest-neighbor index.

    Args:
        dim: Dimensionality of the stored vectors.
        M: Target number of neighbors per node on layers > 0. Layer 0 uses
            ``2 * M`` (a standard HNSW choice that strengthens the base layer).
        ef_construction: Beam width used while inserting. Larger values build
            a higher-quality graph at the cost of slower insertion.
        ef_search: Default beam width used at query time. Larger values
            improve recall at the cost of slower search. Can be overridden
            per-query.
        ml: Level-generation normalization factor. The HNSW paper recommends
            ``ml = 1 / ln(M)``, which is the default when left as None.
        seed: Seed for the layer-assignment RNG, for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if M < 2:
            raise ValueError("M must be >= 2")

        self.dim = int(dim)
        self.M = int(M)
        self.M0 = int(2 * M)  # neighbor budget on the base layer
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.ml = float(ml) if ml is not None else 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)

        # Vector store: a growable list of normalized vectors.
        self._vectors: List[np.ndarray] = []
        self._nodes: List[_Node] = []
        self._entry_point: Optional[int] = None  # internal index
        self._max_layer: int = -1

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return len(self._vectors)

    # ------------------------------------------------------------------ #
    # Distance helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Return an L2-normalized copy of ``vec`` (float32)."""
        v = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            v = v / norm
        return v

    def _distance(self, a_idx: int, query: np.ndarray) -> float:
        """Cosine distance between stored vector ``a_idx`` and ``query``.

        Both operands are unit-norm, so cosine similarity is a dot product
        and cosine distance is ``1 - dot``.
        """
        sim = float(np.dot(self._vectors[a_idx], query))
        return 1.0 - sim

    # ------------------------------------------------------------------ #
    # Level assignment
    # ------------------------------------------------------------------ #
    def _random_level(self) -> int:
        """Draw a node's top layer from the exponential-decay distribution."""
        u = float(self._rng.random())
        # Guard against u == 0 producing inf.
        u = max(u, 1e-12)
        return int(math.floor(-math.log(u) * self.ml))

    # ------------------------------------------------------------------ #
    # Core search primitive
    # ------------------------------------------------------------------ #
    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """Best-first beam search on a single ``layer``.

        Implements Algorithm 2 of the HNSW paper. Maintains a candidate
        min-heap (closest first) and a result max-heap (farthest first,
        stored with negated distance) capped at ``ef`` elements.

        Args:
            query: Normalized query vector.
            entry_points: Internal indices to seed the search from.
            ef: Beam width / size of the dynamic result list.
            layer: Which graph layer to traverse.

        Returns:
            Up to ``ef`` ``(distance, idx)`` pairs, unsorted (heap order).
        """
        visited: set[int] = set(entry_points)
        # Candidate heap: (distance, idx), min-heap on distance.
        candidates: List[Tuple[float, int]] = []
        # Result heap: (-distance, idx), max-heap on distance via negation.
        results: List[Tuple[float, int]] = []

        for ep in entry_points:
            d = self._distance(ep, query)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            # Farthest distance currently in the result set.
            farthest = -results[0][0]
            if dist_c > farthest and len(results) >= ef:
                break  # all remaining candidates are worse than our worst

            for neighbor in self._nodes[c].neighbors[layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._distance(neighbor, query)
                farthest = -results[0][0] if results else float("inf")
                if d < farthest or len(results) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)  # drop the farthest

        return [(-neg_d, idx) for neg_d, idx in results]

    def _select_neighbors_heuristic(
        self,
        candidates: List[Tuple[float, int]],
        m: int,
    ) -> List[int]:
        """Choose ``m`` neighbors using HNSW's diversity heuristic.

        Implements Algorithm 4 (the neighbor-selection heuristic). A
        candidate is kept only if it is closer to the new point than to any
        already-selected neighbor. This avoids clustering all edges in one
        direction and keeps the graph navigable.

        Args:
            candidates: ``(distance_to_query, idx)`` pairs.
            m: Maximum number of neighbors to return.

        Returns:
            A list of selected internal indices, closest-first.
        """
        # Sort candidates closest-first.
        ordered = sorted(candidates, key=lambda pair: pair[0])
        selected: List[int] = []
        for dist_cand, cand in ordered:
            if len(selected) >= m:
                break
            good = True
            cand_vec = self._vectors[cand]
            for sel in selected:
                # Distance from candidate to an already-selected neighbor.
                d_cand_sel = 1.0 - float(np.dot(cand_vec, self._vectors[sel]))
                if d_cand_sel < dist_cand:
                    # The candidate is closer to an existing neighbor than to
                    # the query point -> it is redundant, skip it.
                    good = False
                    break
            if good:
                selected.append(cand)
        return selected

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def add(self, vector: np.ndarray) -> int:
        """Insert ``vector`` into the index.

        Args:
            vector: A vector of length ``dim``. It is L2-normalized on entry.

        Returns:
            The internal integer index assigned to the new vector.
        """
        vec = self._normalize(vector)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"vector dim {vec.shape[0]} != index dim {self.dim}"
            )

        idx = len(self._vectors)
        self._vectors.append(vec)
        level = self._random_level()
        node = _Node(idx, level)
        self._nodes.append(node)

        # First element becomes the entry point and we are done.
        if self._entry_point is None:
            self._entry_point = idx
            self._max_layer = level
            return idx

        ep = self._entry_point
        # Phase 1: greedily descend from the top down to level+1 using ef=1.
        for layer in range(self._max_layer, level, -1):
            nearest = self._search_layer(vec, [ep], ef=1, layer=layer)
            ep = min(nearest, key=lambda pair: pair[0])[1]

        # Phase 2: from min(level, max_layer) down to 0, search and connect.
        for layer in range(min(level, self._max_layer), -1, -1):
            found = self._search_layer(
                vec, [ep], ef=self.ef_construction, layer=layer
            )
            m = self.M0 if layer == 0 else self.M
            neighbors = self._select_neighbors_heuristic(found, m)

            node.neighbors[layer] = list(neighbors)
            # Add reciprocal edges and prune the neighbors' adjacency.
            for nb in neighbors:
                nb_node = self._nodes[nb]
                nb_node.neighbors[layer].append(idx)
                max_conn = self.M0 if layer == 0 else self.M
                if len(nb_node.neighbors[layer]) > max_conn:
                    self._prune(nb, layer, max_conn)

            # Descend using the closest found point as the next entry point.
            if found:
                ep = min(found, key=lambda pair: pair[0])[1]

        # Update the global entry point if this node reached a new top.
        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = idx

        return idx

    def _prune(self, node_idx: int, layer: int, max_conn: int) -> None:
        """Re-select a node's neighbor list down to ``max_conn`` edges."""
        node = self._nodes[node_idx]
        base = self._vectors[node_idx]
        cands = [
            (1.0 - float(np.dot(base, self._vectors[nb])), nb)
            for nb in node.neighbors[layer]
        ]
        node.neighbors[layer] = self._select_neighbors_heuristic(
            cands, max_conn
        )

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Return the approximate ``k`` nearest neighbors of ``query``.

        Args:
            query: A vector of length ``dim``.
            k: Number of neighbors to return.
            ef_search: Optional override for the beam width. Must be ``>= k``
                to give the search room to find ``k`` good candidates;
                clamped up automatically otherwise.

        Returns:
            A list of ``(internal_idx, cosine_similarity)`` pairs sorted by
            descending similarity (closest first).
        """
        if self._entry_point is None or k <= 0:
            return []

        q = self._normalize(query)
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, k)

        # Descend the upper layers greedily with ef=1.
        ep = self._entry_point
        for layer in range(self._max_layer, 0, -1):
            nearest = self._search_layer(q, [ep], ef=1, layer=layer)
            ep = min(nearest, key=lambda pair: pair[0])[1]

        # Beam search on the base layer.
        found = self._search_layer(q, [ep], ef=ef, layer=0)
        found.sort(key=lambda pair: pair[0])  # closest first by distance
        top = found[:k]
        # Convert cosine distance back to cosine similarity for the caller.
        return [(idx, 1.0 - dist) for dist, idx in top]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_state(self) -> dict:
        """Serialize the graph topology and parameters (vectors separate).

        The dense vector matrix is returned separately by the store layer
        (saved as ``.npy``); this method only captures the small graph.
        """
        return {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "ml": self.ml,
            "entry_point": self._entry_point,
            "max_layer": self._max_layer,
            "nodes": [
                {"idx": n.idx, "layer": n.layer, "neighbors": n.neighbors}
                for n in self._nodes
            ],
        }

    @classmethod
    def from_state(cls, state: dict, vectors: np.ndarray) -> "HNSWIndex":
        """Rebuild an index from :meth:`to_state` output plus the vectors."""
        idx = cls(
            dim=int(state["dim"]),
            M=int(state["M"]),
            ef_construction=int(state["ef_construction"]),
            ef_search=int(state["ef_search"]),
            ml=float(state["ml"]),
        )
        idx._entry_point = state["entry_point"]
        idx._max_layer = int(state["max_layer"])
        idx._vectors = [
            np.asarray(vectors[i], dtype=np.float32)
            for i in range(len(vectors))
        ]
        nodes: List[_Node] = []
        for nd in state["nodes"]:
            node = _Node(int(nd["idx"]), int(nd["layer"]))
            node.neighbors = [list(map(int, lst)) for lst in nd["neighbors"]]
            nodes.append(node)
        idx._nodes = nodes
        return idx


def brute_force_search(
    vectors: np.ndarray, query: np.ndarray, k: int
) -> List[Tuple[int, float]]:
    """Exact nearest-neighbor search by full cosine-similarity scan.

    Used by the test-suite as the ground truth to measure HNSW recall, and
    available as a fallback for tiny corpora.

    Args:
        vectors: An ``(n, dim)`` matrix of (not necessarily normalized) rows.
        query: A query vector of length ``dim``.
        k: Number of neighbors to return.

    Returns:
        ``(row_index, cosine_similarity)`` pairs, closest first.
    """
    if len(vectors) == 0 or k <= 0:
        return []
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_n = mat / norms

    q = np.asarray(query, dtype=np.float32).ravel()
    qn = q / (np.linalg.norm(q) or 1.0)

    sims = mat_n @ qn
    k = min(k, len(sims))
    # argpartition for the top-k, then sort just those.
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), float(sims[i])) for i in top_idx]
