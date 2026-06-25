"""In-memory knowledge graph, from scratch (no networkx).

The graph stores a directed, weighted adjacency map::

    adjacency: node -> list of (neighbor, relation, weight)

For pathfinding and subgraph expansion we also keep a mirrored undirected view
so traversal can move "backwards" along a relation (the fact that Beta was
acquired *by* Acme still links the two for recall purposes).

Implemented here:

* ``add_entity`` / ``add_relation`` — mutate the adjacency map; relations are
  deduplicated and their weights accumulate.
* ``neighbors`` — directed or undirected neighbor listing.
* ``shortest_path`` — Dijkstra by edge cost (``cost = 1 / weight`` so stronger
  links are cheaper), with a BFS fast-path falling out naturally for uniform
  weights. Returns the node sequence.
* ``subgraph_for_query`` — match query terms to seed nodes, then BFS outward up
  to ``k`` hops to collect a connected neighborhood.

No third-party graph library is used.
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

Edge = Tuple[str, str, float]  # (neighbor, relation, weight)


@dataclass
class KnowledgeGraph:
    """A directed, weighted knowledge graph with from-scratch traversal."""

    adjacency: Dict[str, List[Edge]] = field(default_factory=lambda: defaultdict(list))
    _nodes: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ build
    def add_entity(self, name: str) -> None:
        """Register a node (idempotent)."""
        name = name.strip()
        if not name:
            return
        self._nodes.add(name)
        self.adjacency.setdefault(name, [])

    def add_relation(self, head: str, relation: str, tail: str, weight: float = 1.0) -> None:
        """Add a directed edge ``head -[relation]-> tail``.

        If the same ``(head, relation, tail)`` edge already exists its weight is
        increased instead of duplicating the edge.
        """
        head, tail = head.strip(), tail.strip()
        if not head or not tail or head == tail:
            return
        self.add_entity(head)
        self.add_entity(tail)
        for i, (nbr, rel, w) in enumerate(self.adjacency[head]):
            if nbr == tail and rel == relation:
                self.adjacency[head][i] = (nbr, rel, w + weight)
                return
        self.adjacency[head].append((tail, relation, weight))

    # ------------------------------------------------------------------ query
    @property
    def nodes(self) -> List[str]:
        """All node names, sorted."""
        return sorted(self._nodes)

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return sum(len(v) for v in self.adjacency.values())

    def edges(self) -> List[Tuple[str, str, str, float]]:
        """All directed edges as ``(head, relation, tail, weight)`` tuples."""
        out: List[Tuple[str, str, str, float]] = []
        for head, edges in self.adjacency.items():
            for tail, rel, w in edges:
                out.append((head, rel, tail, w))
        return out

    def neighbors(self, node: str, undirected: bool = False) -> List[Edge]:
        """Return outgoing edges of ``node`` (plus incoming if ``undirected``)."""
        node = node.strip()
        out: List[Edge] = list(self.adjacency.get(node, []))
        if undirected:
            for head, edges in self.adjacency.items():
                if head == node:
                    continue
                for tail, rel, w in edges:
                    if tail == node:
                        out.append((head, rel, w))
        return out

    def _undirected_adj(self) -> Dict[str, List[Edge]]:
        """Build a symmetric adjacency map for traversal."""
        sym: Dict[str, List[Edge]] = defaultdict(list)
        for head, edges in self.adjacency.items():
            for tail, rel, w in edges:
                sym[head].append((tail, rel, w))
                sym[tail].append((head, rel, w))
        return sym

    def shortest_path(
        self, src: str, dst: str, undirected: bool = True
    ) -> Optional[List[str]]:
        """Dijkstra shortest path from ``src`` to ``dst`` by edge cost.

        Cost of an edge is ``1 / weight`` (clamped), so heavier links are
        cheaper to cross. Returns the list of nodes on the path, or ``None`` if
        unreachable. With uniform weights this degenerates to BFS hop-count.
        """
        src, dst = src.strip(), dst.strip()
        if src not in self._nodes or dst not in self._nodes:
            return None
        if src == dst:
            return [src]

        adj = self._undirected_adj() if undirected else self.adjacency
        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, str] = {}
        heap: List[Tuple[float, str]] = [(0.0, src)]
        visited: Set[str] = set()

        while heap:
            d, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == dst:
                break
            for nbr, _rel, w in adj.get(node, []):
                cost = 1.0 / max(w, 1e-6)
                nd = d + cost
                if nd < dist.get(nbr, float("inf")):
                    dist[nbr] = nd
                    prev[nbr] = node
                    heapq.heappush(heap, (nd, nbr))

        if dst not in prev and dst != src:
            return None
        # reconstruct
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def match_seeds(self, query: str) -> List[str]:
        """Find nodes whose surface form appears in the query (case-insensitive).

        Multi-word node names are matched as substrings; single tokens are
        matched on word boundaries to avoid spurious partial hits.
        """
        ql = query.lower()
        q_tokens = set(re.findall(r"[a-z0-9]+", ql))
        seeds: List[str] = []
        for node in self._nodes:
            nl = node.lower()
            if " " in nl or "-" in nl:
                if nl in ql:
                    seeds.append(node)
            else:
                if nl in q_tokens:
                    seeds.append(node)
        return seeds

    def subgraph_for_query(
        self, query: str, k: int = 1, undirected: bool = True
    ) -> Set[str]:
        """Collect nodes within ``k`` hops of any query-matched seed node.

        Performs a bounded BFS from every seed simultaneously and returns the
        union of reachable nodes (including the seeds). Used by the retriever to
        gather entity terms that expand the query.
        """
        seeds = self.match_seeds(query)
        if not seeds:
            return set()
        adj = self._undirected_adj() if undirected else self.adjacency
        visited: Set[str] = set(seeds)
        frontier: deque = deque((s, 0) for s in seeds)
        while frontier:
            node, depth = frontier.popleft()
            if depth >= k:
                continue
            for nbr, _rel, _w in adj.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    frontier.append((nbr, depth + 1))
        return visited

    def expand_terms(self, query: str, k: int = 1) -> List[str]:
        """Return entity terms reachable from query seeds, excluding seeds.

        These are the *new* terms graph expansion contributes to the query.
        """
        seeds = set(self.match_seeds(query))
        sub = self.subgraph_for_query(query, k=k)
        return sorted(sub - seeds)
