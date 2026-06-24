"""Skill co-occurrence graph + transferable-skill discovery.

Build a weighted graph where nodes are skills and an edge connects two skills
that frequently appear together (in job postings / resumes). Then, given the
skills someone *has*, surface adjacent skills they're likely close to acquiring
— the graph-traversal version of "people who know X often also know Y."

A small from-scratch graph: adjacency via co-occurrence counts, neighbor
ranking by edge weight. No networkx.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple


class SkillGraph:
    def __init__(self):
        # skill -> {neighbor: co-occurrence weight}
        self.adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def build(self, skill_sets: List[Set[str]]) -> "SkillGraph":
        """Each element of skill_sets is the set of skills from one document."""
        for skills in skill_sets:
            for a, b in combinations(sorted(skills), 2):
                self.adj[a][b] += 1.0
                self.adj[b][a] += 1.0
        return self

    def neighbors(self, skill: str, top: int = 5) -> List[Tuple[str, float]]:
        items = sorted(self.adj.get(skill, {}).items(), key=lambda kv: -kv[1])
        return items[:top]

    def suggest(self, have: Set[str], top: int = 5) -> List[Tuple[str, float]]:
        """Skills adjacent to what you have, that you don't have yet.

        Scores a candidate by summing edge weights from every skill you have to
        that candidate — i.e. how strongly your skill set 'points at' it.
        """
        scores: Dict[str, float] = defaultdict(float)
        for s in have:
            for nbr, w in self.adj.get(s, {}).items():
                if nbr not in have:
                    scores[nbr] += w
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top]
