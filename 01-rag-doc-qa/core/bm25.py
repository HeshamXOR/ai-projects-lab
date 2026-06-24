"""BM25 — the classic probabilistic keyword ranking function, from scratch.

BM25 scores how well a document matches a query based on term frequency (with
saturation, so the 10th occurrence of a word matters less than the 2nd) and
inverse document frequency (rare words count more), normalized by document
length. It's still the backbone of search engines like Elasticsearch.

We implement it directly so hybrid retrieval can fuse it with vector search —
keyword precision (exact terms, names, numbers) complementing semantic recall.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # k1 controls term-frequency saturation; b controls length normalization.
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_len = 0.0

    def index(self, documents: List[str]) -> None:
        self.docs = [tokenize(d) for d in documents]
        self.doc_freq = {}
        for tokens in self.docs:
            for term in set(tokens):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        n = len(self.docs)
        self.avg_len = sum(len(d) for d in self.docs) / n if n else 0.0
        # idf with the standard BM25 +0.5 smoothing
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def _score(self, query_terms: List[str], doc_idx: int) -> float:
        tokens = self.docs[doc_idx]
        freqs = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        query_terms = tokenize(query)
        scored = [(i, self._score(query_terms, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
