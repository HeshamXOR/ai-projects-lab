"""BM25F — field-weighted BM25 for structured (code) documents, from scratch.

Plain BM25 treats a document as one bag of words. Code isn't flat: a match in
the *function name* or *signature* should count more than one in a comment.
BM25F handles this by combining per-field term frequencies with field weights
before the BM25 saturation, so ranking respects structure.

Fields here: name, signature, body, comments — each with its own boost.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+")


def tokenize(text: str) -> List[str]:
    # split snake_case / camelCase so "readFile" matches "read" and "file"
    out = []
    for tok in _TOKEN_RE.findall(text):
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok).replace("_", " ").split()
        out.extend(p.lower() for p in parts)
    return out


class BM25F:
    def __init__(self, field_weights: Dict[str, float] = None, k1: float = 1.5, b: float = 0.75):
        self.field_weights = field_weights or {
            "name": 3.0, "signature": 2.0, "comments": 1.5, "body": 1.0
        }
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_len = 0.0

    def index(self, docs: List[Dict[str, str]]) -> None:
        """Each doc is a dict of field_name -> text."""
        self.docs = docs
        self.doc_freq = {}
        total_len = 0
        for doc in docs:
            seen = set()
            for field, text in doc.items():
                for term in tokenize(text):
                    if term not in seen:
                        self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
                        seen.add(term)
                total_len += len(tokenize(text)) * self.field_weights.get(field, 1.0)
        n = len(docs)
        self.avg_len = total_len / n if n else 0.0
        self.idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self.doc_freq.items()
        }

    def _weighted_tf(self, query_term: str, doc: Dict[str, str]) -> float:
        tf = 0.0
        for field, text in doc.items():
            counts = Counter(tokenize(text))
            tf += counts.get(query_term, 0) * self.field_weights.get(field, 1.0)
        return tf

    def _doc_len(self, doc: Dict[str, str]) -> float:
        return sum(
            len(tokenize(text)) * self.field_weights.get(field, 1.0)
            for field, text in doc.items()
        )

    def search(self, query: str, k: int = 5):
        q_terms = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = self._doc_len(doc)
            for term in q_terms:
                tf = self._weighted_tf(term, doc)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
