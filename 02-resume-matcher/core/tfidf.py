"""TF-IDF vectorizer, from scratch.

Turns documents into sparse-ish term weight vectors. Term Frequency × Inverse
Document Frequency: a term matters in proportion to how often it appears in a
document, discounted by how common it is across all documents (so "the" counts
for almost nothing, "kubernetes" counts a lot). This is the classic text
representation that powers search ranking and many NLP baselines.

Implemented over plain NumPy — no scikit-learn — so the resume matcher computes
its own features rather than calling a library.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    def __init__(self, max_features: int = 4000, ngram: int = 1):
        self.max_features = max_features
        self.ngram = ngram
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = None

    def _terms(self, tokens: List[str]) -> List[str]:
        terms = list(tokens)
        for n in range(2, self.ngram + 1):
            terms += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return terms

    def fit(self, docs: List[str]) -> "TfidfVectorizer":
        # document frequency for every term
        df = Counter()
        tokenized = [self._terms(tokenize(d)) for d in docs]
        for terms in tokenized:
            for t in set(terms):
                df[t] += 1
        # keep the most frequent terms as the vocabulary
        most_common = [t for t, _ in df.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(sorted(most_common))}
        n = len(docs)
        idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0  # smoothed idf
        self.idf = idf
        return self

    def transform(self, docs: List[str]) -> np.ndarray:
        rows = np.zeros((len(docs), len(self.vocab)))
        for r, d in enumerate(docs):
            terms = self._terms(tokenize(d))
            tf = Counter(terms)
            total = sum(tf.values()) or 1
            for t, c in tf.items():
                j = self.vocab.get(t)
                if j is not None:
                    rows[r, j] = (c / total) * self.idf[j]
        # L2-normalize rows so cosine similarity is just a dot product
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return rows / norms

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)
