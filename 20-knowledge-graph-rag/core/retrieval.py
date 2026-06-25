"""Graph-augmented retrieval, from scratch.

Two retrievers over a passage collection:

* :meth:`Retriever.retrieve` — a **baseline** lexical retriever using a
  from-scratch BM25-lite scorer (term frequency saturation + IDF + length
  normalization). No sklearn, no rank_bm25.
* :meth:`Retriever.graph_augmented_retrieve` — matches query terms to entities
  in a :class:`~core.graph.KnowledgeGraph`, expands to neighbors within ``k``
  hops, appends those entity terms to the query, and re-scores. This lifts
  passages that share *no* surface terms with the original question but are
  connected through the graph (multi-hop recall).

An extractive ``assemble_answer`` stitches the top passages into a short answer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .graph import KnowledgeGraph

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word/number tokenizer."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredPassage:
    """A passage with its retrieval score and originating index."""

    index: int
    text: str
    score: float


@dataclass
class Retriever:
    """BM25-lite lexical retriever with optional graph expansion.

    Build with :meth:`fit` (or pass passages to the constructor and call
    :meth:`fit`). Scores are computed from scratch; ``graph`` is optional and
    only needed for :meth:`graph_augmented_retrieve`.
    """

    passages: List[str] = field(default_factory=list)
    graph: Optional[KnowledgeGraph] = None
    k1: float = 1.5
    b: float = 0.75

    # fitted state
    _docs: List[List[str]] = field(default_factory=list)
    _df: Dict[str, int] = field(default_factory=dict)
    _idf: Dict[str, float] = field(default_factory=dict)
    _doc_len: List[int] = field(default_factory=list)
    _avg_len: float = 0.0

    def fit(self) -> "Retriever":
        """Tokenize passages and precompute document frequencies and IDF."""
        self._docs = [tokenize(p) for p in self.passages]
        self._doc_len = [len(d) for d in self._docs]
        n = len(self._docs)
        self._avg_len = (sum(self._doc_len) / n) if n else 0.0
        df: Dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._df = df
        # BM25 idf with +1 smoothing to keep it non-negative
        self._idf = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        return self

    def add_passages(self, new_passages: List[str]) -> None:
        """Append passages and refit (in-memory incremental ingest)."""
        self.passages.extend(new_passages)
        self.fit()

    # --------------------------------------------------------------- scoring
    def _score_doc(self, query_terms: List[str], doc_idx: int) -> float:
        doc = self._docs[doc_idx]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avg_len, 1e-6))
            score += idf * (f * (self.k1 + 1)) / max(denom, 1e-6)
        return score

    def _rank(self, query_terms: List[str], k: int) -> List[ScoredPassage]:
        scored = [
            ScoredPassage(i, self.passages[i], self._score_doc(query_terms, i))
            for i in range(len(self.passages))
        ]
        scored.sort(key=lambda s: (-s.score, s.index))
        return [s for s in scored if s.score > 0][:k]

    # ------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int = 3) -> List[ScoredPassage]:
        """Baseline lexical retrieval (no graph)."""
        return self._rank(tokenize(query), k)

    def graph_augmented_retrieve(
        self, query: str, k: int = 3, hops: int = 1
    ) -> Tuple[List[ScoredPassage], List[str]]:
        """Retrieve using the query expanded with graph-reachable entity terms.

        Returns the ranked passages and the list of expansion terms added. If no
        graph is attached or no seeds match, this falls back to the baseline.
        """
        terms = tokenize(query)
        expansion: List[str] = []
        if self.graph is not None:
            expansion = self.graph.expand_terms(query, k=hops)
            for ent in expansion:
                terms.extend(tokenize(ent))
        return self._rank(terms, k), expansion


def assemble_answer(question: str, passages: List[ScoredPassage], max_sentences: int = 2) -> str:
    """Extractively assemble a short answer from the top passages.

    Picks the sentences across the retrieved passages with the highest overlap
    against the question's content words. Purely extractive — no generation.
    """
    if not passages:
        return "No relevant information found."
    q_terms = set(tokenize(question))
    sentences: List[str] = []
    for p in passages:
        sentences.extend(re.split(r"(?<=[.!?])\s+", p.text.strip()))
    scored: List[Tuple[float, str]] = []
    for s in sentences:
        if not s.strip():
            continue
        overlap = len(q_terms & set(tokenize(s)))
        scored.append((overlap, s.strip()))
    scored.sort(key=lambda t: -t[0])
    chosen = [s for ov, s in scored[:max_sentences] if ov > 0]
    if not chosen:
        chosen = [passages[0].text]
    return " ".join(chosen)
