"""TextRank — extractive summarization via graph centrality, from scratch.

TextRank is PageRank applied to sentences. Build a graph where each sentence is
a node and edges are weighted by sentence similarity; the most "central"
sentences (those similar to many other important sentences) are the summary.
Centrality is computed by power iteration — the same eigenvector idea that
ranked the early web.

No networkx, no sklearn — the similarity, the graph, and the power iteration are
all here. Works on CPU instantly; needs no model (uses lexical overlap), so it's
a true from-scratch counterpart to the abstractive (transformer) summarizer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import numpy as np


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 0]


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _similarity(a: List[str], b: List[str]) -> float:
    """Overlap of words, normalized by sentence lengths (TextRank's measure)."""
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    denom = math.log(len(a) + 1) + math.log(len(b) + 1)
    return common / denom if denom > 0 else 0.0


def textrank(text: str, top_k: int = 3, d: float = 0.85, iters: int = 50) -> List[str]:
    """Return the top_k sentences by TextRank score, in original order."""
    sentences = split_sentences(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    tokens = [_tokenize(s) for s in sentences]
    # build weighted adjacency
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = _similarity(tokens[i], tokens[j])
            W[i, j] = W[j, i] = sim

    # column-normalize to a transition matrix (avoid divide-by-zero)
    col_sums = W.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1
    M = W / col_sums

    # power iteration: r = (1-d)/n + d * Mᵀ r
    r = np.ones(n) / n
    for _ in range(iters):
        r_new = (1 - d) / n + d * (M.T @ r)
        if np.allclose(r_new, r, atol=1e-8):
            r = r_new
            break
        r = r_new

    top_idx = sorted(np.argsort(-r)[:top_k])
    return [sentences[i] for i in top_idx]


def summarize_extractive(text: str, ratio: float = 0.3, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    k = max(1, min(max_sentences, int(len(sentences) * ratio)))
    return " ".join(textrank(text, top_k=k))
