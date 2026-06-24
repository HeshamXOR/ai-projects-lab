"""ROUGE — the standard summarization metric, from scratch.

ROUGE measures n-gram overlap between a candidate summary and a reference.
- ROUGE-N: overlap of n-grams (recall/precision/F1).
- ROUGE-L: longest common subsequence, rewarding in-order overlap.

This is how summarization systems are evaluated in research. Implemented
directly so the app can score its own extractive vs. abstractive summaries.
"""

from __future__ import annotations

import re
from typing import Dict, List


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: List[str], n: int):
    from collections import Counter

    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int = 1) -> Dict[str, float]:
    cand, ref = _tokens(candidate), _tokens(reference)
    cg, rg = _ngrams(cand, n), _ngrams(ref, n)
    overlap = sum((cg & rg).values())
    precision = overlap / max(sum(cg.values()), 1)
    recall = overlap / max(sum(rg.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: List[str], b: List[str]) -> int:
    # classic dynamic-programming longest common subsequence
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    cand, ref = _tokens(candidate), _tokens(reference)
    lcs = _lcs_length(cand, ref)
    precision = lcs / max(len(cand), 1)
    recall = lcs / max(len(ref), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_report(candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
    return {
        "rouge-1": rouge_n(candidate, reference, 1),
        "rouge-2": rouge_n(candidate, reference, 2),
        "rouge-l": rouge_l(candidate, reference),
    }
