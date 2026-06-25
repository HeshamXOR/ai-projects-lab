"""Evaluation metrics for language models — perplexity & accuracy, from scratch.

The WHY: to know whether fine-tuning helped, we need numbers, and we want to be
sure those numbers are the real definitions rather than whatever a library does
under the hood. So both metrics are computed directly from logits/labels here.

PERPLEXITY. A language model defines a distribution P(token | context). Its
quality on held-out text is the average negative log-likelihood (NLL) per token:

        NLL  =  -(1/N) * sum_t log P(x_t | x_<t)

Perplexity is the exponential of that mean NLL:

        PPL  =  exp(NLL)

Intuition: PPL is the effective branching factor — "on average the model was as
confused as if it had to choose uniformly among PPL tokens." A perfect model has
PPL 1; uniform over a V-token vocab has PPL V. We compute log P via a numerically
stable log-softmax of the logits, gather the log-prob of the gold token at each
position, mask out IGNORE_INDEX / shift positions, and average. Test (5) pins
this to a hand-computed value.

ACCURACY. For tasks with a discrete answer we also report token-level accuracy:
of the supervised (non-masked) positions, what fraction did argmax(logits) get
right. We additionally expose exact-match over whole sequences. Test (6) checks
the token-level number against a hand value.

All functions accept the same [B, T, V] logits and [B, T] labels convention the
trainer produces, and respect IGNORE_INDEX so padded/prompt tokens don't count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .data import IGNORE_INDEX


# ============================================================================
# Core log-prob machinery
# ============================================================================
def gather_token_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return log P(label_t) at each *supervised* position, flattened.

    Args:
        logits: [B, T, V] raw scores (pre-softmax).
        labels: [B, T] gold token ids; positions equal to ignore_index are skipped.

    Returns:
        1-D tensor of the gold-token log-probabilities for every non-ignored
        position (length = number of supervised tokens). Empty if none.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got shape {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} must match logits batch/time "
            f"{tuple(logits.shape[:2])}"
        )

    log_probs = F.log_softmax(logits.float(), dim=-1)        # [B, T, V]
    mask = labels != ignore_index                             # [B, T]

    # Replace ignored labels with 0 so gather is in-range, then drop via mask.
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    gold = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    return gold[mask]                                         # [num_supervised]


def mean_negative_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Average NLL per supervised token: -(1/N) sum log P(gold)."""
    token_lp = gather_token_log_probs(logits, labels, ignore_index)
    if token_lp.numel() == 0:
        return float("nan")
    return float(-token_lp.mean().item())


def perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Perplexity = exp(mean NLL) over supervised tokens.

    Returns +inf if the mean NLL overflows, NaN if there are no supervised tokens.
    """
    nll = mean_negative_log_likelihood(logits, labels, ignore_index)
    if nll != nll:  # NaN check
        return float("nan")
    try:
        import math

        return math.exp(nll)
    except OverflowError:
        return float("inf")


# ============================================================================
# Accuracy
# ============================================================================
def token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Fraction of supervised positions where argmax(logits) == label.

    Returns NaN if there are no supervised positions.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    preds = logits.argmax(dim=-1)                 # [B, T]
    mask = labels != ignore_index
    total = int(mask.sum().item())
    if total == 0:
        return float("nan")
    correct = int(((preds == labels) & mask).sum().item())
    return correct / total


def sequence_exact_match(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    normalize: bool = True,
) -> float:
    """Fraction of (prediction, reference) pairs that match exactly.

    With normalize=True we lower-case and collapse surrounding whitespace, the
    standard light normalization used by exact-match QA metrics so that trivial
    formatting differences don't count as wrong.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        return float("nan")

    def norm(s: str) -> str:
        return " ".join(s.strip().lower().split()) if normalize else s

    hits = sum(1 for p, r in zip(predictions, references) if norm(p) == norm(r))
    return hits / len(predictions)


# ============================================================================
# Aggregating evaluator over a dataset
# ============================================================================
@dataclass
class EvalResult:
    """Bundle of metrics from one evaluation pass."""

    perplexity: float
    token_accuracy: float
    num_tokens: int
    mean_nll: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "perplexity": self.perplexity,
            "token_accuracy": self.token_accuracy,
            "num_tokens": self.num_tokens,
            "mean_nll": self.mean_nll,
        }


class Evaluator:
    """Accumulate NLL and accuracy across many batches, then summarize.

    Streaming accumulation (sum of NLL and counts) means we can evaluate datasets
    larger than memory one batch at a time and still get the exact corpus-level
    perplexity, which is exp of the *token-weighted* mean NLL — not the average of
    per-batch perplexities (that would be wrong for ragged batches).
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        self.ignore_index = ignore_index
        self._nll_sum = 0.0
        self._token_count = 0
        self._correct = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        token_lp = gather_token_log_probs(logits, labels, self.ignore_index)
        n = token_lp.numel()
        if n == 0:
            return
        self._nll_sum += float(-token_lp.sum().item())
        self._token_count += n

        preds = logits.argmax(dim=-1)
        mask = labels != self.ignore_index
        self._correct += int(((preds == labels) & mask).sum().item())

    def result(self) -> EvalResult:
        import math

        if self._token_count == 0:
            return EvalResult(float("nan"), float("nan"), 0, float("nan"))
        mean_nll = self._nll_sum / self._token_count
        try:
            ppl = math.exp(mean_nll)
        except OverflowError:
            ppl = float("inf")
        acc = self._correct / self._token_count
        return EvalResult(ppl, acc, self._token_count, mean_nll)
