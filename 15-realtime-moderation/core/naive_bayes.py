"""Multinomial Naive Bayes text classifier, implemented from scratch.

This is a faithful multinomial NB:

    P(c | d) proportional to  P(c) * product_t P(t | c) ** count(t, d)

Worked in log space to avoid underflow:

    log P(c | d) = log P(c) + sum_t count(t, d) * log P(t | c)

with Laplace (add-alpha) smoothing of the per-class word likelihoods:

    P(t | c) = (count(t, c) + alpha) / (total_count(c) + alpha * |V|)

No numpy, no scikit-learn -- only ``math`` and the standard library. See
EXPLAINER.md for the full derivation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass
class MultinomialNaiveBayes:
    """A multinomial Naive Bayes classifier over token sequences.

    Attributes:
        alpha: Laplace smoothing constant (``alpha > 0``). 1.0 is add-one.
        classes_: Sorted list of class labels seen during ``fit``.
        vocabulary_: Sorted list of distinct tokens seen during ``fit``.
        log_prior_: ``{class: log P(class)}``.
        log_likelihood_: ``{class: {token: log P(token | class)}}``.
        log_default_: ``{class: log P(unseen token | class)}`` used for tokens
            that never appeared in the training data for that class.
    """

    alpha: float = 1.0
    classes_: List[str] = field(default_factory=list)
    vocabulary_: List[str] = field(default_factory=list)
    log_prior_: Dict[str, float] = field(default_factory=dict)
    log_likelihood_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    log_default_: Dict[str, float] = field(default_factory=dict)
    _fitted: bool = False

    def fit(
        self,
        documents: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> "MultinomialNaiveBayes":
        """Estimate priors and likelihoods from tokenized ``documents``.

        Args:
            documents: A sequence of token lists (one per document).
            labels: Parallel sequence of class labels.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: On length mismatch, empty input, or non-positive alpha.
        """
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0 for valid smoothing")
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have equal length")
        if len(documents) == 0:
            raise ValueError("cannot fit on an empty dataset")

        # Per-class document counts (for priors) and token counts (for
        # likelihoods), plus the global vocabulary.
        class_doc_counts: Dict[str, int] = defaultdict(int)
        class_token_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        class_total_tokens: Dict[str, int] = defaultdict(int)
        vocab: set[str] = set()

        for tokens, label in zip(documents, labels):
            class_doc_counts[label] += 1
            bucket = class_token_counts[label]
            for tok in tokens:
                bucket[tok] += 1
                class_total_tokens[label] += 1
                vocab.add(tok)

        self.classes_ = sorted(class_doc_counts)
        self.vocabulary_ = sorted(vocab)
        n_docs = len(documents)
        v_size = len(self.vocabulary_)

        self.log_prior_ = {}
        self.log_likelihood_ = {}
        self.log_default_ = {}

        for c in self.classes_:
            # log P(c) = log(docs in c / total docs)
            self.log_prior_[c] = math.log(class_doc_counts[c] / n_docs)

            denom = class_total_tokens[c] + self.alpha * v_size
            counts = class_token_counts[c]
            likelihoods: Dict[str, float] = {}
            for tok in self.vocabulary_:
                numer = counts.get(tok, 0) + self.alpha
                likelihoods[tok] = math.log(numer / denom)
            self.log_likelihood_[c] = likelihoods

            # Likelihood of a token entirely unseen in training: numerator is
            # just alpha. Used defensively if a token slips outside the vocab.
            self.log_default_[c] = math.log(self.alpha / denom)

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("classifier is not fitted; call fit() first")

    def joint_log_likelihood(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Return ``{class: log P(class) + sum log P(token | class)}``.

        Tokens outside the vocabulary are ignored (their smoothed probability
        is identical across classes and cannot change the argmax), which is the
        standard multinomial NB treatment.
        """
        self._check_fitted()
        scores: Dict[str, float] = {}
        for c in self.classes_:
            total = self.log_prior_[c]
            likelihoods = self.log_likelihood_[c]
            for tok in tokens:
                if tok in likelihoods:
                    total += likelihoods[tok]
            scores[c] = total
        return scores

    def predict_log_proba(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Return normalized ``{class: log P(class | document)}``.

        Normalization uses the log-sum-exp trick for numerical stability.
        """
        jll = self.joint_log_likelihood(tokens)
        max_score = max(jll.values())
        # log sum exp = m + log sum exp(x - m)
        log_norm = max_score + math.log(
            sum(math.exp(v - max_score) for v in jll.values())
        )
        return {c: v - log_norm for c, v in jll.items()}

    def predict_proba(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Return a normalized probability distribution over classes."""
        return {c: math.exp(v) for c, v in self.predict_log_proba(tokens).items()}

    def predict(self, tokens: Sequence[str]) -> str:
        """Return the most probable class label for ``tokens``."""
        jll = self.joint_log_likelihood(tokens)
        return max(jll, key=jll.get)

    def score(
        self,
        documents: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> float:
        """Return classification accuracy over a labeled dataset."""
        if len(documents) == 0:
            raise ValueError("cannot score on empty dataset")
        correct = sum(
            1 for toks, y in zip(documents, labels) if self.predict(toks) == y
        )
        return correct / len(documents)
