"""Moderation pipeline: tokenize -> classify -> rule-eval -> score -> explain.

The :class:`ModerationPipeline` wires the from-scratch components together and
exposes a single :meth:`moderate` entry point, plus a chunked streaming helper
for incremental moderation of long inputs.

It also owns loading and training the bundled Naive Bayes classifier from
``data/train.tsv`` so callers get a ready-to-use moderator with one line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .explain import Explanation, build_explanation
from .naive_bayes import MultinomialNaiveBayes
from .policy import PolicyEngine, RuleHit
from .scoring import Decision, ScoreResult, score_text
from .tokenizer import Tokenizer

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "train.tsv",
)


def load_dataset(path: str = _DATA_PATH) -> Tuple[List[str], List[str]]:
    """Load the bundled labeled dataset from a two-column TSV.

    The file has a header ``label\\ttext`` and one example per line.

    Returns:
        ``(texts, labels)`` parallel lists.
    """
    texts: List[str] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            label, _, text = line.partition("\t")
            if not text:
                continue
            labels.append(label.strip())
            texts.append(text.strip())
    if not texts:
        raise ValueError(f"no training rows found in {path!r}")
    return texts, labels


@dataclass
class ModerationResult:
    """The full result of moderating one piece of text.

    Attributes:
        text: The input text.
        decision: ``allow`` / ``flag`` / ``block``.
        score: The underlying :class:`ScoreResult`.
        hits: Rule hits collected by the policy engine.
        explanation: Structured rationale.
    """

    text: str
    decision: Decision
    score: ScoreResult
    hits: List[RuleHit]
    explanation: Explanation

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "decision": self.decision.value,
            "score": self.score.to_dict(),
            "hits": [h.to_dict() for h in self.hits],
            "explanation": self.explanation.to_dict(),
        }


class ModerationPipeline:
    """End-to-end moderation orchestrator."""

    def __init__(
        self,
        classifier: Optional[MultinomialNaiveBayes] = None,
        policy: Optional[PolicyEngine] = None,
        tokenizer: Optional[Tokenizer] = None,
        *,
        toxic_label: str = "toxic",
        flag_threshold: float = 0.35,
        block_threshold: float = 0.70,
    ) -> None:
        """Construct a pipeline.

        If ``classifier`` is ``None``, a fresh Naive Bayes model is trained on
        the bundled dataset. Other components default to standard instances.
        """
        self.tokenizer = tokenizer or Tokenizer(ngram_range=(1, 2))
        self.policy = policy or PolicyEngine()
        self.toxic_label = toxic_label
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold

        if classifier is None:
            classifier = self._train_default_classifier()
        self.classifier = classifier

    def _train_default_classifier(self) -> MultinomialNaiveBayes:
        """Train a Naive Bayes model on the bundled dataset."""
        texts, labels = load_dataset()
        docs = [self.tokenizer.tokenize(t) for t in texts]
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(docs, labels)
        return model

    def _toxic_probability(self, tokens: Sequence[str]) -> float:
        """Return the classifier's ``P(toxic)`` for ``tokens``."""
        proba = self.classifier.predict_proba(tokens)
        return proba.get(self.toxic_label, 0.0)

    def moderate(self, text: str) -> ModerationResult:
        """Run the full pipeline on a single ``text``.

        Steps: tokenize -> classify -> evaluate rules -> score -> explain.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        tokens = self.tokenizer.tokenize(text)
        toxic_p = self._toxic_probability(tokens)
        hits = self.policy.evaluate(text)
        score = score_text(
            hits,
            toxic_p,
            flag_threshold=self.flag_threshold,
            block_threshold=self.block_threshold,
        )
        explanation = build_explanation(text, hits, score)
        return ModerationResult(
            text=text,
            decision=score.decision,
            score=score,
            hits=hits,
            explanation=explanation,
        )

    def moderate_batch(self, texts: Iterable[str]) -> List[ModerationResult]:
        """Moderate a batch of texts, preserving order."""
        return [self.moderate(t) for t in texts]

    def moderate_stream(
        self,
        chunks: Iterable[str],
        *,
        cumulative: bool = False,
    ) -> Iterator[ModerationResult]:
        """Yield a :class:`ModerationResult` per incoming chunk.

        Args:
            chunks: An iterable of text fragments (e.g. tokens of a stream).
            cumulative: When ``True``, each verdict is computed over the text
                accumulated so far (useful for catching context that only
                becomes harmful once enough has arrived). When ``False`` each
                chunk is moderated independently.

        Yields:
            One :class:`ModerationResult` per chunk.
        """
        buffer: List[str] = []
        for chunk in chunks:
            if cumulative:
                buffer.append(chunk)
                yield self.moderate(" ".join(buffer))
            else:
                yield self.moderate(chunk)
