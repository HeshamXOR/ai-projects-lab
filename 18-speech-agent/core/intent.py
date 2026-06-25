"""Intent classifier from scratch: TF-IDF vectorizer + logistic regression.

No sklearn for the model. Two pieces, both implemented by hand:

1. **TfidfVectorizer** — builds a vocabulary, computes term frequency per
   document, inverse document frequency ``idf = log((1 + N) / (1 + df)) + 1``
   (smoothed, sklearn-style), multiplies tf * idf, and L2-normalizes each row.
2. **LogisticRegression** — multinomial (softmax) classifier trained by batch
   gradient descent in NumPy: forward pass -> softmax -> cross-entropy loss ->
   analytic gradient -> weight update. Falls back to one-vs-rest sigmoids only
   conceptually; here softmax handles the multiclass case directly.

``IntentClassifier`` wires the two together: ``fit(texts, labels)``,
``predict(text) -> label``, and ``predict_proba(text) -> {label: prob}``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase and split into word tokens."""
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """Hand-rolled TF-IDF vectorizer with smoothed idf and L2 row normalization."""

    def __init__(self) -> None:
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.empty(0, dtype=np.float64)

    def fit(self, documents: Sequence[str]) -> "TfidfVectorizer":
        """Learn the vocabulary and idf weights from ``documents``."""
        vocab: Dict[str, int] = {}
        doc_tokens: List[List[str]] = []
        for doc in documents:
            toks = tokenize(doc)
            doc_tokens.append(toks)
            for tok in set(toks):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocabulary_ = vocab

        n_docs = len(documents)
        n_terms = len(vocab)
        df = np.zeros(n_terms, dtype=np.float64)
        for toks in doc_tokens:
            for tok in set(toks):
                df[vocab[tok]] += 1.0
        # smoothed idf, sklearn convention: log((1 + N) / (1 + df)) + 1
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Map ``documents`` to L2-normalized TF-IDF row vectors."""
        if not self.vocabulary_:
            raise ValueError("TfidfVectorizer must be fit before transform.")
        n_terms = len(self.vocabulary_)
        matrix = np.zeros((len(documents), n_terms), dtype=np.float64)
        for row, doc in enumerate(documents):
            toks = tokenize(doc)
            if not toks:
                continue
            # raw term frequency
            for tok in toks:
                idx = self.vocabulary_.get(tok)
                if idx is not None:
                    matrix[row, idx] += 1.0
            # tf * idf
            matrix[row] *= self.idf_
            # L2 normalize the row
            norm = np.linalg.norm(matrix[row])
            if norm > 0.0:
                matrix[row] /= norm
        return matrix

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(documents).transform(documents)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid (kept for the binary/one-vs-rest case)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    """Multinomial logistic regression trained by batch gradient descent.

    Weights ``W`` have shape ``(n_features, n_classes)`` and bias ``b`` has shape
    ``(n_classes,)``. Training minimizes mean cross-entropy with L2 regularization.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iters: int = 800,
        l2: float = 1e-4,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2
        self.W: np.ndarray = np.empty(0)
        self.b: np.ndarray = np.empty(0)
        self.classes_: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit on feature matrix ``X`` and integer-coded labels ``y``."""
        n_samples, n_features = X.shape
        self.classes_ = sorted(set(int(v) for v in y))
        n_classes = len(self.classes_)
        class_to_col = {c: i for i, c in enumerate(self.classes_)}

        # one-hot targets
        Y = np.zeros((n_samples, n_classes), dtype=np.float64)
        for row, label in enumerate(y):
            Y[row, class_to_col[int(label)]] = 1.0

        rng = np.random.default_rng(0)
        self.W = rng.normal(scale=0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes, dtype=np.float64)

        for _ in range(self.n_iters):
            logits = X @ self.W + self.b
            probs = softmax(logits)
            # gradient of mean cross-entropy w.r.t. logits is (probs - Y) / n
            error = (probs - Y) / n_samples
            grad_W = X.T @ error + self.l2 * self.W
            grad_b = error.sum(axis=0)
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability rows for each sample in ``X``."""
        return softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted integer class labels for ``X``."""
        cols = self.predict_proba(X).argmax(axis=1)
        return np.array([self.classes_[c] for c in cols], dtype=int)

    def cross_entropy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy loss (useful for monitoring convergence)."""
        class_to_col = {c: i for i, c in enumerate(self.classes_)}
        probs = self.predict_proba(X)
        eps = 1e-12
        losses = [
            -np.log(probs[row, class_to_col[int(label)]] + eps)
            for row, label in enumerate(y)
        ]
        return float(np.mean(losses))


class IntentClassifier:
    """End-to-end intent classifier: text -> TF-IDF -> logistic regression -> label."""

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iters: int = 800,
        l2: float = 1e-4,
    ) -> None:
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(
            learning_rate=learning_rate, n_iters=n_iters, l2=l2
        )
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: Dict[int, str] = {}

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "IntentClassifier":
        """Train on parallel ``texts`` and string ``labels``."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length.")
        unique = sorted(set(labels))
        self._label_to_id = {lab: i for i, lab in enumerate(unique)}
        self._id_to_label = {i: lab for lab, i in self._label_to_id.items()}

        X = self.vectorizer.fit_transform(texts)
        y = np.array([self._label_to_id[lab] for lab in labels], dtype=int)
        self.model.fit(X, y)
        return self

    def predict(self, text: str) -> str:
        """Return the most likely intent label for a single ``text``."""
        X = self.vectorizer.transform([text])
        pred_id = int(self.model.predict(X)[0])
        return self._id_to_label[pred_id]

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return a ``{label: probability}`` mapping for a single ``text``."""
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        return {
            self._id_to_label[self.model.classes_[col]]: float(probs[col])
            for col in range(len(self.model.classes_))
        }
