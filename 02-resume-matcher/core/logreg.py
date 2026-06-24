"""Logistic regression trained with gradient descent — from scratch.

The math of binary classification: model P(y=1|x) = sigmoid(w·x + b), and fit w,
b by minimizing binary cross-entropy via gradient descent. No scikit-learn — the
forward pass, the loss, the gradients, and the update are all written out, so
this demonstrates understanding of how a linear classifier actually learns.

Used by the resume matcher to score "is this resume a match for this job?" from
features (TF-IDF cosine, skill overlap, length ratios) instead of a hand-tuned
weighted sum.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    def __init__(self, lr: float = 0.5, epochs: int = 500, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray = None
        self.b: float = 0.0
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = sigmoid(z)
            # binary cross-entropy gradient
            error = p - y                      # dL/dz
            grad_w = X.T @ error / n + self.l2 * self.w
            grad_b = error.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            # track loss
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.history.append(loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
