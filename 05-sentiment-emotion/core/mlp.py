"""A small neural-network classifier in NumPy — forward and backprop by hand.

A 1-hidden-layer MLP with softmax output, trained by manual backpropagation
(no autograd, no framework). This is the "I can implement backprop" companion
to the pretrained sentiment model: same task, but here the gradients are
derived and coded by hand.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    def __init__(self, in_dim: int, hidden: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialization for the ReLU layer
        self.W1 = rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, n_classes)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_classes)
        self.history = []

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)          # ReLU
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)
        cache = (X, z1, a1, probs)
        return probs, cache

    def fit(self, X, y, epochs=300, lr=0.1, l2=1e-4):
        n, k = X.shape[0], self.W2.shape[1]
        Y = np.eye(k)[y]               # one-hot targets
        for _ in range(epochs):
            probs, (X_, z1, a1, _) = self._forward(X)
            # cross-entropy loss
            loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
            self.history.append(loss)

            # ---- backprop, derived by hand ----
            dz2 = (probs - Y) / n                       # dL/dz2
            dW2 = a1.T @ dz2 + l2 * self.W2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)                         # ReLU gradient
            dW1 = X_.T @ dz1 + l2 * self.W1
            db1 = dz1.sum(axis=0)

            # SGD update
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1
        return self

    def predict_proba(self, X):
        return self._forward(X)[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
