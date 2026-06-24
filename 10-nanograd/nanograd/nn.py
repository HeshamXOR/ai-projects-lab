"""A tiny neural-network library built on the nanograd autodiff engine.

Mirrors the PyTorch API shape (Module / Linear / SGD) so the code reads
familiarly, but every gradient flows through the from-scratch engine — there is
no PyTorch here at all.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .engine import Tensor


class Module:
    def parameters(self) -> List[Tensor]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """y = x @ W + b, with Kaiming-ish init."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(rng.standard_normal((in_features, out_features)) * scale)
        self.b = Tensor(np.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP(Module):
    """A multi-layer perceptron with ReLU activations."""

    def __init__(self, sizes: List[int], rng: np.random.Generator = None):
        rng = rng or np.random.default_rng(0)
        self.layers = [Linear(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, params: List[Tensor], lr: float = 0.05, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            self._velocity[i] = self.momentum * self._velocity[i] - self.lr * p.grad
            p.data += self._velocity[i]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
