"""nanograd — a reverse-mode automatic differentiation engine over NumPy arrays.

This is the thing PyTorch does under the hood, written from scratch: a `Tensor`
that records the operations performed on it into a computation graph, then walks
that graph backward (reverse-mode autodiff / backprop) to compute gradients.

Unlike micrograd (scalar-valued), nanograd is *tensor-valued* — every op is a
NumPy array op, with correctly broadcasting gradients. That's enough to build
and train real multi-layer networks (see nn.py).

The whole idea in three rules:
  1. Every Tensor remembers the Tensors it came from (`_prev`) and a closure
     (`_backward`) that knows how to push gradient to those parents.
  2. To differentiate, we topologically sort the graph and call each node's
     `_backward` in reverse order, accumulating into `.grad`.
  3. The chain rule is applied locally at each op — we never write the global
     derivative, only "given dL/d(out), what is dL/d(inputs)?"
"""

from __future__ import annotations

from typing import Callable, Set, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int]


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum `grad` back down to `shape`, reversing NumPy broadcasting.

    When a+b broadcasts (e.g. (3,4) + (4,) -> (3,4)), the gradient flowing back
    to the smaller operand must be summed over the broadcast axes so its shape
    matches. This is the single most error-prone part of tensor autodiff, so it
    lives in one well-tested place.
    """
    # 1) sum over leading axes that were added by broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) sum over axes that were size-1 in the original but expanded
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that tracks gradients."""

    def __init__(self, data: ArrayLike, _children: Tuple["Tensor", ...] = (), _op: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set(_children)
        self._op = _op

    # ---- helpers ----
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    @staticmethod
    def _t(x: ArrayLike) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- elementwise ops ----
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._t(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._t(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, p: float) -> "Tensor":
        assert isinstance(p, (int, float))
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = self._t(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # dL/dA = dL/dout @ B^T ;  dL/dB = A^T @ dL/dout
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    # ---- activations / unary ----
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---- reductions ----
    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def softmax_cross_entropy(self, targets: np.ndarray) -> "Tensor":
        """Numerically-stable softmax + cross-entropy in one fused op.

        `targets` is an array of integer class indices, shape (batch,).
        Returns the mean loss over the batch. Fusing the two means a clean,
        stable backward pass: dL/dlogits = (softmax - onehot) / batch.
        """
        logits = self.data
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        loss_val = -np.log(probs[np.arange(n), targets] + 1e-12).mean()
        out = Tensor(loss_val, (self,), "softmax_xent")

        def _backward():
            grad = probs.copy()
            grad[np.arange(n), targets] -= 1
            grad /= n
            self.grad += grad * out.grad

        out._backward = _backward
        return out

    # ---- arithmetic sugar ----
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._t(other))

    def __rsub__(self, other):
        return self._t(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (self._t(other) ** -1)

    # ---- the engine ----
    def backward(self):
        """Reverse-mode autodiff: topologically sort, then chain-rule backward."""
        topo = []
        visited = set()

        def build(v: "Tensor"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        # seed: dL/dL = 1
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
