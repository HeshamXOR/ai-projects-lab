"""Gradient checking: the proof that the autodiff engine is correct.

For each op we compare the engine's analytic gradient against a numerical
finite-difference gradient. If reverse-mode backprop is implemented correctly,
they agree to ~1e-6. This is exactly how real autodiff libraries are tested.
"""

import numpy as np
import pytest

from nanograd.engine import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference numerical gradient of scalar f wrt array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def _check(make_loss, x0, rng):
    """Compare analytic vs numerical grad for loss = make_loss(Tensor(x))."""
    # analytic
    xt = Tensor(x0.copy())
    loss = make_loss(xt)
    loss.backward()
    analytic = xt.grad
    # numerical
    num = numerical_grad(lambda a: float(make_loss(Tensor(a)).data), x0.copy())
    np.testing.assert_allclose(analytic, num, rtol=1e-4, atol=1e-6)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_add_mul(rng):
    x = rng.standard_normal((3, 4))
    _check(lambda t: (t * 2.0 + 1.0).sum(), x, rng)


def test_pow(rng):
    x = rng.standard_normal((4,)) + 3  # keep positive-ish
    _check(lambda t: (t ** 3).sum(), x, rng)


def test_relu(rng):
    x = rng.standard_normal((5, 5))
    _check(lambda t: t.relu().sum(), x, rng)


def test_tanh(rng):
    x = rng.standard_normal((5,))
    _check(lambda t: t.tanh().sum(), x, rng)


def test_exp_log(rng):
    x = rng.standard_normal((4,)) + 2  # positive for log
    _check(lambda t: t.exp().sum(), x, rng)
    _check(lambda t: t.log().sum(), x, rng)


def test_matmul(rng):
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))
    Bt = Tensor(B)

    def make(t):
        return (t @ Bt).sum()

    _check(make, A, rng)


def test_broadcasting_add(rng):
    # (3,4) + (4,) must unbroadcast the bias gradient correctly
    x = rng.standard_normal((3, 4))
    bias = Tensor(rng.standard_normal((4,)))
    _check(lambda t: (t + bias).sum(), x, rng)


def test_mean(rng):
    x = rng.standard_normal((6,))
    _check(lambda t: t.mean(), x, rng)


def test_softmax_cross_entropy(rng):
    logits = rng.standard_normal((4, 3))
    targets = np.array([0, 2, 1, 0])
    _check(lambda t: t.softmax_cross_entropy(targets), logits, rng)


def test_chained_expression(rng):
    # a deeper graph exercising accumulation through shared nodes
    x = rng.standard_normal((3, 3))

    def make(t):
        y = (t * t).relu()
        z = (y + t).tanh()
        return z.sum()

    _check(make, x, rng)
