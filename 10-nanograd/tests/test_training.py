"""End-to-end: the engine can actually train a network to fit a nonlinear set.

If autodiff + SGD work, an MLP should drive cross-entropy loss down and classify
the two-moons dataset well above chance. This is the integration test that the
whole stack (engine + nn + optimizer) composes.
"""

import numpy as np

from nanograd.engine import Tensor
from nanograd.nn import MLP, SGD


def make_moons(n=200, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    t = np.linspace(0, np.pi, n2)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (n2 * 2, 2))
    y = np.array([0] * n2 + [1] * n2)
    return X, y


def test_mlp_learns_two_moons():
    X, y = make_moons(n=200, seed=1)
    model = MLP([2, 16, 16, 2], rng=np.random.default_rng(0))
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9)

    Xt = Tensor(X)
    first_loss = None
    for step in range(300):
        opt.zero_grad()
        logits = model(Xt)
        loss = logits.softmax_cross_entropy(y)
        loss.backward()
        opt.step()
        if step == 0:
            first_loss = float(loss.data)
    final_loss = float(loss.data)

    # loss decreased substantially
    assert final_loss < first_loss * 0.5

    # accuracy well above chance
    preds = np.argmax(model(Xt).data, axis=1)
    acc = (preds == y).mean()
    assert acc > 0.90, f"accuracy too low: {acc}"
