"""Proofs for the from-scratch MLP classifier and metrics."""

import numpy as np

from core.mlp import MLPClassifier
from core.metrics import (
    confusion_matrix,
    precision_recall_f1,
    temperature_scale,
    expected_calibration_error,
)


def test_mlp_learns_xor():
    # XOR is the canonical non-linearly-separable problem; a 1-hidden-layer
    # MLP must solve it, proving the hidden layer + backprop work.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    # repeat for a bit of gradient signal
    Xb = np.tile(X, (50, 1)) + np.random.default_rng(0).normal(0, 0.05, (200, 2))
    yb = np.tile(y, 50)
    clf = MLPClassifier(2, 8, 2, seed=1).fit(Xb, yb, epochs=800, lr=0.2)
    assert (clf.predict(X) == y).all()
    assert clf.history[-1] < clf.history[0]


def test_confusion_and_f1():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.sum() == 6
    assert cm[1, 1] == 2
    m = precision_recall_f1(cm)
    assert 0 <= m["macro_f1"] <= 1


def test_temperature_softens():
    logits = np.array([[3.0, 0.0, 0.0]])
    sharp = temperature_scale(logits, 0.5).max()
    base = temperature_scale(logits, 1.0).max()
    soft = temperature_scale(logits, 2.0).max()
    assert sharp > base > soft  # higher T -> less confident


def test_ece_perfect_calibration():
    # predictions that are right exactly as often as their confidence -> low ECE
    probs = np.array([[0.9, 0.1]] * 10)
    y = [0] * 9 + [1]  # 90% accuracy at 90% confidence
    ece, diagram = expected_calibration_error(probs, y, n_bins=10)
    assert ece < 0.05
