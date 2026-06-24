"""Classification metrics & calibration — from scratch.

Confusion matrix, per-class precision/recall/F1, and temperature scaling for
probability calibration, plus the data for a reliability diagram. These are the
numbers you actually report when evaluating a classifier — implemented directly
rather than imported from sklearn.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "macro_f1": float(f1.mean())}


def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    """Divide logits by a temperature T before softmax. T>1 softens
    (less overconfident) probabilities; T<1 sharpens them."""
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y_true, n_bins: int = 10) -> Tuple[float, list]:
    """ECE: average gap between confidence and accuracy across confidence bins.
    Also returns per-bin (confidence, accuracy, count) for a reliability diagram.
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    diagram = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            diagram.append((float((lo + hi) / 2), 0.0, 0))
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
        diagram.append((bin_conf, bin_acc, count))
    return ece, diagram
