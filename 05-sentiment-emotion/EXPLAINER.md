# EXPLAINER — Sentiment & Emotion: classifier + metrics from scratch

## What I implemented from scratch

- A **1-hidden-layer MLP classifier** with **hand-derived backpropagation** in NumPy — no autograd, no framework (`core/mlp.py`).
- **Evaluation metrics**: confusion matrix, per-class precision/recall/F1, macro-F1 (`core/metrics.py`).
- **Calibration**: temperature scaling + **Expected Calibration Error** and reliability-diagram data (`core/metrics.py`).

The pretrained RoBERTa/DistilBERT classifiers remain available in `analyzer.py`; the from-scratch MLP is the "I understand how a classifier learns" counterpart, trainable on top of embeddings.

## The backprop, by hand (`core/mlp.py`)

Forward: `z1 = XW1+b1 → a1 = relu(z1) → z2 = a1W2+b2 → softmax`. Then the gradients, derived and coded directly:

- `dz2 = (softmax − onehot)/n` — the clean gradient of softmax+cross-entropy.
- `dW2 = a1ᵀ dz2`, `db2 = Σ dz2`
- `da1 = dz2 W2ᵀ`, `dz1 = da1 ⊙ (z1>0)` — the ReLU gradient gates by activation.
- `dW1 = Xᵀ dz1`, `db1 = Σ dz1`

If this is right, the network solves XOR — the textbook proof that a hidden layer + correct backprop can learn a non-linearly-separable function.

## Why calibration matters

A model can be accurate but *overconfident* (says 99% when it's right 80% of the time). **Temperature scaling** divides logits by T before softmax to fix this; **ECE** measures the average gap between confidence and accuracy across bins. These are exactly the diagnostics you'd run before trusting a classifier's probabilities in production.

## Proof it works

`tests/test_core.py`:
- The MLP **learns XOR** exactly (the non-linear-separability test) with decreasing loss.
- Confusion matrix and F1 compute correctly.
- Temperature scaling softens confidence as T rises.
- ECE is near zero for well-calibrated predictions.

## Limitations

- The from-scratch MLP trains on embedding features; shipping a labeled dataset to train it in the app is a natural next step (the code is ready).
- It won't beat the pretrained transformer on raw accuracy — the point is the mechanism and the honest evaluation around it.
