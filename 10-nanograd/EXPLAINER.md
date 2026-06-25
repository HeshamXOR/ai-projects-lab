# EXPLAINER — nanograd

## What I implemented from scratch

A **reverse-mode automatic differentiation engine** (the core of PyTorch's autograd) and a small neural-network library on top — in pure NumPy, no deep-learning framework.

## How it works

- **`engine.py` — the autograd `Tensor`.** Every operation (add, mul, matmul, relu, etc.) records its inputs and a local gradient function, building a dynamic computation graph. Calling `.backward()` does a topological sort of the graph and applies the chain rule in reverse, accumulating `.grad` on every node. This is reverse-mode autodiff: one backward pass computes the gradient of a scalar loss with respect to every parameter.
- **`nn.py` — layers and optimizers.** `Linear`, activation functions, an MLP container, and an SGD optimizer, all built only on the `Tensor` primitive.

## Proof it works

The decisive test is **gradient checking**: for each operation, the analytic gradient produced by the engine is compared against a numerical finite-difference gradient. They agree to within ~1e-6, which is how real autodiff libraries are validated. The MLP then trains on a toy dataset and the loss decreases as expected.

Run `pytest` to see the gradient checks pass.

The README contains the fuller walk-through of the design and the math; this file records the one-line answer to "what did you actually build": **the autograd engine itself.**
