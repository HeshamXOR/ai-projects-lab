# ⚙️ nanograd — autodiff from scratch

**What I implemented from scratch:** a reverse-mode automatic differentiation engine over NumPy tensors (the core of PyTorch's `autograd`), plus a small neural-net library (Linear, MLP, SGD with momentum) built on top — **no PyTorch, no TensorFlow, no autodiff library.**

![preview](preview.gif)
<!-- Record the live-training demo on Lightning and save as preview.gif. -->

## Why this matters

Most "AI projects" call `model.fit()`. This one *is* the thing underneath `model.fit()`. It demonstrates that I understand backpropagation, the chain rule, and computational graphs at the level where I can build them — not just use them.

## What's inside

| File | What it is |
|------|-----------|
| `nanograd/engine.py` | The `Tensor` class: records a computation graph and walks it backward (topological sort + chain rule) to compute gradients. Handles broadcasting, matmul, reductions, and a fused numerically-stable softmax-cross-entropy. |
| `nanograd/nn.py` | `Linear`, `MLP`, and an `SGD` optimizer with momentum — a mini PyTorch-shaped API where every gradient flows through my engine. |
| `app.py` | A Gradio demo that trains a net live on 2D datasets and plots the loss curve + decision boundary. |
| `tests/` | **Gradient checks** (analytic vs. numerical finite differences) for every op, plus an end-to-end training test. |

## The core idea (3 rules)

1. Every `Tensor` remembers the tensors it came from and a `_backward` closure that knows how to push gradient to those parents.
2. `.backward()` topologically sorts the graph and calls each node's `_backward` in reverse, accumulating into `.grad`.
3. The chain rule is applied **locally** at each op — we never hand-derive a global gradient.

The trickiest part — and the most-tested — is `_unbroadcast`: when an op broadcasts (e.g. `(3,4) + (4,)`), the gradient to the smaller operand must be summed back down to its shape. Get this wrong and everything silently learns garbage; it lives in one well-tested function.

## Proof it's correct

```bash
pip install -r requirements.txt
pytest -q          # gradient checks: analytic gradients match numerical to ~1e-4
```

`tests/test_gradcheck.py` compares every operation's analytic gradient against a central-difference numerical gradient — exactly how real autodiff libraries are validated. `tests/test_training.py` proves the full stack learns (>90% accuracy on two-moons).

## Run the demo

```bash
python app.py      # http://localhost:7860 (+ public gradio.live link on Lightning)
```

Pick a dataset (moons / circles / xor), adjust the network, and watch the from-scratch engine drive the loss down and carve out a decision boundary.

## Limitations & next steps

- CPU + dense tensors only; no GPU kernels (the point is clarity, not speed).
- A natural next step is using this engine to power a tiny classifier elsewhere in the lab, or adding conv/attention ops. The sibling project **microgpt** takes the "from scratch" idea up to a transformer.
