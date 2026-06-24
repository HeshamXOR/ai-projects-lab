"""nanograd — live demo.

Trains a neural network in your browser using a from-scratch autodiff engine
(no PyTorch/TensorFlow). Watch the loss fall and the decision boundary form on
classic 2D datasets. The point: every gradient here flows through code in
nanograd/engine.py that I wrote by hand.
"""

from __future__ import annotations

import io

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanograd.engine import Tensor
from nanograd.nn import MLP, SGD


def make_dataset(kind: str, n=240, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    if kind == "moons":
        t = np.linspace(0, np.pi, n2)
        outer = np.stack([np.cos(t), np.sin(t)], axis=1)
        inner = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
        X = np.vstack([outer, inner]) + rng.normal(0, noise, (n2 * 2, 2))
        y = np.array([0] * n2 + [1] * n2)
    elif kind == "circles":
        t = np.linspace(0, 2 * np.pi, n2)
        inner = 0.4 * np.stack([np.cos(t), np.sin(t)], axis=1)
        outer = 1.0 * np.stack([np.cos(t), np.sin(t)], axis=1)
        X = np.vstack([inner, outer]) + rng.normal(0, noise * 0.6, (n2 * 2, 2))
        y = np.array([0] * n2 + [1] * n2)
    else:  # xor-ish blobs
        c = rng.normal(0, noise, (n2 * 2, 2))
        q = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        pts = np.repeat(q, n // 4, axis=0)[: n2 * 2]
        X = pts + c
        y = np.array([0, 0, 1, 1]).repeat(n // 4)[: n2 * 2]
    return X, y


def train_and_plot(dataset, hidden, lr, epochs):
    X, y = make_dataset(dataset)
    sizes = [2] + [int(hidden)] * 2 + [2]
    model = MLP(sizes, rng=np.random.default_rng(0))
    opt = SGD(model.parameters(), lr=float(lr), momentum=0.9)
    Xt = Tensor(X)

    losses = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(Xt)
        loss = logits.softmax_cross_entropy(y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    preds = np.argmax(model(Xt).data, axis=1)
    acc = (preds == y).mean()

    # decision boundary grid
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.argmax(model(Tensor(grid)).data, axis=1).reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(losses, color="#7c5cff", lw=2)
    ax1.set_title("Training loss (cross-entropy)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(alpha=0.2)
    ax2.contourf(xx, yy, zz, alpha=0.25, levels=1, colors=["#34d6df", "#7c5cff"])
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="cool", edgecolors="k", s=22)
    ax2.set_title(f"Decision boundary — accuracy {acc:.0%}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    img = Image.open(buf)
    report = (
        f"**Final loss:** {losses[-1]:.4f}  ·  **Accuracy:** {acc:.1%}  ·  "
        f"**Params:** {sum(p.data.size for p in model.parameters())}  ·  "
        f"trained with from-scratch reverse-mode autodiff (no PyTorch)."
    )
    return img, report


with gr.Blocks(title="nanograd", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ⚙️ nanograd — autodiff from scratch\n"
        "This trains a neural network using a **reverse-mode automatic "
        "differentiation engine I wrote from scratch in NumPy** — the same idea "
        "behind PyTorch's `autograd`, no deep-learning framework involved. "
        "Pick a dataset and watch it learn."
    )
    with gr.Row():
        dataset = gr.Dropdown(["moons", "circles", "xor"], value="moons", label="Dataset")
        hidden = gr.Slider(4, 64, value=16, step=4, label="Hidden units / layer")
        lr = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Learning rate")
        epochs = gr.Slider(50, 800, value=300, step=50, label="Epochs")
    btn = gr.Button("Train", variant="primary")
    plot = gr.Image(label="Result")
    report = gr.Markdown()
    btn.click(train_and_plot, [dataset, hidden, lr, epochs], [plot, report])
    demo.load(train_and_plot, [dataset, hidden, lr, epochs], [plot, report])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
