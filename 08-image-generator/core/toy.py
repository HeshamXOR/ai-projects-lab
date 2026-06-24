"""Toy diffusion on 2D data — the whole diffusion loop, visible and from scratch.

Stable Diffusion is too big to see what's happening. Here we train a tiny MLP
noise-predictor (in NumPy, with hand-written backprop) on a 2D point cloud
(e.g. two moons / a spiral), then sample new points by running the reverse
diffusion process. You can literally watch Gaussian noise turn into the target
shape — the same principle SD uses on image latents, stripped to 2 dimensions.
"""

from __future__ import annotations

import numpy as np

from .ddim import Diffusion


def make_spiral(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.random(n)) * 3 * np.pi
    x = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    x = x / x.std(0)
    return x + rng.normal(0, 0.05, x.shape)


def make_two_moons(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    a = np.linspace(0, np.pi, n2)
    outer = np.stack([np.cos(a), np.sin(a)], axis=1)
    inner = np.stack([1 - np.cos(a), 1 - np.sin(a) - 0.5], axis=1)
    x = np.vstack([outer, inner])
    x = (x - x.mean(0)) / x.std(0)
    return x + rng.normal(0, 0.05, x.shape)


def _time_embed(t_norm, dim=16):
    """Sinusoidal-ish embedding of the (normalized) timestep."""
    freqs = np.arange(1, dim // 2 + 1)
    ang = t_norm[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class NoisePredictor:
    """A small MLP: (x_t, t) -> predicted noise. NumPy + hand-written backprop."""

    def __init__(self, hidden=128, t_dim=16, seed=0):
        rng = np.random.default_rng(seed)
        in_dim = 2 + t_dim
        self.t_dim = t_dim
        self.W1 = rng.standard_normal((in_dim, hidden)) * np.sqrt(2 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * np.sqrt(2 / hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((hidden, 2)) * np.sqrt(2 / hidden)
        self.b3 = np.zeros(2)

    def _feats(self, x, t_norm):
        return np.concatenate([x, _time_embed(t_norm, self.t_dim)], axis=1)

    def forward(self, x, t_norm):
        f = self._feats(x, t_norm)
        z1 = f @ self.W1 + self.b1; a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(0, z2)
        out = a2 @ self.W3 + self.b3
        return out, (f, z1, a1, z2, a2)

    def train(self, data, diffusion: Diffusion, epochs=2000, batch=256, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        T = diffusion.T
        losses = []
        for ep in range(epochs):
            idx = rng.integers(0, len(data), batch)
            x0 = data[idx]
            t = rng.integers(0, T, batch)
            noise = rng.standard_normal(x0.shape)
            ab = diffusion.alpha_bars[t][:, None]
            xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise
            t_norm = t / T

            pred, (f, z1, a1, z2, a2) = self.forward(xt, t_norm)
            diff = pred - noise
            loss = np.mean(diff ** 2)
            losses.append(loss)

            # backprop (MSE through the 3-layer ReLU net)
            n = batch
            dout = 2 * diff / n
            dW3 = a2.T @ dout; db3 = dout.sum(0)
            da2 = dout @ self.W3.T; dz2 = da2 * (z2 > 0)
            dW2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
            dW1 = f.T @ dz1; db1 = dz1.sum(0)

            for p, g in [(self.W1, dW1), (self.b1, db1), (self.W2, dW2),
                         (self.b2, db2), (self.W3, dW3), (self.b3, db3)]:
                p -= lr * g
        return losses

    def predict_noise(self, xt, t):
        t_norm = np.full(len(xt), t / 1.0)  # caller passes normalized t
        out, _ = self.forward(xt, t_norm)
        return out


def sample(model: NoisePredictor, diffusion: Diffusion, n=500, steps=50, seed=0):
    """Reverse diffusion with DDIM: start from noise, denoise to data."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    T = diffusion.T
    ts = np.linspace(T - 1, 0, steps).astype(int)
    for i, t in enumerate(ts):
        t_norm = np.full(n, t / T)
        eps, _ = model.forward(x, t_norm)
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        x = diffusion.ddim_step(x, eps, t, t_prev)
    return x
