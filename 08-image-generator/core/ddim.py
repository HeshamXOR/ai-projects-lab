"""Diffusion schedules and the DDPM/DDIM sampling math — from scratch.

This module implements the *math* of diffusion models independent of any
particular network:

  * the noise schedule (betas) and the derived alphas / cumulative products,
  * the forward process q(x_t | x_0): how to add t steps of noise in closed form,
  * the DDPM reverse step (stochastic) and the DDIM reverse step (deterministic,
    far fewer steps),
  * classifier-free guidance mixing.

A model that predicts the noise ε is the only learned piece; everything about
*how to use* that prediction to denoise is here. `toy.py` plugs a tiny network
into these functions to show the whole loop working on 2D data.
"""

from __future__ import annotations

import numpy as np


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02) -> np.ndarray:
    return np.linspace(beta_start, beta_end, T)


class Diffusion:
    def __init__(self, T: int = 200):
        self.T = T
        self.betas = linear_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)  # ᾱ_t = Π α_s

    # ---- forward process: add noise in closed form ----
    def q_sample(self, x0: np.ndarray, t: int, noise: np.ndarray = None):
        """Sample x_t ~ q(x_t | x_0) = √ᾱ_t · x0 + √(1-ᾱ_t) · ε."""
        if noise is None:
            noise = np.random.standard_normal(x0.shape)
        ab = self.alpha_bars[t]
        return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise, noise

    # ---- DDPM reverse step (stochastic) ----
    def ddpm_step(self, xt, eps_pred, t, rng):
        """One step of the stochastic reverse process p(x_{t-1} | x_t)."""
        beta = self.betas[t]
        alpha = self.alphas[t]
        ab = self.alpha_bars[t]
        # posterior mean using the predicted noise
        mean = (xt - beta / np.sqrt(1 - ab) * eps_pred) / np.sqrt(alpha)
        if t == 0:
            return mean
        noise = rng.standard_normal(xt.shape)
        return mean + np.sqrt(beta) * noise

    # ---- DDIM reverse step (deterministic, skip steps) ----
    def ddim_step(self, xt, eps_pred, t, t_prev):
        """Deterministic DDIM update from timestep t to an earlier t_prev.

        Predicts x0 from (xt, eps), then re-noises to t_prev with no randomness.
        This is what lets DDIM use ~20 steps instead of hundreds.
        """
        ab_t = self.alpha_bars[t]
        ab_prev = self.alpha_bars[t_prev] if t_prev >= 0 else 1.0
        x0_pred = (xt - np.sqrt(1 - ab_t) * eps_pred) / np.sqrt(ab_t)
        return np.sqrt(ab_prev) * x0_pred + np.sqrt(1 - ab_prev) * eps_pred


def classifier_free_guidance(eps_uncond, eps_cond, scale: float):
    """Push the prediction toward the conditioned direction:
    ε = ε_uncond + scale·(ε_cond − ε_uncond). scale=0 ignores the condition."""
    return eps_uncond + scale * (eps_cond - eps_uncond)
