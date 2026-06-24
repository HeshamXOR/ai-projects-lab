"""Proofs for attention rollout and Grad-CAM."""

import numpy as np

from core.attention_rollout import (
    attention_rollout, average_heads, saliency_to_grid,
)
from core.gradcam import grad_cam, upscale_nearest


def test_rollout_rows_are_distributions():
    rng = np.random.default_rng(0)
    layers = []
    for _ in range(3):
        a = rng.random((5, 5))
        a = a / a.sum(axis=1, keepdims=True)
        layers.append(a)
    roll = attention_rollout(layers)
    # each row should still sum to 1 (it's a composition of stochastic matrices)
    np.testing.assert_allclose(roll.sum(axis=1), np.ones(5), atol=1e-6)


def test_rollout_identity_attention_is_identity():
    # if every layer attends only to itself, rollout is the identity
    I = np.eye(4)
    roll = attention_rollout([I, I, I])
    np.testing.assert_allclose(roll, I, atol=1e-6)


def test_average_heads():
    a = np.stack([np.ones((3, 3)), np.zeros((3, 3))])  # 2 heads
    np.testing.assert_allclose(average_heads(a), np.full((3, 3), 0.5))


def test_gradcam_highlights_high_gradient_channel():
    # channel 0 has strong positive gradient AND a hot spot in the top-left;
    # channel 1 has zero gradient. Grad-CAM should highlight the top-left.
    C, H, W = 2, 4, 4
    activations = np.zeros((C, H, W))
    activations[0, 0, 0] = 1.0          # channel-0 hot spot
    activations[1] = 1.0                # channel-1 uniformly on
    gradients = np.zeros((C, H, W))
    gradients[0] = 1.0                  # only channel 0 matters
    cam = grad_cam(activations, gradients)
    assert cam.shape == (H, W)
    assert cam[0, 0] == cam.max() and cam.max() == 1.0


def test_gradcam_relu_clips_negatives():
    activations = np.ones((1, 2, 2))
    gradients = -np.ones((1, 2, 2))     # negative -> ReLU should zero it out
    cam = grad_cam(activations, gradients)
    assert np.all(cam == 0)


def test_upscale_preserves_corners():
    cam = np.array([[0.0, 1.0], [1.0, 0.0]])
    up = upscale_nearest(cam, 4, 4)
    assert up.shape == (4, 4)
    assert up[0, 0] == 0.0 and up[0, -1] == 1.0


def test_saliency_to_grid_square():
    sal = np.arange(16.0)
    grid = saliency_to_grid(sal)
    assert grid.shape == (4, 4)
    assert 0.0 <= grid.min() and grid.max() <= 1.0
