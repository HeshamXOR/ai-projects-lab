# EXPLAINER — Image Generator: understanding diffusion

## What I implemented from scratch

- The **diffusion math**: noise schedule, the forward process `q(x_t|x_0)`, the **DDPM** (stochastic) and **DDIM** (deterministic, few-step) reverse steps, and classifier-free guidance (`core/ddim.py`).
- A **toy diffusion model**: a tiny NumPy noise-predictor with hand-written backprop, trained on 2D point clouds, sampled with the DDIM loop (`core/toy.py`).

Stable Diffusion remains the high-quality image path; this is the from-scratch demonstration that I understand what a diffusion sampler actually does — visible in 2D where you can watch it happen.

## How diffusion works (the math in `ddim.py`)

- **Forward process**: gradually add Gaussian noise over `T` steps. The key trick is the closed form `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`, where `ᾱ_t` is the cumulative product of `(1−β)`. So we can jump to any noise level in one step during training.
- **Training target**: a network learns to predict the noise `ε` that was added.
- **DDPM reverse**: undo one noise step at a time using the predicted ε — stochastic, needs many steps.
- **DDIM reverse**: predict `x₀` from `(x_t, ε)`, then deterministically re-noise to an earlier timestep. Because it's non-Markovian, you can **skip** timesteps — ~20–50 steps instead of hundreds. The test proves that if ε is known exactly, DDIM recovers `x₀`.
- **Classifier-free guidance**: `ε = ε_uncond + scale·(ε_cond − ε_uncond)` steers samples toward a condition without a separate classifier.

## The toy model (`toy.py`)

A 3-layer MLP maps `(noisy point, timestep embedding) → predicted noise`, trained by hand-written backprop to denoise 2D shapes (two-moons / spiral). Sampling starts from pure Gaussian noise and runs the DDIM loop until the points snap into the target shape — the exact principle SD applies to image latents, in 2D you can plot.

## Proof it works

`tests/test_core.py`:
- `ᾱ_t` decreases monotonically in (0,1]; `q_sample` interpolates correctly.
- **DDIM exactly recovers `x₀`** when the true noise is supplied (the core identity).
- Classifier-free guidance scales the conditioning direction correctly.
- The toy model's loss decreases and its generated cloud has a data-like scale (it learned the distribution, didn't diverge).

## Limitations

- The toy model is 2D — it shows the *mechanism*, not image quality.
- For real images the app still uses Stable Diffusion (a pretrained UNet); writing/training an image-scale UNet is beyond a single Lightning session.
