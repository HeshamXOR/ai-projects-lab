# EXPLAINER — Image Q&A: attention you can see

## What I implemented from scratch

- **Attention rollout** — composes a transformer's per-layer attention into a single influence map (`core/attention_rollout.py`).
- **Grad-CAM** — gradient-weighted activation heatmaps showing where a vision model looked (`core/gradcam.py`).

BLIP still answers the questions; these add **explainability** — turning the model from a black box into something whose focus you can visualize.

## Why a single attention layer isn't enough

A transformer mixes information across every layer via residual connections, so one layer's attention matrix doesn't tell you what an output token really depends on. **Attention rollout** fixes this:

1. Add the identity to each layer's attention (`A' = ½A + ½I`) to account for the residual path that carries a token's own information forward.
2. Re-normalize rows so they remain probability distributions.
3. Multiply the layers: `rollout = A'_L · … · A'_1`.

The rollout's row for the `[CLS]`/answer position is a distribution over input patches — a saliency map. The test confirms a key property: composing stochastic matrices keeps rows summing to 1, and all-identity attention rolls up to the identity.

## How Grad-CAM works

Grad-CAM asks: *which feature-map channels, if amplified, would most increase the target score?*

```
weights_c = average_pool( ∂score/∂A_c )      # how much channel c matters
CAM       = ReLU( Σ_c  weights_c · A_c )      # weighted sum of activations
```

The ReLU keeps only features that *support* the prediction. The result is a coarse heatmap, upscaled (nearest-neighbor, also from scratch) and overlaid on the image. Tests verify it highlights the high-gradient channel's hot spot and zeroes out negative contributions.

## Proof it works

`tests/test_core.py` validates both algorithms on hand-constructed inputs with known answers (rollout row-sums and identity case; Grad-CAM channel weighting and ReLU clipping; nearest-neighbor upscaling).

## Limitations

- The algorithms are model-agnostic and unit-tested here; wiring them to BLIP's exact internal tensors requires forward/backward hooks on the specific model build (the app handles this, with a clear message if the attentions aren't exposed by the installed version).
- Grad-CAM is a coarse, low-resolution explanation by nature.
