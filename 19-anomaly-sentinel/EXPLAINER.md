# EXPLAINER — anomaly-sentinel: the monitoring math, from scratch

## What I implemented from scratch

- **PSI** and a **two-sample KS test** for distribution drift (`core/drift.py`).
- A streaming **EWMA/z-score** detector and a simplified **Isolation Forest** (`core/detector.py`).
- A severity-graded **rule engine** with dedup/cooldown (`core/rules.py`).

No SciPy or scikit-learn — every formula below is computed directly from NumPy.

## 1. Population Stability Index (PSI)

PSI measures how much a distribution's *mass* has redistributed relative to a
baseline. We bin the **reference** sample at deciles (10 quantile bins), then
compute, per bin `i`, the expected proportion `e_i` (from the reference) and the
actual proportion `a_i` (from the new sample):

```
PSI = Σ_i (a_i − e_i) · ln(a_i / e_i)
```

This is a symmetrised relative-entropy-style divergence: each term is positive
whether mass moved in or out of a bin, so the total only grows with discrepancy.
Empty bins would send `ln(0)` to `−∞`, so both proportion vectors are floored at
`ε = 1e-6`. Quantile edges are de-duplicated (ties collapse) and the outer edges
pushed to `±∞` so every new value lands somewhere.

**Interpretation thresholds** (industry convention): `< 0.1` stable, `0.1–0.2`
moderate shift worth watching, `> 0.2` significant drift → investigate/retrain.
The verdict flag uses `PSI > 0.2`.

## 2. Two-sample Kolmogorov–Smirnov test

KS compares two samples without assuming a distribution. We evaluate both
**empirical CDFs** on the pooled, sorted observations — `F(x) = (#obs ≤ x)/n`,
computed with `np.searchsorted(..., side="right")` — and take the largest gap:

```
D = max_x |F_a(x) − F_b(x)|
```

To get a p-value we use the **asymptotic Kolmogorov distribution**. With the
effective sample size `n_eff = n_a·n_b/(n_a+n_b)`, the scaled statistic (with the
Stephens small-sample correction) is

```
t = (√n_eff + 0.12 + 0.11/√n_eff) · D
```

and the survival function is the alternating series

```
Q(t) = P(K > t) = 2 · Σ_{k=1..∞} (−1)^(k−1) · exp(−2 k² t²)
```

truncated at 100 terms and clamped to `[0,1]`. Small p (`< 0.05`) ⇒ the samples
are unlikely to share a distribution ⇒ drift.

## 3. EWMA / streaming z-score detector

A production detector must score each point in O(1) and adapt to slow regime
changes, so we keep **exponentially weighted** moments rather than a growing
buffer. With smoothing factor `α ∈ (0,1]`, for each new `x`:

```
delta = x − mean
mean  = mean + α · delta
var   = (1 − α) · (var + α · delta²)
```

The point is standardised against the statistics learned *before* it arrived:
`z = (x − mean)/√var`. `|z|` over the threshold (after a short warmup) flags an
anomaly. Larger `α` ⇒ shorter memory / faster adaptation; smaller `α` ⇒ steadier
baseline. This is the EWMA analogue of Welford's online variance.

## 4. Simplified Isolation Forest

Isolation Forest exploits that anomalies are *few and different*: a random
axis-aligned partition isolates them in **fewer splits** than dense inliers. Each
tree recursively picks a random feature and a random split value in
`[min, max]`, stopping at a height cap `⌈log₂(sample_size)⌉` or a singleton.

The **path length** `h(x)` is the number of splits to reach `x`'s leaf, plus a
correction `c(leaf_size)` for the unbuilt subtree. The normaliser is the average
path length of an unsuccessful BST search of `n` points:

```
c(n) = 2·H_{n−1} − 2·(n−1)/n        (H = harmonic number)
```

Averaging `h(x)` over all trees gives `E[h(x)]`, and the anomaly score is

```
score(x) = 2^( −E[h(x)] / c(n) )
```

→ near **1** = isolated quickly = anomaly; near **0.5** = normal; near **0** =
deep in a dense region. The conventional decision threshold is `0.5`.

## 5. Rule engine: severity, dedup, cooldown

Detectors emit raw signals; the rule engine turns them into actionable alerts.
A `Rule` pairs a predicate over a `Signal` with a `Severity` (INFO < WARNING <
CRITICAL). The default set escalates: **drift AND anomaly** or a **hard
threshold breach** → CRITICAL; drift alone or a point anomaly → WARNING. When
several rules match, the **highest severity wins**.

**Cooldown/dedup**: each `(metric, rule)` key remembers its last fire time; a
repeat within `cooldown_seconds` is suppressed. A sustained anomaly therefore
alerts once, not on every sample. The clock is injectable, so tests drive
cooldown deterministically with a fake counter.

## Proof it works

`tests/test_core.py`: PSI ≈ 0 for same distribution and `> 0.2` under a mean
shift; KS gives `p < 0.05` for separated Gaussians and `p > 0.05` for identical
ones; the EWMA detector flags an injected spike but not pure noise; the
isolation scorer ranks an obvious outlier above inliers; the rule engine emits
CRITICAL on combined drift+anomaly and honours cooldown. `tests/test_api.py`
exercises the FastAPI surface end-to-end.
