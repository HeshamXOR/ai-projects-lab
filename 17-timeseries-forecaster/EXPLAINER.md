# EXPLAINER — timeseries-forecaster: the math, from scratch

## What I implemented from scratch

- **Holt-Winters** triple exponential smoothing (`core/holtwinters.py`).
- **AR(p)** least-squares model (`core/ar.py`).
- **Ensemble** with residual-based prediction intervals (`core/ensemble.py`).
- **MAPE / sMAPE / MASE** metrics (`core/metrics.py`).
- **Rolling-origin cross-validation** (`core/backtest.py`).

The FastAPI layer is plumbing; everything below is hand-rolled NumPy.

## Holt-Winters exponential smoothing (`holtwinters.py`)

Holt-Winters tracks three slowly-varying components: a **level** `ℓ`, a **trend**
`b`, and a **seasonal** profile `s` of period `m`. Each new observation nudges
all three. The additive recursions are:

```
ℓ_t = α (y_t − s_{t−m})       + (1−α)(ℓ_{t−1} + b_{t−1})
b_t = β (ℓ_t − ℓ_{t−1})       + (1−β) b_{t−1}
s_t = γ (y_t − ℓ_t)           + (1−γ) s_{t−m}
```

Intuition: the level is a deseasonalized smoothing of `y`; the trend is a smoothed
first difference of the level; the seasonal index is a smoothed deviation of the
observation from the level. α, β, γ ∈ [0,1] control how fast each adapts.

**Multiplicative** seasonality swaps subtraction for division: `ℓ_t = α(y_t/s_{t−m}) + …`
and `s_t = γ(y_t/ℓ_t) + …`, which fits series whose seasonal swing grows with the level.

**Initialization** matters because the recursion is sensitive to its start. The
level is seeded from the mean of the first full season; the trend from the
average per-step change between the first two seasons; the seasonal indices from
the first season's deviation (additive) or ratio (multiplicative) to that mean.

**Forecasting** `h` steps ahead extends the trend linearly and reuses the seasonal
profile cyclically:

```
ŷ_{T+h} = ℓ_T + h · b_T + s_{T−m+1+((h−1) mod m)}      (additive)
ŷ_{T+h} = (ℓ_T + h · b_T) · s_{…}                       (multiplicative)
```

With `m = 1` the seasonal term drops out and this degrades gracefully to Holt's
linear-trend (double) smoothing.

## AR(p) by least squares (`ar.py`)

An order-`p` autoregression models each point as a linear function of its last
`p` values plus noise:

```
y_t = c + φ₁ y_{t−1} + … + φ_p y_{t−p} + e_t
```

To fit, we stack these equations into a design matrix `X` whose row `i` is
`[1, y_{i+p−1}, …, y_i]` (intercept then lag-1 … lag-p) with target `Y[i] = y_{i+p}`,
then minimize `‖Y − Xβ‖²`. Rather than inverting `XᵀX` directly (the textbook
normal equations `β = (XᵀX)⁻¹XᵀY`), we call `np.linalg.lstsq`, which solves the
same least-squares problem via SVD — numerically stabler and graceful under rank
deficiency.

**Forecasting** is recursive: predict `ŷ_{T+1}` from the last `p` observed values,
then slide the window forward, feeding each prediction back in as a new lag. Errors
therefore compound with horizon, which is exactly why the intervals widen.

## Prediction intervals (`ensemble.py`)

After fitting, the in-sample residuals give a noise scale `σ`. Under a
random-walk-of-errors approximation the `h`-step forecast variance grows linearly,
so the interval half-width is `z · σ · √h`, where `z` is the normal quantile for the
requested confidence (e.g. 1.96 for 95%). The ensemble blends Holt-Winters and AR
either equally or by **inverse in-sample MSE** (the better-fitting model gets more
weight), and combines their residual variances for `σ`.

## MASE and friends (`metrics.py`)

- **MAPE** = mean(|y−ŷ|/|y|) — interpretable but explodes near zero and is asymmetric.
- **sMAPE** = mean(|y−ŷ| / ((|y|+|ŷ|)/2)) — bounded and symmetric in over/under-forecast.
- **MASE** scales the forecast MAE by the in-sample MAE of a **seasonal-naive**
  forecast (`ŷ_t = y_{t−m}`):

  ```
  scale = mean_{t=m..n} |y_train_t − y_train_{t−m}|
  MASE  = mean(|y − ŷ|) / scale
  ```

  MASE < 1 means you beat the naive baseline; > 1 means you'd have been better off
  copying last season. It's scale-free and well-defined even when actuals hit zero,
  which is why M-competitions favor it.

## Rolling-origin cross-validation (`backtest.py`)

A single train/test split is one noisy estimate of skill. Rolling-origin CV
(a.k.a. time-series CV) instead picks several **origins**: train on `y[:k]`,
forecast `h` steps, score against `y[k:k+h]`, then advance `k` and repeat — always
predicting the *future*, never leaking it. Using an **expanding window** the training
set grows at each origin. We aggregate sMAPE per horizon (so you can see error grow
with `h`) and report mean MAPE/sMAPE/MASE across origins. A fresh forecaster is built
per origin so no fitted state leaks between folds.

## Proof it works

`tests/test_core.py` synthesizes known series and asserts real accuracy:
Holt-Winters forecasts a trend+seasonal series to sMAPE < 0.10; AR(2) recovers
planted coefficients φ₁=0.5, φ₂=−0.3 within 0.15; the metrics match hand-computed
values exactly; the backtest returns finite, sane errors; and ensemble intervals
satisfy `lower ≤ point ≤ upper` with non-decreasing width.
