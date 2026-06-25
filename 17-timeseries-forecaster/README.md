# 📈 timeseries-forecaster — multi-horizon forecasting service

A small forecasting service that takes a numeric series and returns a
multi-step-ahead forecast with prediction intervals. The statistical core —
Holt-Winters smoothing, an AR(p) model, their ensemble, the error metrics and a
rolling-origin backtester — is implemented **from scratch in NumPy**. Library
models are optional; the math here is mine.

## What I implemented from scratch

- **Holt-Winters exponential smoothing** — level/trend/seasonal recursions, seasonal initialization, additive *and* multiplicative seasonality, multi-step forecast — `core/holtwinters.py`
- **AR(p) model** — lagged design matrix, least-squares fit via the normal equations (`np.linalg.lstsq`), recursive multi-step forecast — `core/ar.py`
- **Ensemble** — blends Holt-Winters + AR (equal or inverse-error weighting) and builds prediction intervals from residual σ that widen with horizon — `core/ensemble.py`
- **Metrics** — MAPE, sMAPE and MASE (with seasonal-naive scaling) — `core/metrics.py`
- **Rolling-origin cross-validation** — expanding-window backtest that refits and scores over multiple origins, returning per-horizon and aggregate errors — `core/backtest.py`

The FastAPI layer just validates input and calls into this core. See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## Why it's here

Forecasting libraries hide the recursions behind a `.fit()`. This project shows
the actual exponential-smoothing state updates, the least-squares AR fit, how
prediction intervals widen with the horizon, and how you honestly score a
forecaster with rolling-origin backtesting instead of a single train/test split.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload          # http://localhost:8000

# forecast 6 steps of a seasonal series with the ensemble
curl -s -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"series":[50,55,60,58,52,57,62,60,54,59,64,62,56,61,66,64,58,63,68,66,60,65,70,68],
       "horizon":6, "season_length":12, "model":"ensemble"}'
```

## API

### `POST /forecast`

Request:

| field           | type                                   | default      | notes                                    |
|-----------------|----------------------------------------|--------------|------------------------------------------|
| `series`        | `float[]`                              | —            | observations, oldest first; finite       |
| `horizon`       | `int > 0`                              | —            | steps to forecast                        |
| `season_length` | `int >= 1`                             | `1`          | seasonal period (1 = none); needs `len(series) >= 2*season_length` |
| `model`         | `"holtwinters" \| "ar" \| "ensemble"` | `"ensemble"` | which forecaster                         |
| `confidence`    | `0 < float < 1`                        | `0.95`       | prediction-interval level                |
| `ar_order`      | `int >= 1`                             | `2`          | AR order p (AR / ensemble)               |

Response:

```json
{
  "point":  [61.2, 66.1, 70.9, 68.4, 62.7, 67.5],
  "lower":  [58.1, 61.7, 65.2, 61.4, 54.6, 58.2],
  "upper":  [64.3, 70.5, 76.6, 75.4, 70.8, 76.8],
  "model":  "ensemble",
  "metrics": {"mape": 0.03, "smape": 0.03, "mase": 0.41},
  "weights": {"holtwinters": 0.62, "ar": 0.38}
}
```

`metrics` is a best-effort rolling-origin backtest summary (omitted when the
series is too short). `weights` is present only for the ensemble. Invalid input
(e.g. `horizon <= 0`, series shorter than `2*season_length`) returns `422`.

### `GET /health`

Liveness probe → `{"status": "ok"}`.

### `GET /`

Service metadata and endpoint list.

## Verify

```bash
pytest -q   # Holt-Winters accuracy, AR coefficient recovery, exact metric values,
            # backtest sanity, interval bracketing, API happy-path + validation
```

## Limitations

- Smoothing parameters (α, β, γ) are user-supplied, not optimized via MLE — a grid/gradient search would tighten fits (drop-in).
- Prediction intervals use a normal approximation (`z * σ * √h`); they don't capture skew or fat tails.
- AR assumes (weak) stationarity — differencing/ARIMA would handle trended, non-stationary data more rigorously.
- The ensemble is a static two-model blend; stacking or time-varying weights would adapt better to regime changes.
