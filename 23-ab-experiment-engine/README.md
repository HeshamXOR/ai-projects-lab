## What I implemented from scratch

This is a production-grade A/B experiment platform whose **entire statistics
engine is hand-written in pure Python/NumPy** -- no SciPy, no statsmodels. The
hard parts -- the probability distributions that every test depends on -- are
implemented numerically from first principles:

- **Standard normal CDF** via the error function (`erf`), plus a hand-rolled
  Abramowitz & Stegun `erf` series so the math is genuinely from scratch.
- **Inverse normal CDF (quantile / `ppf`)** via Acklam's rational approximation
  refined with a Halley step to full double precision.
- **Student-t CDF** via the regularized incomplete beta function evaluated with
  a Lentz continued fraction -- the exact textbook route, accurate even for tiny
  degrees of freedom.

On top of those distributions:

- **Two-proportion z-test** -- pooled standard error for the statistic, unpooled
  SE for the confidence interval, two-sided p-value, absolute & relative lift.
- **Welch's t-test** -- unequal-variance t statistic with the
  Welch-Satterthwaite degrees-of-freedom correction, p-value from the
  from-scratch t-CDF, and a CI built from a bisection-inverted t critical value.
- **Power / sample-size** -- required n per arm from baseline rate, minimum
  detectable effect, alpha and power; plus the inverse (achieved power for a
  given n), all driven by the inverse normal CDF.
- **Sequential testing** -- Wald's SPRT (log-likelihood-ratio with A/B
  boundaries) **and** group-sequential alpha-spending (Pocock and
  O'Brien-Fleming boundaries via the Lan-DeMets representation).
- **CUPED variance reduction** -- pre-experiment covariate adjustment
  `Y - theta*(X - E[X])` with `theta = Cov(Y,X)/Var(X)`, reporting the achieved
  variance reduction (`rho^2`).

See `EXPLAINER.md` for the full derivations.

---

## Overview

The service tracks experiments in memory, ingests per-user observations, and on
demand computes the full decision picture for each treatment-vs-control
comparison: lift, p-value, confidence interval, significance flag, statistical
power, sequential-monitoring status, and a single recommended decision
(**ship / no-ship / keep-running**) with a plain-English rationale.

```
core/
  distributions.py   erf, normal cdf/pdf/ppf, Student-t cdf/sf, incomplete beta
  tests_stats.py     two-proportion z-test, Welch's t-test, confidence intervals
  power.py           sample-size & power calculations
  sequential.py      SPRT + alpha-spending group-sequential boundaries
  cuped.py           CUPED variance reduction
app.py               FastAPI service + in-memory store + decision engine
tests/               pytest suite proving the stats against textbook values
```

## Run it

```bash
# (one time) install deps
pip install -r requirements.txt

# run the API
uvicorn app:app --reload --port 8000
# open http://localhost:8000/docs for interactive Swagger UI

# run the test suite (proves the math against textbook constants)
pytest -q

# or with Docker
docker build -t ab-experiment-engine .
docker run -p 8000:8000 ab-experiment-engine
```

The statistics modules have **zero runtime dependency on the web framework** --
you can `from core import two_proportion_z_test` and use the engine directly in a
notebook or batch job.

## API

### `POST /experiment` -- create an experiment

Binary (conversion) metric:

```json
{
  "name": "checkout-button-color",
  "metric_type": "binary",
  "variants": [{"name": "control"}, {"name": "blue"}],
  "control": "control",
  "alpha": 0.05,
  "power_target": 0.8,
  "baseline_rate": 0.10,
  "mde_absolute": 0.02,
  "spending": "obrien_fleming"
}
```

Response includes the planned sample size per arm:

```json
{ "id": "a1b2c3d4e5f6", "planned_n_per_arm": 3841, ... }
```

Continuous metric (uses Welch's t-test and, when covariates are supplied, CUPED):

```json
{
  "name": "revenue-per-user",
  "metric_type": "continuous",
  "variants": [{"name": "control"}, {"name": "treatment"}],
  "control": "control"
}
```

### `POST /event` -- record an observation

Binary:

```json
{ "experiment_id": "a1b2c3d4e5f6", "variant": "blue", "user_id": "u_123", "converted": true }
```

Continuous (with optional pre-experiment covariate for CUPED):

```json
{ "experiment_id": "...", "variant": "treatment", "value": 42.5, "covariate": 38.0 }
```

### `GET /results/{id}` -- analyze

Returns, per treatment variant:

```json
{
  "experiment_id": "...",
  "control": {"name": "control", "conversions": 100, "n": 1000, "rate": 0.10},
  "comparisons": [{
    "variant": "blue",
    "test": {"statistic": 2.10, "p_value": 0.0355, "ci": [0.002, 0.058], "significant": true},
    "absolute_lift": 0.03,
    "relative_lift": 0.30,
    "power": {"achieved_power": 0.86, "planned_n_per_arm": 3841, "current_n_per_arm": 1000},
    "sequential": {"group_sequential": {...}, "sprt": {...}},
    "recommendation": "ship",
    "rationale": "Statistically significant positive effect."
  }]
}
```

### `GET /experiments` -- list  ·  `GET /healthz` -- liveness

## Decision logic

The recommendation folds three signals together:

1. **Sequential monitor** (if a planned sample size / MDE is configured): if the
   alpha-spending boundary is crossed with a positive effect, recommend **ship**
   immediately; if it stops for futility, **no-ship**.
2. Otherwise the **fixed-horizon test**: a significant positive effect -> ship; a
   significant negative effect -> no-ship.
3. If not yet significant but **underpowered** -> **keep-running**; not
   significant at adequate power -> no-ship (a genuine null result).

For continuous metrics, when both arms carry pre-experiment covariates the engine
runs **CUPED** and bases the decision on the variance-reduced test.
