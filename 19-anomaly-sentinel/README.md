# 🛡️ anomaly-sentinel — model/data monitoring with from-scratch drift & anomaly detection

## What I implemented from scratch

- **Population Stability Index (PSI)** — quantile-binned reference vs. actual, `sum((actual-expected)*ln(actual/expected))` with empty-bin epsilon flooring — `core/drift.py`
- **Two-sample Kolmogorov–Smirnov test** — empirical CDFs over the pooled sample, `D = max|F1-F2|`, asymptotic Kolmogorov-distribution p-value (with the Stephens correction) — `core/drift.py`
- **EWMA / streaming z-score detector** — O(1)-per-sample exponentially weighted running mean & variance, flags points whose `|z|` exceeds a threshold — `core/detector.py`
- **Simplified Isolation Forest scorer** — random axis-split trees built from scratch, anomaly score from mean path length normalised by `c(n)` via `2^(-E[h]/c(n))` — `core/detector.py`
- **Alert rule engine** — severity-graded (INFO/WARNING/CRITICAL) rules over combined signals, with dedup/cooldown — `core/rules.py`

The core is **pure NumPy** — no SciPy, no scikit-learn. FastAPI only wraps it in a service.

## Why it's here

Shipping a model is the easy part; knowing when it has quietly gone wrong is the hard part. This service watches the two things that break models in production — the **input distribution drifting** away from training, and **anomalous values** appearing in a metric stream — and turns those raw signals into **severity-graded, de-duplicated alerts**. The detection math is the interesting part, so it's all hand-written rather than imported.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Stream some values, then ask for alerts:

```bash
# stream a batch of metric values
curl -s -X POST localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"metric":"latency_ms","values":[100,101,99,102,100,500]}'

# register a reference distribution, then test a shifted sample for drift
curl -s -X POST localhost:8000/reference \
  -H 'content-type: application/json' \
  -d '{"metric":"score","values":[0.1,0.2,0.15,0.22,0.18,0.2,0.17,0.19]}'

curl -s -X POST localhost:8000/drift/check \
  -H 'content-type: application/json' \
  -d '{"metric":"score","sample":[0.8,0.9,0.85,0.92,0.88,0.9,0.87,0.91]}'

# list alerts (filterable)
curl -s "localhost:8000/alerts?metric=latency_ms&severity=CRITICAL"
```

## API

| Method & path      | Body / params                                              | Returns |
|--------------------|------------------------------------------------------------|---------|
| `GET /`            | —                                                          | service banner + tracked metrics |
| `GET /health`      | —                                                          | `{status, metrics, alerts}` |
| `POST /ingest`     | `{metric, values: float\|[float], timestamp?}`             | `{ingested, buffer_size, new_alerts}` |
| `GET /alerts`      | query `metric?`, `severity?` (INFO/WARNING/CRITICAL)       | `{count, alerts[]}` |
| `POST /reference`  | `{metric, values:[float] (>=2)}`                           | `{metric, reference_size}` |
| `POST /drift/check`| `{metric, sample:[float] (>=2), n_bins?}`                  | `{psi:{...}, ks:{...}, drift}` |

**Ingest** updates the metric's EWMA detector, scores the point with the isolation forest over the recent window, builds a `Signal`, and runs the rule engine; emitted `Alert`s are stored. **drift/check** runs PSI + KS of `sample` against the metric's registered reference and ORs the two verdicts. Validation is via Pydantic (`422` on bad bodies); missing references give `404`, unknown severities `400`.

## Verify

```bash
pytest -q   # PSI ~0 vs shifted, KS separates distributions, EWMA flags a spike
            # not noise, isolation scores an outlier high, rules + cooldown
```

## Limitations

- Detectors are univariate per metric; multivariate drift (joint distribution) isn't modelled — the isolation forest accepts 2-D input but the service feeds it one feature.
- The KS p-value is the asymptotic approximation (good for `n` in the hundreds+); for tiny samples an exact permutation test would be tighter.
- State is in-memory: alerts and detector state reset on restart. A real deployment would persist to a store and shard detectors per metric.
- PSI binning uses the reference's quantiles; very low-cardinality metrics collapse to fewer bins (handled, but coarser).
