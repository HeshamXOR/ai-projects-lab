"""FastAPI service for model/data monitoring — anomaly-sentinel.

Streams per-metric observations through from-scratch detectors and a rule
engine, stores the resulting alerts, and answers drift queries against a
registered reference distribution.

Endpoints
---------
* ``GET  /``             -- service banner / metadata.
* ``GET  /health``       -- liveness + state summary.
* ``POST /ingest``       -- push value(s) for a metric; runs detectors + rules.
* ``GET  /alerts``       -- list stored alerts (filter by metric / severity).
* ``POST /reference``    -- register a reference distribution for a metric.
* ``POST /drift/check``  -- PSI + KS of a sample vs. the registered reference.

All state is in-memory (dicts of detectors / references / alerts); the heavy
lifting lives in the pure-NumPy :mod:`core` package.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from core.detector import EWMADetector, IsolationForestScorer
from core.drift import drift_report
from core.rules import Alert, RuleEngine, Severity, Signal

app = FastAPI(
    title="anomaly-sentinel",
    version="1.0.0",
    description="Model/data monitoring: drift detection, online anomaly "
    "detection and severity-graded alerting. Pure-NumPy core.",
)


# --------------------------------------------------------------------------- #
# In-memory state
# --------------------------------------------------------------------------- #
class _MetricState:
    """Per-metric monitoring state: a streaming detector + a value buffer."""

    def __init__(self, window: int = 512) -> None:
        self.detector = EWMADetector()
        self.window = window
        self.buffer: List[float] = []
        self.reference: Optional[List[float]] = None

    def push(self, value: float) -> None:
        self.buffer.append(float(value))
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]


_metrics: Dict[str, _MetricState] = {}
_alerts: List[Alert] = []
_engine = RuleEngine()


def _state(metric: str) -> _MetricState:
    if metric not in _metrics:
        _metrics[metric] = _MetricState()
    return _metrics[metric]


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    """Body for ``POST /ingest``."""

    metric: str = Field(..., min_length=1, description="Metric name.")
    values: Union[float, List[float]] = Field(
        ..., description="A single value or a list of values to stream in order."
    )
    timestamp: Optional[float] = Field(
        None, description="Optional epoch seconds for the (last) sample."
    )

    @field_validator("values")
    @classmethod
    def _non_empty(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("values list must not be empty")
        return v


class ReferenceRequest(BaseModel):
    """Body for ``POST /reference``."""

    metric: str = Field(..., min_length=1)
    values: List[float] = Field(..., min_length=2, description="Reference sample.")


class DriftCheckRequest(BaseModel):
    """Body for ``POST /drift/check``."""

    metric: str = Field(..., min_length=1)
    sample: List[float] = Field(..., min_length=2, description="Sample to test.")
    n_bins: int = Field(10, ge=2, le=100, description="PSI quantile-bin count.")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict:
    """Service banner."""
    return {
        "service": "anomaly-sentinel",
        "version": "1.0.0",
        "endpoints": ["/ingest", "/alerts", "/reference", "/drift/check", "/health"],
        "metrics_tracked": list(_metrics.keys()),
    }


@app.get("/health")
def health() -> dict:
    """Liveness probe with a small state summary."""
    return {
        "status": "ok",
        "metrics": len(_metrics),
        "alerts": len(_alerts),
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    """Stream value(s) into a metric's detector and run the rule engine.

    For each incoming value we update the EWMA/z-score detector, compute an
    isolation-forest anomaly score over the recent window (once enough samples
    exist), assemble a :class:`Signal`, and evaluate the rule engine. Any
    resulting alerts are stored and returned.
    """
    st = _state(req.metric)
    values = req.values if isinstance(req.values, list) else [req.values]
    ts = req.timestamp if req.timestamp is not None else time.time()

    new_alerts: List[dict] = []
    for v in values:
        st.push(v)
        point = st.detector.update(v)

        anomaly_score = 0.0
        if len(st.buffer) >= 16:
            scorer = IsolationForestScorer(
                n_trees=40, sample_size=min(128, len(st.buffer)), random_state=0
            ).fit(np.asarray(st.buffer))
            anomaly_score = float(scorer.score(np.array([[v]]))[0])

        signal = Signal(
            metric=req.metric,
            value=float(v),
            z=point.z,
            anomaly_score=anomaly_score,
            timestamp=ts,
        )
        alert = _engine.evaluate(signal)
        if alert is not None:
            _alerts.append(alert)
            new_alerts.append(alert.as_dict())

    return {
        "metric": req.metric,
        "ingested": len(values),
        "buffer_size": len(st.buffer),
        "new_alerts": new_alerts,
    }


@app.get("/alerts")
def get_alerts(
    metric: Optional[str] = Query(None, description="Filter by metric name."),
    severity: Optional[str] = Query(
        None, description="Filter by severity (INFO/WARNING/CRITICAL)."
    ),
) -> dict:
    """Return stored alerts, optionally filtered by metric and/or severity."""
    sev_level: Optional[int] = None
    if severity is not None:
        try:
            sev_level = int(Severity[severity.upper()])
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"unknown severity '{severity}' (use INFO/WARNING/CRITICAL)",
            )

    out = []
    for a in _alerts:
        if metric is not None and a.metric != metric:
            continue
        if sev_level is not None and int(a.severity) != sev_level:
            continue
        out.append(a.as_dict())
    return {"count": len(out), "alerts": out}


@app.post("/reference")
def set_reference(req: ReferenceRequest) -> dict:
    """Register a reference distribution used by ``/drift/check``."""
    st = _state(req.metric)
    st.reference = [float(x) for x in req.values]
    return {"metric": req.metric, "reference_size": len(st.reference)}


@app.post("/drift/check")
def drift_check(req: DriftCheckRequest) -> dict:
    """Compute PSI + KS of ``sample`` against the metric's registered reference."""
    st = _metrics.get(req.metric)
    if st is None or st.reference is None:
        raise HTTPException(
            status_code=404,
            detail=f"no reference distribution registered for metric '{req.metric}'",
        )
    try:
        report = drift_report(st.reference, req.sample, n_bins=req.n_bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"metric": req.metric, **report}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
