"""FastAPI service for the A/B experiment engine.

Exposes a small REST API backed by an in-memory experiment store:

* ``POST /experiment``      -- create an experiment (variants + metric config).
* ``POST /event``           -- record an observation for a (experiment, variant, user).
* ``GET  /results/{id}``    -- compute lift, p-value, CI, significance, power,
                               sequential status, and a recommended decision.
* ``GET  /experiments``     -- list experiments.
* ``GET  /healthz``         -- liveness probe.

The statistics are delegated entirely to :mod:`core`. This file is the
orchestration / decision layer: it chooses the right test for the metric type,
runs the power and sequential analyses, and folds everything into a single
recommendation (ship / no-ship / keep-running).

Run locally::

    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from core import (
    SequentialDecision,
    SpendingFunction,
    apply_cuped,
    group_sequential_decision,
    power_for_sample_size,
    sample_size_two_proportion,
    sprt_bernoulli,
    two_proportion_z_test,
    welch_t_test,
)

app = FastAPI(
    title="A/B Experiment Engine",
    version="1.0.0",
    description=(
        "Production-grade A/B testing service with a from-scratch statistics "
        "engine: two-proportion z-test, Welch's t-test, power/sample-size, "
        "sequential testing (SPRT + alpha-spending), and CUPED variance reduction."
    ),
)


# ---------------------------------------------------------------------------
# Domain model (in-memory store)
# ---------------------------------------------------------------------------
class MetricType(str, Enum):
    """Supported metric families."""

    BINARY = "binary"          # conversion-style; uses two-proportion z-test
    CONTINUOUS = "continuous"  # revenue/duration-style; uses Welch's t-test


@dataclass
class VariantData:
    """Accumulated observations for one variant."""

    name: str
    # Binary metric: count of conversions out of n.
    conversions: int = 0
    n: int = 0
    # Continuous metric: streaming sums for mean/variance (Welford-friendly).
    values: list[float] = field(default_factory=list)
    covariates: list[float] = field(default_factory=list)
    seen_users: set[str] = field(default_factory=set)

    def record_binary(self, converted: bool, user_id: str | None) -> None:
        if user_id is not None:
            self.seen_users.add(user_id)
        self.n += 1
        if converted:
            self.conversions += 1

    def record_continuous(
        self, value: float, covariate: float | None, user_id: str | None
    ) -> None:
        if user_id is not None:
            self.seen_users.add(user_id)
        self.n += 1
        self.values.append(value)
        self.covariates.append(covariate if covariate is not None else math.nan)


@dataclass
class Experiment:
    """An experiment definition plus its accumulated data."""

    id: str
    name: str
    metric_type: MetricType
    control: str
    alpha: float
    power_target: float
    baseline_rate: float | None
    mde_absolute: float | None
    spending: SpendingFunction
    variants: dict[str, VariantData]

    def control_variant(self) -> VariantData:
        return self.variants[self.control]

    def treatment_variants(self) -> list[VariantData]:
        return [v for name, v in self.variants.items() if name != self.control]


class _Store:
    """Thread-safe in-memory experiment store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._experiments: dict[str, Experiment] = {}

    def create(self, exp: Experiment) -> None:
        with self._lock:
            self._experiments[exp.id] = exp

    def get(self, exp_id: str) -> Experiment:
        with self._lock:
            exp = self._experiments.get(exp_id)
        if exp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"experiment '{exp_id}' not found",
            )
        return exp

    def all(self) -> list[Experiment]:
        with self._lock:
            return list(self._experiments.values())


STORE = _Store()


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------
class VariantConfig(BaseModel):
    """A single variant in a new experiment."""

    name: str = Field(..., min_length=1, max_length=64, examples=["control"])


class ExperimentCreate(BaseModel):
    """Payload for ``POST /experiment``."""

    name: str = Field(..., min_length=1, max_length=128)
    metric_type: MetricType
    variants: list[VariantConfig] = Field(..., min_length=2, max_length=16)
    control: str = Field(..., description="Name of the control variant")
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    power_target: float = Field(0.80, gt=0.0, lt=1.0)
    baseline_rate: float | None = Field(
        None, gt=0.0, lt=1.0, description="Required for binary metrics' power calc"
    )
    mde_absolute: float | None = Field(
        None, description="Absolute minimum detectable effect for power/sequential"
    )
    spending: SpendingFunction = SpendingFunction.OBRIEN_FLEMING

    @field_validator("variants")
    @classmethod
    def _unique_variant_names(cls, v: list[VariantConfig]) -> list[VariantConfig]:
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        return v

    @model_validator(mode="after")
    def _control_in_variants(self) -> "ExperimentCreate":
        names = {c.name for c in self.variants}
        if self.control not in names:
            raise ValueError("control must be one of the variant names")
        if self.metric_type is MetricType.BINARY and self.baseline_rate is None:
            raise ValueError("baseline_rate is required for binary metrics")
        return self


class ExperimentCreated(BaseModel):
    """Response for ``POST /experiment``."""

    id: str
    name: str
    metric_type: MetricType
    variants: list[str]
    control: str
    planned_n_per_arm: int | None = None


class EventIn(BaseModel):
    """Payload for ``POST /event``."""

    experiment_id: str
    variant: str
    user_id: str | None = None
    # Binary metric:
    converted: bool | None = None
    # Continuous metric:
    value: float | None = None
    covariate: float | None = Field(
        None, description="Pre-experiment covariate for CUPED (continuous metrics)"
    )

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> "EventIn":
        if self.converted is None and self.value is None:
            raise ValueError("provide 'converted' (binary) or 'value' (continuous)")
        if self.converted is not None and self.value is not None:
            raise ValueError("provide only one of 'converted' or 'value'")
        return self


class EventAccepted(BaseModel):
    experiment_id: str
    variant: str
    total_observations: int


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------
class Recommendation(str, Enum):
    SHIP = "ship"
    NO_SHIP = "no_ship"
    KEEP_RUNNING = "keep_running"


def _decide(
    significant: bool,
    estimate: float,
    seq_decision: SequentialDecision,
    achieved_power: float | None,
    power_target: float,
) -> tuple[Recommendation, str]:
    """Fold the statistical signals into a single recommendation + rationale.

    Logic
    -----
    * If the sequential monitor says stop-and-reject-H0 AND the effect is
      positive -> SHIP (sequential control already guards the error rate).
    * If sequential says stop-for-futility -> NO_SHIP.
    * Otherwise fall back to the fixed-horizon test: a significant *positive*
      effect with adequate power -> SHIP; significant *negative* -> NO_SHIP;
      not significant but underpowered -> KEEP_RUNNING; not significant and
      adequately powered -> NO_SHIP (genuine null).
    """
    if seq_decision is SequentialDecision.STOP_REJECT_H0:
        if estimate > 0:
            return Recommendation.SHIP, "Sequential boundary crossed with a positive effect."
        return Recommendation.NO_SHIP, "Sequential boundary crossed but the effect is negative."
    if seq_decision is SequentialDecision.STOP_ACCEPT_H0:
        return Recommendation.NO_SHIP, "Sequential test stopped for futility (no effect)."

    # Sequential says continue -> consult the fixed-horizon view.
    if significant:
        if estimate > 0:
            return Recommendation.SHIP, "Statistically significant positive effect."
        return Recommendation.NO_SHIP, "Statistically significant negative effect."

    if achieved_power is not None and achieved_power < power_target:
        return (
            Recommendation.KEEP_RUNNING,
            f"Not significant yet and underpowered "
            f"({achieved_power:.0%} < target {power_target:.0%}); collect more data.",
        )
    return (
        Recommendation.NO_SHIP,
        "Not significant at adequate power; treat as no detectable effect.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post(
    "/experiment",
    response_model=ExperimentCreated,
    status_code=status.HTTP_201_CREATED,
    tags=["experiments"],
)
def create_experiment(payload: ExperimentCreate) -> ExperimentCreated:
    """Create a new experiment and (for binary metrics) plan its sample size."""
    exp_id = uuid.uuid4().hex[:12]
    variants = {c.name: VariantData(name=c.name) for c in payload.variants}

    exp = Experiment(
        id=exp_id,
        name=payload.name,
        metric_type=payload.metric_type,
        control=payload.control,
        alpha=payload.alpha,
        power_target=payload.power_target,
        baseline_rate=payload.baseline_rate,
        mde_absolute=payload.mde_absolute,
        spending=payload.spending,
        variants=variants,
    )
    STORE.create(exp)

    planned_n: int | None = None
    if (
        payload.metric_type is MetricType.BINARY
        and payload.baseline_rate is not None
        and payload.mde_absolute is not None
    ):
        try:
            planned_n = sample_size_two_proportion(
                payload.baseline_rate,
                payload.mde_absolute,
                alpha=payload.alpha,
                power=payload.power_target,
            ).required_n_per_arm
        except ValueError:
            planned_n = None

    return ExperimentCreated(
        id=exp_id,
        name=exp.name,
        metric_type=exp.metric_type,
        variants=list(variants.keys()),
        control=exp.control,
        planned_n_per_arm=planned_n,
    )


@app.post(
    "/event",
    response_model=EventAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["events"],
)
def record_event(event: EventIn) -> EventAccepted:
    """Record a single observation against an experiment variant."""
    exp = STORE.get(event.experiment_id)
    variant = exp.variants.get(event.variant)
    if variant is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"variant '{event.variant}' is not part of experiment {exp.id}",
        )

    if exp.metric_type is MetricType.BINARY:
        if event.converted is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="binary metric expects 'converted' (bool)",
            )
        variant.record_binary(event.converted, event.user_id)
    else:
        if event.value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="continuous metric expects 'value' (number)",
            )
        variant.record_continuous(event.value, event.covariate, event.user_id)

    return EventAccepted(
        experiment_id=exp.id,
        variant=variant.name,
        total_observations=variant.n,
    )


def _binary_results(exp: Experiment) -> dict:
    """Compute the full result block for a binary-metric experiment."""
    control = exp.control_variant()
    blocks = []

    for treatment in exp.treatment_variants():
        if control.n < 1 or treatment.n < 1:
            blocks.append(
                {
                    "variant": treatment.name,
                    "status": "insufficient_data",
                    "message": "need observations in both control and this variant",
                }
            )
            continue

        test = two_proportion_z_test(
            control.conversions,
            control.n,
            treatment.conversions,
            treatment.n,
            alpha=exp.alpha,
        )

        # Power achieved given current per-arm size (uses configured MDE).
        achieved_power: float | None = None
        planned_n: int | None = None
        if exp.baseline_rate is not None and exp.mde_absolute is not None:
            n_per_arm = min(control.n, treatment.n)
            try:
                achieved_power = power_for_sample_size(
                    exp.baseline_rate, exp.mde_absolute, n_per_arm, alpha=exp.alpha
                )
                planned_n = sample_size_two_proportion(
                    exp.baseline_rate,
                    exp.mde_absolute,
                    alpha=exp.alpha,
                    power=exp.power_target,
                ).required_n_per_arm
            except ValueError:
                achieved_power = None

        # Sequential monitoring: alpha-spending boundary on the observed z, plus
        # an SPRT view when an MDE is configured.
        seq_block: dict = {}
        seq_decision = SequentialDecision.CONTINUE
        if planned_n:
            gs = group_sequential_decision(
                test.statistic,
                min(control.n, treatment.n),
                planned_n,
                alpha=exp.alpha,
                spending=exp.spending,
            )
            seq_block["group_sequential"] = gs.as_dict()
            seq_decision = gs.decision

        if exp.baseline_rate is not None and exp.mde_absolute is not None:
            p1 = exp.baseline_rate + exp.mde_absolute
            if 0.0 < p1 < 1.0:
                sprt = sprt_bernoulli(
                    control.conversions,
                    control.n,
                    treatment.conversions,
                    treatment.n,
                    p0=exp.baseline_rate,
                    p1=p1,
                    alpha=exp.alpha,
                    beta=1.0 - exp.power_target,
                )
                seq_block["sprt"] = sprt.as_dict()

        rec, rationale = _decide(
            test.significant,
            test.estimate,
            seq_decision,
            achieved_power,
            exp.power_target,
        )

        blocks.append(
            {
                "variant": treatment.name,
                "status": "ok",
                "test": test.as_dict(),
                "absolute_lift": test.estimate,
                "relative_lift": test.relative_lift,
                "power": {
                    "achieved_power": achieved_power,
                    "planned_n_per_arm": planned_n,
                    "current_n_per_arm": min(control.n, treatment.n),
                    "target_power": exp.power_target,
                },
                "sequential": seq_block,
                "recommendation": rec.value,
                "rationale": rationale,
            }
        )

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "metric_type": exp.metric_type.value,
        "control": {
            "name": control.name,
            "conversions": control.conversions,
            "n": control.n,
            "rate": (control.conversions / control.n) if control.n else None,
        },
        "comparisons": blocks,
    }


def _continuous_results(exp: Experiment) -> dict:
    """Compute the full result block for a continuous-metric experiment."""
    control = exp.control_variant()
    blocks = []

    for treatment in exp.treatment_variants():
        if control.n < 2 or treatment.n < 2:
            blocks.append(
                {
                    "variant": treatment.name,
                    "status": "insufficient_data",
                    "message": "need >=2 observations in both control and this variant",
                }
            )
            continue

        # Raw Welch test.
        test = welch_t_test(control.values, treatment.values, alpha=exp.alpha)

        # CUPED-adjusted view when both arms carry valid covariates.
        cuped_block: dict | None = None
        adjusted_test_block: dict | None = None
        if _has_covariates(control) and _has_covariates(treatment):
            cuped = apply_cuped(
                control.values,
                control.covariates,
                treatment.values,
                treatment.covariates,
            )
            cuped_block = cuped.as_dict()
            adj_test = welch_t_test(
                mean_a=cuped.adjusted_mean_a,
                var_a=cuped.adjusted_var_a,
                n_a=cuped.n_a,
                mean_b=cuped.adjusted_mean_b,
                var_b=cuped.adjusted_var_b,
                n_b=cuped.n_b,
                alpha=exp.alpha,
            )
            adjusted_test_block = adj_test.as_dict()
            # Prefer the variance-reduced test for the decision when available.
            test = adj_test

        rec, rationale = _decide(
            test.significant,
            test.estimate,
            SequentialDecision.CONTINUE,
            None,
            exp.power_target,
        )

        blocks.append(
            {
                "variant": treatment.name,
                "status": "ok",
                "test": test.as_dict(),
                "absolute_lift": test.estimate,
                "relative_lift": test.relative_lift,
                "cuped": cuped_block,
                "cuped_adjusted_test": adjusted_test_block,
                "recommendation": rec.value,
                "rationale": rationale,
            }
        )

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "metric_type": exp.metric_type.value,
        "control": {
            "name": control.name,
            "n": control.n,
            "mean": (float(np.mean(control.values)) if control.values else None),
        },
        "comparisons": blocks,
    }


def _has_covariates(v: VariantData) -> bool:
    """True when the variant has at least 2 finite covariate values."""
    finite = [c for c in v.covariates if not math.isnan(c)]
    return len(finite) == len(v.covariates) and len(finite) >= 2


@app.get("/results/{experiment_id}", tags=["results"])
def get_results(experiment_id: str) -> dict:
    """Compute lift, p-value, CI, significance, power, sequential status, decision."""
    exp = STORE.get(experiment_id)
    if exp.metric_type is MetricType.BINARY:
        return _binary_results(exp)
    return _continuous_results(exp)


@app.get("/experiments", tags=["experiments"])
def list_experiments() -> dict:
    """List all experiments with their current observation counts."""
    return {
        "experiments": [
            {
                "id": e.id,
                "name": e.name,
                "metric_type": e.metric_type.value,
                "variants": {
                    name: {"n": v.n, "conversions": v.conversions}
                    for name, v in e.variants.items()
                },
            }
            for e in STORE.all()
        ]
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
