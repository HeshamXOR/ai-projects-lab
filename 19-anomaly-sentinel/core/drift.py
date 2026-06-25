"""Distribution-drift detection, implemented from scratch in NumPy.

Two complementary, model-free drift tests:

1. **Population Stability Index (PSI)** — the workhorse of model monitoring.
   Bin a *reference* distribution into quantile bins, then measure how much an
   *actual* sample's mass redistributed across those same bins. PSI is a
   symmetrised relative-entropy-style score::

       PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

   Conventional reading: ``< 0.1`` stable, ``0.1-0.2`` moderate shift,
   ``> 0.2`` significant drift (retrain / investigate).

2. **Two-sample Kolmogorov-Smirnov (KS) test** — distribution-free comparison
   of two samples via their empirical CDFs. The statistic is the largest
   vertical gap ``D = max|F1(x) - F2(x)|``; the asymptotic Kolmogorov
   distribution converts ``D`` into a p-value. Small p (``< 0.05``) ⇒ the two
   samples are unlikely to come from the same distribution ⇒ drift.

No SciPy: the quantile binning, empirical CDFs and the Kolmogorov series are
all computed directly from NumPy primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# --------------------------------------------------------------------------- #
# Defaults / thresholds
# --------------------------------------------------------------------------- #
PSI_DRIFT_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05
_EPS = 1e-6  # floor for empty bins so ln() stays finite


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class PSIResult:
    """Outcome of a PSI computation."""

    psi: float
    per_bin: List[float]
    bin_edges: List[float]
    drift: bool

    def as_dict(self) -> dict:
        return {
            "psi": self.psi,
            "per_bin": self.per_bin,
            "bin_edges": self.bin_edges,
            "drift": self.drift,
        }


@dataclass
class KSResult:
    """Outcome of a two-sample KS test."""

    statistic: float
    p_value: float
    drift: bool

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "drift": self.drift,
        }


# --------------------------------------------------------------------------- #
# Population Stability Index
# --------------------------------------------------------------------------- #
def _quantile_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges at evenly spaced quantiles of ``reference``.

    Duplicate edges (from ties / low-cardinality data) are collapsed so we never
    create empty, zero-width bins. The outer edges are pushed to +/- inf so that
    every actual-sample value falls into some bin.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, qs)
    edges = np.unique(edges)  # drop ties; may yield fewer than n_bins bins
    if edges.size < 2:
        # degenerate (constant reference) -> single catch-all bin
        edges = np.array([reference.min(), reference.max() + _EPS])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of ``values`` falling in each bin defined by ``edges``."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(counts.shape, _EPS)
    return counts / total


def population_stability_index(
    reference: np.ndarray | List[float],
    actual: np.ndarray | List[float],
    n_bins: int = 10,
) -> PSIResult:
    """Compute the Population Stability Index between two samples.

    Parameters
    ----------
    reference:
        The baseline sample whose quantiles define the bins.
    actual:
        The current sample whose mass is compared against the reference.
    n_bins:
        Number of quantile bins (default 10 — deciles).

    Returns
    -------
    PSIResult with the scalar PSI, the per-bin contributions, the bin edges, and
    a ``drift`` verdict (``psi > 0.2``).
    """
    ref = np.asarray(reference, dtype=float).ravel()
    act = np.asarray(actual, dtype=float).ravel()
    if ref.size == 0 or act.size == 0:
        raise ValueError("reference and actual must be non-empty")

    edges = _quantile_edges(ref, n_bins)
    expected = _bin_proportions(ref, edges)
    observed = _bin_proportions(act, edges)

    # Floor both distributions so neither ratio nor log blows up on empty bins.
    expected = np.clip(expected, _EPS, None)
    observed = np.clip(observed, _EPS, None)

    per_bin = (observed - expected) * np.log(observed / expected)
    psi = float(per_bin.sum())
    return PSIResult(
        psi=psi,
        per_bin=[float(x) for x in per_bin],
        bin_edges=[float(e) for e in edges],
        drift=psi > PSI_DRIFT_THRESHOLD,
    )


# --------------------------------------------------------------------------- #
# Two-sample Kolmogorov-Smirnov test
# --------------------------------------------------------------------------- #
def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic ``D = max|F_a - F_b|`` via merged-order CDFs.

    We evaluate both empirical CDFs on the pooled, sorted set of observations.
    At each pooled point the empirical CDF is ``(#obs <= x) / n``; the statistic
    is the maximum absolute difference between the two step functions.
    """
    a = np.sort(a)
    b = np.sort(b)
    pooled = np.concatenate([a, b])
    pooled.sort()
    # CDF values via searchsorted (right side counts obs <= x).
    cdf_a = np.searchsorted(a, pooled, side="right") / a.size
    cdf_b = np.searchsorted(b, pooled, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _kolmogorov_sf(t: float, terms: int = 100) -> float:
    """Survival function of the Kolmogorov distribution, ``Q(t) = P(K > t)``.

    Uses the standard alternating series::

        Q(t) = 2 * sum_{k=1..inf} (-1)^(k-1) * exp(-2 k^2 t^2)

    which is the asymptotic null distribution of ``sqrt(n_eff) * D``. Clamped to
    ``[0, 1]``; ``t <= 0`` ⇒ p-value 1.
    """
    if t <= 0:
        return 1.0
    k = np.arange(1, terms + 1)
    series = np.sum(((-1.0) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * (t ** 2)))
    p = 2.0 * series
    return float(min(1.0, max(0.0, p)))


def ks_two_sample(
    sample_a: np.ndarray | List[float],
    sample_b: np.ndarray | List[float],
) -> KSResult:
    """Two-sample KS test with an asymptotic Kolmogorov p-value.

    Parameters
    ----------
    sample_a, sample_b:
        The two samples to compare.

    Returns
    -------
    KSResult with the KS statistic ``D``, the approximate p-value, and a
    ``drift`` verdict (``p_value < 0.05``).
    """
    a = np.asarray(sample_a, dtype=float).ravel()
    b = np.asarray(sample_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")

    d = _ks_statistic(a, b)
    # Effective sample size for the asymptotic scaling factor.
    n_eff = (a.size * b.size) / (a.size + b.size)
    t = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * d  # Stephens correction
    p_value = _kolmogorov_sf(t)
    return KSResult(
        statistic=d,
        p_value=p_value,
        drift=p_value < KS_PVALUE_THRESHOLD,
    )


def drift_report(
    reference: np.ndarray | List[float],
    actual: np.ndarray | List[float],
    n_bins: int = 10,
) -> dict:
    """Run both PSI and KS and combine into one verdict dict.

    ``drift`` is the logical OR of the two tests' individual verdicts — either a
    large PSI or a significant KS result is enough to raise the flag.
    """
    psi_res = population_stability_index(reference, actual, n_bins=n_bins)
    ks_res = ks_two_sample(reference, actual)
    return {
        "psi": psi_res.as_dict(),
        "ks": ks_res.as_dict(),
        "drift": bool(psi_res.drift or ks_res.drift),
    }
