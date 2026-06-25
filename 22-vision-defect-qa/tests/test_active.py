"""Tests proving the active-learning sampler picks diverse + uncertain points.

We construct feature sets where the correct answer is known by design and
assert concrete selected indices.
"""

import numpy as np

from core.active import (
    ActiveConfig,
    boundary_uncertainty,
    entropy_uncertainty,
    select_samples,
)


def test_avoids_near_duplicates_under_diversity():
    """With two near-identical points, the sampler must not pick both.

    Layout: points 0 and 1 are near-duplicates at the origin; points 2 and 3
    are far away and far from each other. With equal uncertainty the diversity
    term should force the picks to spread out -> never {0,1} together.
    """
    feats = np.array(
        [
            [0.0, 0.0],
            [0.01, 0.0],   # near-duplicate of 0
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )
    unc = np.ones(4)  # equal uncertainty -> pure diversity drives the choice
    cfg = ActiveConfig(alpha=0.5)

    selected = select_samples(feats, unc, n_select=3, config=cfg)
    assert len(selected) == 3
    # Must include the two far-apart anchors and NOT both near-duplicates.
    assert 2 in selected and 3 in selected
    assert not (0 in selected and 1 in selected)


def test_prefers_high_uncertainty_seed():
    """The first (seed) pick is the single most-uncertain candidate."""
    feats = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    unc = np.array([0.1, 0.9, 0.2, 0.3])
    cfg = ActiveConfig(alpha=1.0)  # pure uncertainty
    selected = select_samples(feats, unc, n_select=1, config=cfg)
    assert selected == [1]


def test_pure_uncertainty_orders_by_score():
    """alpha=1 ignores geometry and returns the top-k uncertain points."""
    feats = np.zeros((5, 2))  # geometry irrelevant under alpha=1
    unc = np.array([0.2, 0.95, 0.1, 0.8, 0.5])
    cfg = ActiveConfig(alpha=1.0)
    selected = select_samples(feats, unc, n_select=3, config=cfg)
    # Top three uncertainties are indices 1 (0.95), 3 (0.8), 4 (0.5).
    assert set(selected) == {1, 3, 4}
    assert selected[0] == 1  # most uncertain is the seed


def test_combined_uncertainty_and_diversity():
    """High alpha still spreads picks when uncertainties tie.

    Three clustered uncertain points plus one isolated certain point. With a
    balanced alpha the sampler should grab one from the cluster (uncertain)
    then jump to the isolated point (diverse) rather than staying in-cluster.
    """
    feats = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [20.0, 20.0],  # isolated
        ],
        dtype=np.float64,
    )
    unc = np.array([0.9, 0.9, 0.9, 0.1])
    cfg = ActiveConfig(alpha=0.5)
    selected = select_samples(feats, unc, n_select=2, config=cfg)
    # Seed is one of the uncertain cluster points; second pick is the far one.
    assert selected[0] in (0, 1, 2)
    assert selected[1] == 3


def test_boundary_uncertainty_peaks_at_threshold():
    scores = np.array([0.0, 5.0, 10.0])
    unc = boundary_uncertainty(scores, threshold=5.0)
    # The point exactly on the threshold is maximally uncertain.
    assert unc[1] == 1.0
    assert unc[1] > unc[0] and unc[1] > unc[2]


def test_entropy_uncertainty_bounds():
    p = np.array([0.5, 0.99, 0.01])
    unc = entropy_uncertainty(p)
    assert np.isclose(unc[0], 1.0)        # 50/50 -> max entropy
    assert unc[1] < 0.2 and unc[2] < 0.2  # confident -> low entropy


def test_n_select_capped_at_pool_size():
    feats = np.array([[0.0], [1.0]])
    unc = np.array([0.5, 0.5])
    selected = select_samples(feats, unc, n_select=10)
    assert len(selected) == 2
