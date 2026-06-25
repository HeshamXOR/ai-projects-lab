"""Tests for the Mahalanobis anomaly model, segmentation, and features."""

import numpy as np

from core.anomaly import AnomalyConfig, MahalanobisAnomalyModel
from core.features import FeatureConfig, integral_image, local_mean_std
from core.segmentation import SegmentationConfig, otsu_threshold, segment


def test_mahalanobis_flags_ood_higher_than_id():
    """An out-of-distribution sample scores higher than an in-distribution one."""
    rng = np.random.default_rng(42)
    # "Good" cluster: correlated 3D Gaussian.
    cov = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
    mean = np.array([5.0, -2.0, 1.0])
    good = rng.multivariate_normal(mean, cov, size=400)

    model = MahalanobisAnomalyModel(AnomalyConfig(ridge=1e-3)).fit(good)

    in_dist = mean.copy()                       # right at the center
    out_dist = mean + np.array([8.0, -8.0, 6.0])  # far in correlated space

    s_in = model.score(in_dist)
    s_out = model.score(out_dist)
    assert s_out > s_in
    # The in-distribution score should be small (near the mean -> near 0).
    assert s_in < s_out
    assert float(s_in) < 5.0


def test_mahalanobis_accounts_for_correlation():
    """Equal Euclidean offsets score differently along/against the covariance."""
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])  # strongly correlated
    good = rng.multivariate_normal([0.0, 0.0], cov, size=500)
    model = MahalanobisAnomalyModel(AnomalyConfig(ridge=1e-4)).fit(good)

    along = np.array([2.0, 2.0])    # along the correlation -> plausible
    against = np.array([2.0, -2.0])  # against it -> very anomalous
    assert model.score(against) > model.score(along)


def test_mahalanobis_handles_singular_covariance():
    """A perfectly collinear feature must not blow up (ridge keeps it stable)."""
    x = np.linspace(0, 1, 50)
    good = np.stack([x, 2.0 * x], axis=1)  # rank-1, singular covariance
    model = MahalanobisAnomalyModel(AnomalyConfig(ridge=1e-2)).fit(good)
    score = model.score(np.array([0.5, 5.0]))  # off the line
    assert np.isfinite(score)
    assert score > 0


def test_integral_image_matches_bruteforce():
    rng = np.random.default_rng(7)
    img = rng.random((6, 8))
    sat = integral_image(img)
    # Sum of a sub-rectangle via the four-corner formula.
    y0, y1, x0, x1 = 1, 4, 2, 6
    s = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
    assert np.isclose(s, img[y0:y1, x0:x1].sum())


def test_local_mean_std_constant_region():
    img = np.full((20, 20), 0.5)
    mean, std = local_mean_std(img, FeatureConfig(window=5))
    assert np.allclose(mean, 0.5)
    assert np.allclose(std, 0.0, atol=1e-9)


def test_otsu_separates_bimodal():
    """Otsu threshold lands between two well-separated clusters."""
    data = np.concatenate([np.zeros(100), np.full(100, 10.0)])
    t = otsu_threshold(data, nbins=64)
    assert 0.0 < t < 10.0


def test_segment_finds_bright_defect():
    """A bright square on a flat field is segmented as foreground."""
    img = np.full((40, 40), 0.2)
    img[15:25, 15:25] = 0.9  # bright defect
    from core.features import local_contrast

    salience = local_contrast(img, FeatureConfig(window=7))
    mask = segment(salience, SegmentationConfig(method="otsu", polarity="abs"))
    # The defect interior/border should be flagged somewhere in that region.
    assert mask[14:26, 14:26].any()
    # Most of the flat background should be quiet.
    assert mask.mean() < 0.5
