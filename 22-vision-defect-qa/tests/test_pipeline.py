"""End-to-end pipeline + RLE round-trip tests."""

import numpy as np

from core.pipeline import InspectionService, rle_decode, rle_encode


def test_rle_roundtrip():
    rng = np.random.default_rng(3)
    mask = rng.random((12, 15)) > 0.7
    enc = rle_encode(mask)
    back = rle_decode(mask.shape, enc["runs"])
    assert np.array_equal(mask, back)


def test_inspect_returns_blobs_and_score():
    svc = InspectionService()

    # Fit on flat "good" images so a defect stands out.
    rng = np.random.default_rng(5)
    good = [0.5 + 0.01 * rng.standard_normal((30, 30)) for _ in range(8)]
    svc.fit_good(good)
    assert svc.is_fitted

    # A test image with one bright defect blob.
    img = 0.5 + 0.01 * rng.standard_normal((30, 30))
    img[10:15, 10:15] = 0.95
    result = svc.inspect(img)

    d = result.as_dict()
    assert d["shape"] == [30, 30]
    assert d["mask"]["encoding"] == "rle_row_major"
    assert d["n_components"] >= 1
    assert isinstance(d["anomaly_score"], float)
    assert all("area" in b and "bbox" in b for b in d["blobs"])


def test_suggest_labels_from_pool():
    svc = InspectionService()
    # Build a pool directly in feature space (skip extraction for determinism).
    feats = np.array(
        [[0.0, 0.0], [0.02, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]],
        dtype=np.float64,
    )
    svc.set_pool(feats)
    assert svc.pool_size() == 5
    picks = svc.suggest_labels(3)
    assert len(picks) == 3
    # Should not pick both near-duplicates (0 and 1).
    assert not (0 in picks and 1 in picks)
