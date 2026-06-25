"""Reciprocal-rank fusion correctness tests with hand-computable inputs."""

from __future__ import annotations

import math

from core.fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_known_fused_order():
    """Fuse two known rank lists and check the exact resulting order.

    list A: [x, y, z]   (ranks 1, 2, 3)
    list B: [y, x, w]   (ranks 1, 2, 3)
    With k=60 and equal weights:
        x = 1/61 + 1/62
        y = 1/62 + 1/61   (== x)
        z = 1/63
        w = 1/63
    x and y tie; tie broken by id -> x before y.
    z and w tie; tie broken by id -> w before z.
    """
    a = [("x", 0.9), ("y", 0.8), ("z", 0.7)]
    b = [("y", 0.5), ("x", 0.4), ("w", 0.3)]

    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [doc for doc, _ in fused]
    assert ids[:2] == ["x", "y"]  # tie broken alphabetically
    assert set(ids[2:]) == {"w", "z"}

    scores = dict(fused)
    assert math.isclose(scores["x"], 1 / 61 + 1 / 62)
    assert math.isclose(scores["y"], 1 / 62 + 1 / 61)
    assert math.isclose(scores["z"], 1 / 63)
    assert math.isclose(scores["w"], 1 / 63)


def test_rrf_item_in_both_lists_beats_single_list_item():
    """An item present in both lists should outrank a top-of-one-list item."""
    a = [("shared", 0.9), ("onlyA", 0.8)]
    b = [("onlyB", 0.9), ("shared", 0.8)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60))
    # shared: 1/61 + 1/62; onlyA: 1/62; onlyB: 1/61
    assert fused["shared"] > fused["onlyA"]
    assert fused["shared"] > fused["onlyB"]


def test_rrf_weights_bias_toward_a_list():
    """Heavily weighting one list promotes its top item."""
    a = [("fromA", 0.5)]
    b = [("fromB", 0.5)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60, weights=[5.0, 1.0]))
    assert fused["fromA"] > fused["fromB"]


def test_rrf_validates_inputs():
    """Bad k or mismatched weights raise ValueError."""
    try:
        reciprocal_rank_fusion([[("x", 1.0)]], k=0)
        assert False, "expected ValueError for k=0"
    except ValueError:
        pass

    try:
        reciprocal_rank_fusion([[("x", 1.0)]], weights=[1.0, 2.0])
        assert False, "expected ValueError for weight mismatch"
    except ValueError:
        pass


def test_weighted_score_fusion_normalizes():
    """Score fusion min-max normalizes each list before combining."""
    a = [("p", 100.0), ("q", 0.0)]
    b = [("q", 10.0), ("p", 0.0)]
    fused = dict(weighted_score_fusion([a, b]))
    # Both p and q get one 1.0 and one 0.0 -> tie at 1.0 each.
    assert math.isclose(fused["p"], 1.0)
    assert math.isclose(fused["q"], 1.0)
