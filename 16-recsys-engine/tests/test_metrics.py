"""Tests for the ranking metrics on tiny, hand-checkable examples.

Every assertion here can be verified by hand from the definitions in
core/eval.py and EXPLAINER.md.
"""

from __future__ import annotations

import numpy as np

from core.eval import (
    dcg_at_k,
    leave_one_out_split,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_basic():
    # Ranked list, relevant = {1, 4}. Top-3 = [1, 2, 3] -> 1 hit of 3.
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 4}
    assert precision_at_k(ranked, relevant, 3) == 1.0 / 3.0
    # Top-5 contains both relevant -> 2/5.
    assert precision_at_k(ranked, relevant, 5) == 2.0 / 5.0


def test_recall_at_k_basic():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 4}
    # Top-3 captures 1 of 2 relevant -> 0.5.
    assert recall_at_k(ranked, relevant, 3) == 0.5
    # Top-5 captures both -> 1.0.
    assert recall_at_k(ranked, relevant, 5) == 1.0


def test_recall_with_no_relevant_is_zero():
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_dcg_known_value():
    # Single relevant item at position 1: rel/log2(2) = 1/1 = 1.0.
    assert dcg_at_k([7], {7}, 5) == 1.0
    # Relevant item at position 2: 1/log2(3).
    val = dcg_at_k([9, 7], {7}, 5)
    assert abs(val - 1.0 / np.log2(3.0)) < 1e-12


def test_ndcg_perfect_ranking_is_one():
    # Two relevant items ranked first -> NDCG = 1.0.
    ranked = [1, 2, 3, 4]
    relevant = {1, 2}
    assert abs(ndcg_at_k(ranked, relevant, 4) - 1.0) < 1e-12


def test_ndcg_single_relevant_at_position_two():
    # One relevant item at rank 2. DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    ranked = [10, 5]
    relevant = {5}
    expected = (1.0 / np.log2(3.0)) / 1.0
    assert abs(ndcg_at_k(ranked, relevant, 2) - expected) < 1e-12


def test_ndcg_is_between_zero_and_one():
    ranked = [4, 1, 2, 3]
    relevant = {1, 3}
    val = ndcg_at_k(ranked, relevant, 4)
    assert 0.0 < val < 1.0


def test_ndcg_rewards_higher_placement():
    relevant = {3}
    high = ndcg_at_k([3, 1, 2], relevant, 3)   # relevant at rank 1
    low = ndcg_at_k([1, 2, 3], relevant, 3)    # relevant at rank 3
    assert high > low
    assert abs(high - 1.0) < 1e-12


def test_leave_one_out_holds_out_one_per_user():
    # 3 users, several ratings each; like_threshold filters to >=4.
    ratings = np.array(
        [
            [0, 0, 5.0], [0, 1, 4.0], [0, 2, 2.0],
            [1, 0, 4.5], [1, 3, 3.0],
            [2, 1, 4.0], [2, 2, 4.0], [2, 3, 5.0],
        ],
        dtype=np.float64,
    )
    train, heldout = leave_one_out_split(ratings, n_users=3, like_threshold=4.0, seed=0)
    # Each user with >=2 ratings and a liked item should appear once.
    assert set(heldout.keys()) == {0, 1, 2}
    # Exactly one row removed per held-out user.
    assert train.shape[0] == ratings.shape[0] - len(heldout)
    # Held-out items are genuinely liked (>=4.0).
    for u, item in heldout.items():
        row = ratings[(ratings[:, 0] == u) & (ratings[:, 1] == item)]
        assert row[0, 2] >= 4.0
