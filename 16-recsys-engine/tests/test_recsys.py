"""Tests proving MF beats the popularity baseline, plus baseline sanity.

The headline test is ``test_mf_beats_popularity``: on a held-out leave-one-out
split of the bundled (genuinely low-rank) dataset, the from-scratch MF model
must achieve a higher NDCG@5 than the popularity baseline.
"""

from __future__ import annotations

import numpy as np

from core.eval import evaluate_model
from core.mf import MatrixFactorization
from core.popularity import PopularityRecommender


def test_popularity_fits_and_scores(loo):
    train, _heldout, _n_users, n_items = loo
    pop = PopularityRecommender(shrinkage=5.0).fit(train, n_items)
    scores = pop.scores_for_user()
    assert scores.shape == (n_items,)
    # Shrinkage pulls scores toward the global mean -> bounded range.
    assert np.all(scores >= 1.0) and np.all(scores <= 5.0)


def test_popularity_shrinkage_pulls_low_count_items():
    # Item 0: one rating of 5.0. Item 1: many ratings averaging ~3.0.
    rows = [(u, 1, 3.0) for u in range(20)]
    rows.append((0, 0, 5.0))
    train = np.array(rows, dtype=np.float64)
    pop = PopularityRecommender(shrinkage=5.0).fit(train, n_items=2)
    # Despite a raw mean of 5.0, the single-rating item should be shrunk
    # below the well-supported 3.0 item is NOT required, but it must be
    # pulled well below its raw mean of 5.0 toward the global mean.
    assert pop.means_[0] == 5.0
    assert pop.scores_[0] < 5.0
    assert pop.scores_[0] < pop.means_[0]


def test_mf_beats_popularity(loo):
    """The core claim: MF > popularity on held-out NDCG@5."""
    train, heldout, n_users, n_items = loo

    mf = MatrixFactorization(
        n_factors=8, n_epochs=40, lr=0.02, reg=0.05, seed=0
    ).fit(train, n_users, n_items)
    pop = PopularityRecommender(shrinkage=5.0).fit(train, n_items)

    mf_eval = evaluate_model(mf, train, heldout, k=5)
    pop_eval = evaluate_model(pop, train, heldout, k=5)

    # Sanity: we actually evaluated a chunk of users.
    assert mf_eval.n_users_evaluated >= 20
    assert mf_eval.n_users_evaluated == pop_eval.n_users_evaluated

    # The headline assertion.
    assert mf_eval.ndcg > pop_eval.ndcg
    # Recall@5 should also be at least as good.
    assert mf_eval.recall >= pop_eval.recall


def test_mf_beats_popularity_at_k10(loo):
    train, heldout, n_users, n_items = loo
    mf = MatrixFactorization(
        n_factors=8, n_epochs=40, lr=0.02, reg=0.05, seed=0
    ).fit(train, n_users, n_items)
    pop = PopularityRecommender(shrinkage=5.0).fit(train, n_items)

    mf_eval = evaluate_model(mf, train, heldout, k=10)
    pop_eval = evaluate_model(pop, train, heldout, k=10)
    assert mf_eval.ndcg >= pop_eval.ndcg
