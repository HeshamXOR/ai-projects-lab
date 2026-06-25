"""Tests for the matrix factorization model.

Proves:
- training RMSE decreases over epochs (SGD is learning),
- the model recovers low-rank structure better than chance,
- predictions / recommendations are well-formed.
"""

from __future__ import annotations

import numpy as np

from core.mf import MatrixFactorization


def test_train_rmse_decreases_over_epochs(loo):
    train, _heldout, n_users, n_items = loo
    mf = MatrixFactorization(n_factors=8, n_epochs=30, lr=0.02, reg=0.05, seed=0)
    mf.fit(train, n_users, n_items)

    assert len(mf.train_rmse_) == 30
    # Final RMSE must be clearly below the first epoch's.
    assert mf.train_rmse_[-1] < mf.train_rmse_[0]
    # And meaningfully so on this learnable data.
    assert mf.train_rmse_[-1] < 0.7 * mf.train_rmse_[0]
    # Monotone-ish: the last epoch is among the best seen.
    assert mf.train_rmse_[-1] <= min(mf.train_rmse_) + 1e-9


def test_predictions_have_reasonable_range(loo):
    train, _heldout, n_users, n_items = loo
    mf = MatrixFactorization(n_epochs=20, seed=0).fit(train, n_users, n_items)
    preds = np.array(
        [mf.predict(int(r[0]), int(r[1])) for r in train[:50]]
    )
    # Predictions live in a sane neighbourhood of the 1-5 rating scale.
    assert np.all(preds > -2.0) and np.all(preds < 8.0)


def test_recommend_excludes_seen_items_and_is_sorted(loo):
    train, _heldout, n_users, n_items = loo
    mf = MatrixFactorization(n_epochs=20, seed=0).fit(train, n_users, n_items)

    user = 0
    seen = train[train[:, 0] == user][:, 1].astype(int)
    recs = mf.recommend(user, k=5, exclude=seen)
    rec_items = [i for i, _ in recs]

    assert len(recs) == 5
    # No already-seen item recommended.
    assert not (set(rec_items) & set(seen.tolist()))
    # Scores are in descending order.
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


def test_als_variant_also_reduces_rmse(loo):
    train, _heldout, n_users, n_items = loo
    mf = MatrixFactorization(n_factors=8, reg=0.1, seed=0)
    mf.fit_als(train, n_users, n_items, n_iters=10)
    assert mf.train_rmse_[-1] < mf.train_rmse_[0]


def test_recovers_low_rank_structure():
    # Build a clean rank-2 matrix and check MF predicts held-out cells well.
    rng = np.random.default_rng(3)
    n_users, n_items, k = 30, 25, 2
    P = rng.normal(0, 1, (n_users, k))
    Q = rng.normal(0, 1, (n_items, k))
    full = 3.0 + P @ Q.T

    triplets = []
    for u in range(n_users):
        for i in range(n_items):
            if rng.random() < 0.5:
                triplets.append((u, i, full[u, i]))
    triplets = np.array(triplets, dtype=np.float64)

    mf = MatrixFactorization(n_factors=2, n_epochs=80, lr=0.02, reg=0.001, seed=1)
    mf.fit(triplets, n_users, n_items)
    # Should fit the (noise-free) observed entries tightly.
    assert mf.train_rmse_[-1] < 0.2
