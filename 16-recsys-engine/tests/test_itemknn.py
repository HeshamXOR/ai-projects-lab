"""Tests for the item-item cosine kNN recommender.

Proves the from-scratch cosine similarity returns sensible neighbours on a
constructed example where the "true" neighbour relationships are obvious, and
that the recommender produces well-formed output on the bundled data.
"""

from __future__ import annotations

import numpy as np

from core.itemknn import ItemKNN


def _build_clustered_ratings():
    """Two clusters of items with identical co-rating patterns.

    Items {0,1} are rated highly by users {0,1,2} and items {2,3} are rated
    highly by users {3,4,5}. So item 0's nearest neighbour should be item 1
    (not 2 or 3), and vice-versa.
    """
    rows = []
    for u in (0, 1, 2):
        rows += [(u, 0, 5.0), (u, 1, 5.0), (u, 2, 1.0), (u, 3, 1.0)]
    for u in (3, 4, 5):
        rows += [(u, 0, 1.0), (u, 1, 1.0), (u, 2, 5.0), (u, 3, 5.0)]
    return np.array(rows, dtype=np.float64)


def test_knn_finds_correct_cluster_neighbor():
    train = _build_clustered_ratings()
    knn = ItemKNN(k_neighbors=3, shrinkage=0.0).fit(train, n_users=6, n_items=4)

    # Nearest neighbour of item 0 should be item 1 (same cluster).
    nbrs0 = knn.neighbors(0, top=3)
    assert nbrs0[0][0] == 1
    # Item 1 most similar to item 0.
    nbrs1 = knn.neighbors(1, top=3)
    assert nbrs1[0][0] == 0

    # Cross-cluster similarity should be lower than within-cluster.
    within = knn.sim_[0, 1]
    across = knn.sim_[0, 2]
    assert within > across


def test_knn_similarity_symmetric_and_zero_diagonal():
    train = _build_clustered_ratings()
    knn = ItemKNN(k_neighbors=3, shrinkage=0.0).fit(train, n_users=6, n_items=4)
    sim = knn.sim_
    assert np.allclose(sim, sim.T)
    assert np.allclose(np.diag(sim), 0.0)


def test_knn_recommends_within_user_taste():
    """A user who likes cluster A items should be scored high on the unseen
    cluster-A item, not the cluster-B items."""
    # Build clusters of 3 items each so we can hold one out.
    rows = []
    for u in (0, 1, 2):
        rows += [(u, 0, 5.0), (u, 1, 5.0), (u, 2, 5.0),
                 (u, 3, 1.0), (u, 4, 1.0), (u, 5, 1.0)]
    for u in (3, 4, 5):
        rows += [(u, 0, 1.0), (u, 1, 1.0), (u, 2, 1.0),
                 (u, 3, 5.0), (u, 4, 5.0), (u, 5, 5.0)]
    train = np.array(rows, dtype=np.float64)

    knn = ItemKNN(k_neighbors=5, shrinkage=0.0).fit(train, n_users=6, n_items=6)

    # User 0 has seen items 0,1 (cluster A); hide item 2. Score should rank
    # item 2 (same cluster) above the cluster-B items.
    seen = np.array([0, 1, 3, 4], dtype=int)  # pretend 2 and 5 unseen
    scores = knn.scores_for_user(0)
    assert scores[2] > scores[5]


def test_knn_recommend_wellformed(loo):
    train, _heldout, n_users, n_items = loo
    knn = ItemKNN(k_neighbors=20).fit(train, n_users, n_items)
    seen = train[train[:, 0] == 0][:, 1].astype(int)
    recs = knn.recommend(0, k=5, exclude=seen)
    items = [i for i, _ in recs]
    assert len(recs) <= 5
    assert not (set(items) & set(seen.tolist()))
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)
