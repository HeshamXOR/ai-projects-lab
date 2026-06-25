"""Shared pytest fixtures for the recommendation-engine test suite."""

from __future__ import annotations

import numpy as np
import pytest

from core.data import ensure_dataset, load_dataset
from core.eval import leave_one_out_split


@pytest.fixture(scope="session")
def dataset():
    """The bundled dataset (generated on first use if absent)."""
    return ensure_dataset()


@pytest.fixture(scope="session")
def loo(dataset):
    """A reproducible leave-one-out split holding out one liked item per user.

    Returns ``(train, heldout, n_users, n_items)``.
    """
    train, heldout = leave_one_out_split(
        dataset.ratings, dataset.n_users, like_threshold=4.0, seed=1
    )
    return train, heldout, dataset.n_users, dataset.n_items
