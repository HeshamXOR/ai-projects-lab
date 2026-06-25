"""Shared pytest fixtures for the neural-search test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy RNG for reproducible random-vector tests."""
    return np.random.default_rng(1234)


@pytest.fixture
def toy_corpus() -> list[dict]:
    """A small, hand-built corpus with predictable lexical signals."""
    return [
        {"id": "a", "text": "the cat sat on the mat", "metadata": {"cat": "animal"}},
        {"id": "b", "text": "dogs are loyal companions and great pets", "metadata": {"cat": "animal"}},
        {"id": "c", "text": "python is a popular programming language", "metadata": {"cat": "tech"}},
        {"id": "d", "text": "machine learning models need lots of data", "metadata": {"cat": "tech"}},
        {"id": "e", "text": "the quick brown fox is very quick", "metadata": {"cat": "animal"}},
    ]
