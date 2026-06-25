"""Shared pytest fixtures for the moderation test suite."""

import pytest

from core.pipeline import ModerationPipeline, load_dataset
from core.policy import PolicyEngine
from core.tokenizer import Tokenizer


@pytest.fixture(scope="session")
def pipeline():
    """A fully-trained moderation pipeline (trained once per session)."""
    return ModerationPipeline()


@pytest.fixture(scope="session")
def dataset():
    """The bundled (texts, labels) training dataset."""
    return load_dataset()


@pytest.fixture
def tokenizer():
    """A default tokenizer with unigrams + bigrams."""
    return Tokenizer(ngram_range=(1, 2))


@pytest.fixture
def policy():
    """A policy engine with the bundled default ruleset."""
    return PolicyEngine()
