"""Shared pytest fixtures."""

from __future__ import annotations

import pathlib

import pytest

SAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "samples"


def _read(name: str) -> str:
    return (SAMPLES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def sample_us() -> str:
    return _read("invoice_us.txt")


@pytest.fixture(scope="session")
def sample_eu() -> str:
    return _read("invoice_eu.txt")


@pytest.fixture(scope="session")
def sample_uk() -> str:
    return _read("invoice_uk.txt")


@pytest.fixture(scope="session")
def sample_receipt() -> str:
    return _read("receipt_grocery.txt")
