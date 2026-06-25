"""BM25 correctness tests on a toy corpus."""

from __future__ import annotations

from core.bm25 import BM25Index


def test_bm25_ranks_right_doc_first(toy_corpus):
    """A query should surface the document that actually contains the term."""
    idx = BM25Index()
    for doc in toy_corpus:
        idx.add(doc["id"], doc["text"])

    # "python programming" most strongly matches doc c.
    results = idx.search("python programming language", k=5)
    assert results, "no BM25 results"
    assert results[0][0] == "c"

    # "cat mat" most strongly matches doc a.
    results = idx.search("cat mat", k=5)
    assert results[0][0] == "a"


def test_bm25_idf_rewards_rare_terms():
    """A rare term should outweigh a term present in every document."""
    idx = BM25Index()
    idx.add("d1", "common common common rare")
    idx.add("d2", "common common common")
    idx.add("d3", "common common common")

    # "rare" appears only in d1; "common" everywhere. d1 must win.
    results = idx.search("rare common", k=3)
    assert results[0][0] == "d1"


def test_bm25_length_normalization():
    """A short doc and a long doc with same TF: shorter should score higher."""
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add("short", "apple")
    idx.add("long", "apple " + "filler " * 50)
    idx.add("other", "banana orange")

    results = dict(idx.search("apple", k=3))
    assert results["short"] > results["long"]


def test_bm25_remove_and_empty():
    """Removing a doc drops it from results; empty index returns nothing."""
    idx = BM25Index()
    idx.add("x", "hello world")
    idx.add("y", "hello there")
    assert idx.num_docs == 2

    assert idx.remove("x") is True
    assert idx.remove("x") is False
    ids = [d for d, _ in idx.search("hello", k=5)]
    assert "x" not in ids and "y" in ids

    idx.remove("y")
    assert idx.search("hello", k=5) == []
