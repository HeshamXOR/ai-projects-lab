"""Retriever + persistence integration tests."""

from __future__ import annotations

import os

from core.retriever import HybridRetriever
from core import store


def test_retriever_indexes_and_searches(toy_corpus):
    """Hybrid search should return live docs and respect deletion."""
    r = HybridRetriever()
    r.add_batch(toy_corpus)
    assert r.num_docs == len(toy_corpus)

    hits, total = r.search("python programming", k=5)
    assert hits, "no hits"
    assert total >= 1
    # doc c is the obvious lexical + semantic match.
    assert any(h.id == "c" for h in hits)

    # Delete it; it must vanish from results.
    assert r.delete("c") is True
    hits2, _ = r.search("python programming", k=5)
    assert all(h.id != "c" for h in hits2)
    assert r.num_docs == len(toy_corpus) - 1


def test_metadata_filter_and_pagination(toy_corpus):
    """Filtering restricts by metadata; offset/limit paginate."""
    r = HybridRetriever()
    r.add_batch(toy_corpus)

    # Filter to animals only.
    hits, total = r.search("the quick fox cat dog", k=10, filter_spec={"cat": "animal"})
    assert hits
    assert all(h.metadata.get("cat") == "animal" for h in hits)

    # Pagination: page size 1.
    page1, total1 = r.search("the quick fox cat dog", k=10, filter_spec={"cat": "animal"}, offset=0, limit=1)
    page2, total2 = r.search("the quick fox cat dog", k=10, filter_spec={"cat": "animal"}, offset=1, limit=1)
    assert total1 == total2
    assert len(page1) == 1 and len(page2) == 1
    assert page1[0].id != page2[0].id


def test_replace_document(toy_corpus):
    """Re-adding an id replaces its text and it stays searchable."""
    r = HybridRetriever()
    r.add_batch(toy_corpus)
    r.add("c", "rust is a systems programming language", {"cat": "tech"})
    assert r.num_docs == len(toy_corpus)
    doc = r.get("c")
    assert doc is not None and "rust" in doc.text

    hits, _ = r.search("rust systems", k=5)
    assert any(h.id == "c" for h in hits)


def test_persistence_roundtrip(tmp_path, toy_corpus):
    """Save then load reproduces docs and search behavior."""
    r = HybridRetriever()
    r.add_batch(toy_corpus)
    before, _ = r.search("python programming", k=5)

    directory = os.path.join(str(tmp_path), "idx")
    store.save(r, directory)
    assert os.path.isfile(os.path.join(directory, "manifest.json"))
    assert os.path.isfile(os.path.join(directory, "vectors.npy"))

    loaded = store.load(directory)
    assert loaded.num_docs == r.num_docs
    after, _ = loaded.search("python programming", k=5)
    assert [h.id for h in after] == [h.id for h in before]
