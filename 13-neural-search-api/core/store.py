"""On-disk persistence for the hybrid retriever.

A saved index is a directory containing:

* ``vectors.npy``  -- the dense HNSW vector matrix (``float32``, shape n x dim).
* ``ann.json``     -- the HNSW graph topology and parameters.
* ``bm25.json``    -- the BM25 postings / term-frequency state.
* ``docs.json``    -- documents, metadata, and id<->index mappings.
* ``manifest.json``-- format version and basic stats.

Vectors are stored separately as a NumPy ``.npy`` file (compact, fast to
load); everything else is small enough for JSON, which keeps saved indexes
human-inspectable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from .ann import HNSWIndex
from .bm25 import BM25Index
from .embed import HashingEmbedder
from .retriever import HybridRetriever

FORMAT_VERSION = 1


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write ``obj`` to ``path`` as UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def _read_json(path: str) -> Dict[str, Any]:
    """Read and parse a UTF-8 JSON file at ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save(retriever: HybridRetriever, directory: str) -> None:
    """Persist a :class:`HybridRetriever` to ``directory``.

    The directory is created if needed. Existing index files inside it are
    overwritten.

    Args:
        retriever: The retriever to save.
        directory: Target directory path.
    """
    os.makedirs(directory, exist_ok=True)

    # Dense vectors: assemble into a single contiguous matrix.
    vectors = retriever.ann._vectors  # internal: list of float32 arrays
    if vectors:
        mat = np.vstack(vectors).astype(np.float32)
    else:
        mat = np.zeros((0, retriever.ann.dim), dtype=np.float32)
    np.save(os.path.join(directory, "vectors.npy"), mat)

    _write_json(os.path.join(directory, "ann.json"), retriever.ann.to_state())
    _write_json(os.path.join(directory, "bm25.json"), retriever.bm25.to_state())
    _write_json(os.path.join(directory, "docs.json"), retriever.docs_state())
    _write_json(
        os.path.join(directory, "manifest.json"),
        {
            "format_version": FORMAT_VERSION,
            "num_vectors": int(mat.shape[0]),
            "dim": retriever.ann.dim,
            "num_docs": retriever.num_docs,
        },
    )


def load(directory: str) -> HybridRetriever:
    """Load a :class:`HybridRetriever` previously written by :func:`save`.

    Args:
        directory: Directory produced by :func:`save`.

    Returns:
        A fully reconstructed retriever with its ANN graph, BM25 index, and
        documents restored.

    Raises:
        FileNotFoundError: If the directory or required files are missing.
        ValueError: If the on-disk format version is unsupported.
    """
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"no manifest.json in {directory!r}")
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported index format version {manifest.get('format_version')}"
        )

    vectors = np.load(os.path.join(directory, "vectors.npy"))
    ann_state = _read_json(os.path.join(directory, "ann.json"))
    bm25_state = _read_json(os.path.join(directory, "bm25.json"))
    docs_state = _read_json(os.path.join(directory, "docs.json"))

    dim = int(ann_state["dim"])
    embedder = HashingEmbedder(dim=dim)
    retriever = HybridRetriever(embedder=embedder)

    retriever.ann = HNSWIndex.from_state(ann_state, vectors)
    retriever.bm25 = BM25Index.from_state(bm25_state)
    retriever.load_docs_state(docs_state)
    return retriever
