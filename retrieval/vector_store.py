"""
Vector store — loads the FAISS index and catalog metadata at startup.
Exposes a search() function used by the retriever.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_faiss_index: Optional[faiss.Index] = None
_catalog_metadata: dict[str, dict] = {}


def load(index_path: str, metadata_path: str) -> None:
    """Load the FAISS index and metadata from disk. Called once at startup."""
    global _faiss_index, _catalog_metadata

    idx_p = Path(index_path)
    meta_p = Path(metadata_path)

    if not idx_p.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Run 'python scripts/build_index.py' to create it."
        )
    if not meta_p.exists():
        raise FileNotFoundError(
            f"Catalog metadata not found at '{metadata_path}'. "
            "Run 'python scripts/build_index.py' to create it."
        )

    _faiss_index = faiss.read_index(str(idx_p))
    with open(meta_p, encoding="utf-8") as f:
        _catalog_metadata = json.load(f)

    logger.info(
        "Vector store loaded: %d vectors (dim=%d), %d metadata entries.",
        _faiss_index.ntotal,
        _faiss_index.d,
        len(_catalog_metadata),
    )


def search(query_vector: np.ndarray, k: int = 15) -> list[dict]:
    """
    Run a top-K similarity search against the FAISS index.
    Returns a list of catalog entries (dicts).
    """
    if _faiss_index is None:
        raise RuntimeError("Vector store is not loaded. Call load() first.")

    query = query_vector.reshape(1, -1).astype(np.float32)
    actual_k = min(k, _faiss_index.ntotal)
    _, indices = _faiss_index.search(query, actual_k)

    results: list[dict] = []
    for idx in indices[0]:
        if idx == -1:
            continue
        entry = _catalog_metadata.get(str(idx))
        if entry:
            results.append(entry)
    return results


def get_all_urls() -> set[str]:
    """Return the set of all catalog URLs — used for URL validation."""
    return {entry["url"] for entry in _catalog_metadata.values()}
