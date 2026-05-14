"""
Indexer — embeds catalog entries and builds the FAISS vector index.
Outputs:
  data/faiss.index            — FAISS binary index
  data/catalog_metadata.json  — int index → full catalog entry mapping
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _build_text(entry: dict) -> str:
    """Concatenate name + description for embedding."""
    return f"{entry['name']}. {entry.get('description', '')}"


def build_index(
    catalog: list[dict],
    embed_model: str = "all-MiniLM-L6-v2",
    index_path: str = "data/faiss.index",
    metadata_path: str = "data/catalog_metadata.json",
) -> None:
    """
    Embed all catalog entries and persist the FAISS index and metadata.
    """
    if not catalog:
        raise ValueError("Cannot build index from an empty catalog.")

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model: %s", embed_model)
    model = SentenceTransformer(embed_model)

    texts = [_build_text(e) for e in catalog]
    logger.info("Encoding %d catalog entries…", len(texts))
    embeddings: np.ndarray = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to %s (%d vectors, dim=%d).", index_path, index.ntotal, dim)

    # Metadata: map integer index → full catalog entry
    metadata = {str(i): entry for i, entry in enumerate(catalog)}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Catalog metadata saved to %s.", metadata_path)
