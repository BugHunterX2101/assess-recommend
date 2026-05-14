"""
Embedder — thin wrapper around SentenceTransformer.
Provides a singleton encode() function used by the retriever.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model: %s", model_name)
    return SentenceTransformer(model_name)


def encode(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode a list of strings into a float32 embedding matrix."""
    model = _get_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype(np.float32)
