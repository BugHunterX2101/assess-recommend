"""
Retriever — builds a retrieval query from conversation history,
embeds it, and returns the top-K catalog entries.
"""

from __future__ import annotations

import logging

from retrieval import embedder, vector_store

logger = logging.getLogger(__name__)


def retrieve(messages: list[dict], k: int = 15) -> list[dict]:
    """
    Build a retrieval query from the last 2–3 user turns,
    embed it, and return the top-K matching catalog entries.

    Args:
        messages: Full conversation history. Each dict has 'role' and 'content'.
        k: Number of results to retrieve.

    Returns:
        List of catalog entry dicts.
    """
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_turns:
        logger.warning("No user turns found in messages for retrieval.")
        return []

    # Use last 3 user turns for richer context
    query_text = " ".join(user_turns[-3:])
    logger.debug("Retrieval query: %r", query_text[:120])

    query_vector = embedder.encode([query_text])
    results = vector_store.search(query_vector[0], k=k)
    logger.debug("Retrieved %d catalog entries.", len(results))
    return results
