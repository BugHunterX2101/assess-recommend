"""
Parser module — validates and normalises raw catalog entries
into the canonical schema expected by the indexer.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"name", "url", "test_type"}


def parse_entry(raw: dict) -> Optional[dict]:
    """
    Validate and normalise a single raw catalog entry.
    Returns None if the entry is missing required fields.
    """
    missing = REQUIRED_FIELDS - raw.keys()
    if missing:
        logger.warning("Skipping entry missing fields %s: %s", missing, raw.get("name", "<unnamed>"))
        return None

    name: str = str(raw["name"]).strip()
    url: str = str(raw["url"]).strip()
    test_type = raw["test_type"]

    if not name or not url:
        return None

    # Normalise test_type to a list of strings
    if isinstance(test_type, str):
        test_type = [t.strip() for t in test_type.split(",") if t.strip()]
    elif isinstance(test_type, list):
        test_type = [str(t).strip() for t in test_type if str(t).strip()]
    else:
        test_type = ["Knowledge & Skills"]

    # Derive id from URL slug if not present
    entry_id = raw.get("id") or url.rstrip("/").split("/")[-1]

    description = str(raw.get("description", "")).strip()
    if not description:
        description = f"SHL assessment: {name}."

    return {
        "id": entry_id,
        "name": name,
        "url": url,
        "test_type": test_type,
        "description": description,
    }


def parse_catalog(raw_entries: list[dict]) -> list[dict]:
    """
    Parse and validate a list of raw catalog entries.
    Deduplicates by URL. Returns only valid entries.
    """
    seen_urls: set[str] = set()
    parsed: list[dict] = []

    for raw in raw_entries:
        entry = parse_entry(raw)
        if entry is None:
            continue
        if entry["url"] in seen_urls:
            logger.debug("Duplicate URL skipped: %s", entry["url"])
            continue
        seen_urls.add(entry["url"])
        parsed.append(entry)

    logger.info("Parsed %d valid catalog entries (from %d raw).", len(parsed), len(raw_entries))
    return parsed
