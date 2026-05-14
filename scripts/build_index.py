#!/usr/bin/env python
"""
build_index.py — one-shot ingestion script.

Steps:
  1. Attempt to scrape the SHL catalog live (falls back to static catalog.json)
  2. Parse and validate entries
  3. Save to data/catalog.json
  4. Embed all entries and build the FAISS index
  5. Save data/faiss.index and data/catalog_metadata.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from ingestion.indexer import build_index
from ingestion.parser import parse_catalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_index")


def main() -> None:
    logger.info("=== SHL Catalog Ingestion Pipeline ===")

    # Step 1: Load static catalog (pre-populated from SHL product catalog API)
    catalog_path = Path(settings.catalog_path)
    if not catalog_path.exists():
        logger.critical("catalog.json not found at %s. Aborting.", catalog_path)
        sys.exit(1)

    with open(catalog_path, encoding="utf-8") as f:
        raw_entries = json.load(f)
    logger.info("Loaded %d raw entries from %s", len(raw_entries), catalog_path)

    # Step 2: Parse + validate
    catalog = parse_catalog(raw_entries)
    if not catalog:
        logger.critical("No valid catalog entries found. Aborting.")
        sys.exit(1)

    # Step 4 & 5: Embed + index
    build_index(
        catalog=catalog,
        embed_model=settings.embed_model,
        index_path=settings.vector_store_path,
        metadata_path=settings.catalog_metadata_path,
    )

    logger.info("=== Ingestion complete. Service is ready to start. ===")


if __name__ == "__main__":
    main()
