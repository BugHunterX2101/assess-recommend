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
from ingestion.scraper import scrape_catalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_index")


def main() -> None:
    logger.info("=== SHL Catalog Ingestion Pipeline ===")

    # Step 1: Scrape (or load fallback)
    raw_entries = scrape_catalog()

    # Step 2: Parse + validate
    catalog = parse_catalog(raw_entries)
    if not catalog:
        logger.critical("No valid catalog entries found. Aborting.")
        sys.exit(1)

    # Step 3: Save catalog.json
    catalog_path = Path(settings.catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    logger.info("Catalog saved: %s (%d entries)", catalog_path, len(catalog))

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
