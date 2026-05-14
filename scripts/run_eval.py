#!/usr/bin/env python
"""
run_eval.py — entry point for the evaluation harness.

Usage:
  python scripts/run_eval.py
  python scripts/run_eval.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.harness import run_harness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHL Recommender Evaluation Harness")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API service.",
    )
    parser.add_argument(
        "--catalog",
        default="data/catalog.json",
        help="Path to catalog.json for URL validation.",
    )
    args = parser.parse_args()

    logger.info("Running evaluation against: %s", args.base_url)
    report = run_harness(base_url=args.base_url, catalog_path=args.catalog)

    print("\n=== Evaluation Report ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
