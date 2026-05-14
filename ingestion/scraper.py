"""
SHL catalog scraper.
Attempts to scrape the live SHL product catalog.
Falls back to the static data/catalog.json if scraping fails.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# SHL catalog base URL
CATALOG_BASE = "https://www.shl.com"
CATALOG_LISTING_URL = "https://www.shl.com/solutions/products/product-catalog/"

# Fallback static catalog (bundled with the repo)
STATIC_CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _fetch(url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def _collect_detail_urls(soup: BeautifulSoup) -> list[str]:
    """Extract assessment detail page URLs from the catalog listing page."""
    urls: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"]
        if "/product-catalog/view/" in href:
            full = href if href.startswith("http") else CATALOG_BASE + href
            if full not in urls:
                urls.append(full)
    return urls


def _parse_detail_page(url: str, soup: BeautifulSoup) -> Optional[dict]:
    """Extract structured fields from a single assessment detail page."""
    try:
        # Assessment name — usually the page <h1>
        h1 = soup.find("h1")
        name = h1.get_text(strip=True) if h1 else None
        if not name:
            return None

        # Derive ID from URL slug
        slug = url.rstrip("/").split("/")[-1]

        # Test type badges / labels — SHL typically labels these near the title
        test_type: list[str] = []
        for badge in soup.find_all(class_=lambda c: c and "test-type" in c.lower()):
            txt = badge.get_text(strip=True)
            if txt and txt not in test_type:
                test_type.append(txt)

        # Also try looking for a "Type" label in a definition list
        for dt in soup.find_all("dt"):
            if "type" in dt.get_text(strip=True).lower():
                dd = dt.find_next_sibling("dd")
                if dd:
                    for val in dd.get_text(separator=",").split(","):
                        val = val.strip()
                        if val and val not in test_type:
                            test_type.append(val)

        if not test_type:
            test_type = ["Knowledge & Skills"]  # Sensible default

        # Description — first substantive paragraph after the h1
        description = ""
        for p in soup.find_all("p"):
            txt = p.get_text(strip=True)
            if len(txt) > 60:
                description = txt
                break

        return {
            "id": slug,
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": description or f"SHL assessment: {name}.",
        }
    except Exception as exc:
        logger.warning("Failed to parse detail page %s: %s", url, exc)
        return None


def scrape_catalog() -> list[dict]:
    """
    Attempt to scrape the SHL product catalog live.
    Returns the scraped entries on success; loads the static fallback on failure.
    """
    logger.info("Attempting live scrape of SHL catalog from %s", CATALOG_LISTING_URL)

    listing_soup = _fetch(CATALOG_LISTING_URL)
    if listing_soup is None:
        logger.warning("Could not reach SHL catalog listing page — using static fallback.")
        return _load_static_fallback()

    detail_urls = _collect_detail_urls(listing_soup)
    logger.info("Found %d assessment detail URLs on the listing page.", len(detail_urls))

    if not detail_urls:
        logger.warning("No detail URLs found — using static fallback.")
        return _load_static_fallback()

    entries: list[dict] = []
    for url in detail_urls:
        soup = _fetch(url)
        if soup is None:
            continue
        entry = _parse_detail_page(url, soup)
        if entry:
            entries.append(entry)
        time.sleep(0.5)  # Polite crawl delay

    if len(entries) < 5:
        logger.warning(
            "Only %d entries scraped — too few, using static fallback.", len(entries)
        )
        return _load_static_fallback()

    logger.info("Successfully scraped %d catalog entries.", len(entries))
    return entries


def _load_static_fallback() -> list[dict]:
    """Load the bundled static catalog.json as a fallback."""
    logger.info("Loading static fallback catalog from %s", STATIC_CATALOG_PATH)
    with open(STATIC_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d entries from static catalog.", len(data))
    return data
