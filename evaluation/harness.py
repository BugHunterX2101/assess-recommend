"""
Evaluation harness — replays scripted traces against the live POST /chat API
and computes evaluation metrics.

Usage:
  python scripts/run_eval.py --base-url http://localhost:8000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx

from evaluation.metrics import TraceResult, aggregate_report

logger = logging.getLogger(__name__)

TRACES_DIR = Path(__file__).parent / "traces"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_traces() -> list[dict]:
    traces = []
    for path in sorted(TRACES_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            traces.append(json.load(f))
    logger.info("Loaded %d evaluation traces.", len(traces))
    return traces


def _check_schema(response: dict) -> bool:
    required = {"reply", "recommendations", "end_of_conversation"}
    if not required.issubset(response.keys()):
        return False
    if not isinstance(response["reply"], str) or not response["reply"]:
        return False
    if not isinstance(response["recommendations"], list):
        return False
    if not isinstance(response["end_of_conversation"], bool):
        return False
    for rec in response["recommendations"]:
        if not all(k in rec for k in ("name", "url", "test_type")):
            return False
    return True


def _check_urls(response: dict, catalog_urls: set[str]) -> bool:
    for rec in response.get("recommendations", []):
        if rec.get("url") not in catalog_urls:
            return False
    return True


def run_harness(
    base_url: str,
    catalog_path: str = "data/catalog.json",
    timeout: float = 35.0,
) -> dict:
    """
    Run the full evaluation harness and return the aggregate report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load catalog for URL validation
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_urls = {e["url"] for e in catalog}
    catalog_names = {e["name"] for e in catalog}

    traces = load_traces()
    if not traces:
        logger.warning("No traces found in %s", TRACES_DIR)
        return {}

    results: list[TraceResult] = []

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for trace in traces:
            trace_id = trace.get("id", "unknown")
            expected_names = trace.get("expected_shortlist", [])
            scripted_turns = trace.get("turns", [])

            messages: list[dict] = []
            all_recommended: list[str] = []
            turn_count = 0
            schema_valid = True
            urls_valid = True
            end_reached = False

            for turn in scripted_turns:
                user_msg = turn.get("user")
                if not user_msg:
                    continue

                messages.append({"role": "user", "content": user_msg})
                turn_count += 1

                try:
                    resp = client.post("/chat", json={"messages": messages})
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.error("[%s] Turn %d failed: %s", trace_id, turn_count, exc)
                    schema_valid = False
                    break

                if not _check_schema(data):
                    schema_valid = False
                if not _check_urls(data, catalog_urls):
                    urls_valid = False

                for rec in data.get("recommendations", []):
                    name = rec.get("name", "")
                    if name and name not in all_recommended:
                        all_recommended.append(name)

                messages.append({"role": "assistant", "content": data["reply"]})

                if data.get("end_of_conversation"):
                    end_reached = True
                    break

            results.append(
                TraceResult(
                    trace_id=trace_id,
                    turns=turn_count,
                    schema_valid=schema_valid,
                    urls_valid=urls_valid,
                    recommended_names=all_recommended,
                    expected_names=expected_names,
                    end_of_conversation_reached=end_reached,
                )
            )
            logger.info(
                "[%s] turns=%d schema=%s urls=%s recall@10=%.2f",
                trace_id,
                turn_count,
                schema_valid,
                urls_valid,
                len(set(expected_names) & set(all_recommended)) / max(len(expected_names), 1),
            )

    report = aggregate_report(results, catalog_names)

    # Save report
    report_path = REPORTS_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved to %s", report_path)

    return report
