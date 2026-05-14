"""
Response parser — converts raw LLM output to a structured ChatResponse.

Strategy:
  1. Try json.loads() directly on the raw string.
  2. If that fails, use regex to extract the first {...} JSON block.
  3. Validate required fields and types.
  4. Strip any URLs not present in the known catalog URL set.
  5. On total failure, return a safe error response dict.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Try to parse a JSON object from a raw string."""
    raw = raw.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt 3: regex extraction of first {...} block
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _safe_list_of_str(val) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [val]
    return []


def parse_response(raw: str, valid_urls: set[str]) -> dict:
    """
    Parse the LLM output into a validated response dict.

    Args:
        raw: Raw string from the LLM.
        valid_urls: Set of known catalog URLs for validation.

    Returns:
        Dict with keys: reply (str), recommendations (list), end_of_conversation (bool).
        On failure, returns a safe fallback dict.
    """
    parsed = _extract_json(raw)

    if parsed is None:
        logger.error("Failed to parse LLM response as JSON. Raw: %r", raw[:300])
        return _fallback_response("I encountered an issue processing my response. Please try again.")

    reply = str(parsed.get("reply", "")).strip()
    if not reply:
        reply = "I'm here to help you find the right SHL assessments. Could you share more details about the role?"

    raw_recs = parsed.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    recommendations: list[dict] = []
    for item in raw_recs[:10]:  # Cap at 10
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        test_type = _safe_list_of_str(item.get("test_type", []))

        if not name or not url:
            continue

        # URL allowlist check — strip URLs not in the catalog
        if url not in valid_urls:
            logger.warning("Stripping non-catalog URL from recommendation: %s", url)
            continue

        recommendations.append({"name": name, "url": url, "test_type": test_type})

    end_of_conversation = bool(parsed.get("end_of_conversation", False))

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }


def _fallback_response(message: str) -> dict:
    return {
        "reply": message,
        "recommendations": [],
        "end_of_conversation": False,
    }


def guardrail_response(reason: str) -> dict:
    """Return a polite refusal response for guardrail triggers."""
    if reason == "prompt_injection":
        msg = (
            "I'm not able to follow that instruction. I'm here solely to help you "
            "find the right SHL assessments for your hiring needs. How can I help you today?"
        )
    else:
        msg = (
            "That topic is outside my area of expertise. I specialise exclusively in "
            "SHL Individual Test Solutions. Please describe the role you're hiring for "
            "and I'll recommend the most suitable assessments."
        )
    return {
        "reply": msg,
        "recommendations": [],
        "end_of_conversation": False,
    }
