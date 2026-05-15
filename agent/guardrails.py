"""
Guardrails — pattern-based pre-filter for:
  1. Prompt injection attempts
  2. Off-topic / out-of-scope requests
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str = ""


_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|above|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"disregard\s+your\s+(system|prior|previous)", re.IGNORECASE),
    re.compile(r"\bpretend\s+you\s+are\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\b.{0,20}\b(DAN|jailbreak|unrestricted)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+your\s+(instructions?|training|guidelines?)\b", re.IGNORECASE),
    re.compile(r"new\s+persona\s*:", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
]

_OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(salary|compensation|pay\s+scale|wage)\b", re.IGNORECASE),
    re.compile(r"\b(visa|immigration|discrimination)\b", re.IGNORECASE),
    re.compile(r"\b(legal\s+advice|sue\s+the\s+company|file\s+a\s+lawsuit)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+interview|interview\s+questions?|offer\s+letter)\b", re.IGNORECASE),
    re.compile(r"\b(write\s+me\s+(a|an)\s+(essay|poem|story|code))\b", re.IGNORECASE),
    re.compile(r"\b(stock\s+price|cryptocurrency|invest(ment|ing)?)\b", re.IGNORECASE),
    re.compile(r"\b(medical\s+advice|diagnosis|prescri(be|ption))\b", re.IGNORECASE),
]


def check_guardrails(user_message: str) -> GuardrailResult:
    """
    Run prompt injection and off-topic checks on the latest user message.
    Returns a GuardrailResult indicating whether the message should be blocked.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(user_message):
            return GuardrailResult(blocked=True, reason="prompt_injection")

    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(user_message):
            return GuardrailResult(blocked=True, reason="out_of_scope")

    return GuardrailResult(blocked=False)
