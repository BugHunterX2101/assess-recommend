"""
State classifier — rule-based classification of the conversation state.
Maps turn count + context signals to one of:
  CLARIFY | RECOMMEND | REFINE | COMPARE | REFUSE
"""

from __future__ import annotations

import re
from enum import Enum


class AgentState(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


# Keywords that indicate sufficient context for a recommendation
_ROLE_SIGNALS = [
    r"\b(developer|engineer|analyst|manager|designer|scientist|architect|specialist|consultant|"
    r"administrator|coordinator|director|recruiter|recruitment|hr|executive|officer|lead|head|"
    r"agent|representative|rep|operator|technician|associate|trainee|intern|"
    r"nurse|doctor|physician|accountant|attorney|teller|clerk|"
    r"admin|assistant|staff|advisor|sales|counselor|worker|personnel)\b",
]
_QUALIFIER_SIGNALS = [
    r"\b(junior|mid[\s-]?level|senior|entry[\s-]?level|graduate|experienced|seniority|"
    r"years?\s+of\s+experience|\d+\s*\+?\s*years?|cxo|director[\s-]level|c[\s-]?suite)\b",
    r"\b(java|python|sql|javascript|typescript|c\+\+|\.net|rust|aws|cloud|devops|"
    r"machine\s+learning|data|excel|word|software|frontend|backend|fullstack|full[\s-]?stack|"
    r"spring|angular|react|docker|kubernetes|linux|networking)\b",
    r"\b(personality|cognitive|aptitude|situational|verbal|numerical|reasoning|skills?|"
    r"knowledge|behaviour|behavioral|safety|dependability|bilingual|language)\b",
    r"\b(remote|onsite|on[\s-]?site|hybrid|contract|permanent|full[\s-]?time|part[\s-]?time|"
    r"high[\s-]?volume|screening|selection|development|talent|audit)\b",
    r"\b(contact\s+cent(?:er|re)|call\s+cent(?:er|re)|customer\s+service|chemical|"
    r"manufacturing|industrial|healthcare|hospital|clinic|retail|warehouse|plant)\b",
]

_REFINE_SIGNALS = [
    r"\b(actually|instead|change|update|replace|add|remove|also\s+include|without)\b",
    r"\b(more\s+focus|less\s+focus|prefer|rather|different)\b",
]

_COMPARE_SIGNALS = [
    r"\b(compare|comparison|difference|vs\.?|versus|which\s+(is\s+)?better|what.{0,15}differ)\b",
    r"\b(pros?\s+and\s+cons?|advantages?|disadvantages?)\b",
]

MAX_CLARIFY_TURNS = 3


def _count_user_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def _has_signal(text: str, patterns: list[str]) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _has_sufficient_context(messages: list[dict]) -> bool:
    """
    Context is sufficient if the combined user turns contain at least one role signal
    and at least one qualifier signal.
    """
    user_text = " ".join(m["content"] for m in messages if m.get("role") == "user")
    has_role = _has_signal(user_text, _ROLE_SIGNALS)
    has_qualifier = any(_has_signal(user_text, [p]) for p in _QUALIFIER_SIGNALS)
    return has_role and has_qualifier


def classify(messages: list[dict], max_turns: int = 8) -> AgentState:
    """
    Classify the conversation state based on turn count and message content.

    Args:
        messages: Full conversation history.
        max_turns: Maximum allowed turns (from config).

    Returns:
        AgentState enum value.
    """
    user_turn_count = _count_user_turns(messages)
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m["content"]
            break

    # Force recommend as turn cap approaches — user has run out of turns
    if user_turn_count >= max_turns - 1:
        return AgentState.RECOMMEND

    # Refine: user is editing a prior constraint
    if user_turn_count > 1 and _has_signal(last_user_msg, _REFINE_SIGNALS):
        return AgentState.REFINE

    # Compare: explicit comparison request
    if _has_signal(last_user_msg, _COMPARE_SIGNALS):
        return AgentState.COMPARE

    # Clarify: turn 1 and vague, or still insufficient context (max 3 clarify turns)
    if not _has_sufficient_context(messages) and user_turn_count <= MAX_CLARIFY_TURNS:
        return AgentState.CLARIFY

    return AgentState.RECOMMEND
