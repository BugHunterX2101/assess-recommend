"""
Main conversation agent orchestrator.

Flow per request:
  1. Classify conversation state
  2. Check guardrails on last user message
  3. Retrieve top-K catalog entries
  4. Build system prompt
  5. Call LLM
  6. Parse and validate response
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI, RateLimitError

from agent import guardrails, prompt_builder, response_parser, state_classifier
from retrieval import retriever, vector_store

logger = logging.getLogger(__name__)


def run(
    messages: list[dict],
    llm_client: OpenAI,
    model: str,
    max_turns: int = 8,
    top_k: int = 15,
) -> dict:
    """
    Run one turn of the conversation agent.

    Args:
        messages: Full conversation history (role + content dicts).
        llm_client: Initialised OpenAI-compatible client.
        model: LLM model identifier string.
        max_turns: Maximum conversation turns from config.
        top_k: Number of catalog entries to retrieve.

    Returns:
        Dict: { reply, recommendations, end_of_conversation }
    """
    # ── 1. State classification ────────────────────────────────────────────────
    state = state_classifier.classify(messages, max_turns=max_turns)
    logger.info("Agent state: %s (turn %d)", state.value, _user_turn_count(messages))

    # ── 2. Guardrails ─────────────────────────────────────────────────────────
    last_user_msg = _last_user_message(messages)
    if last_user_msg:
        guard = guardrails.check_guardrails(last_user_msg)
        if guard.blocked:
            logger.warning("Guardrail triggered: %s", guard.reason)
            return response_parser.guardrail_response(guard.reason)

    # ── 3. Retrieval ──────────────────────────────────────────────────────────
    catalog_entries = retriever.retrieve(messages, k=top_k)
    if not catalog_entries:
        logger.warning("Retrieval returned no results — using empty catalog context.")

    # ── 4. Prompt assembly ────────────────────────────────────────────────────
    system_prompt = prompt_builder.build_system_prompt(catalog_entries, state)

    # Build message list for the LLM (system + history)
    llm_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    # ── 5. LLM call with retry on rate-limit ─────────────────────────────────
    raw_response = ""
    _max_retries = 3
    _last_exc: Exception | None = None
    for _attempt in range(_max_retries):
        try:
            completion = llm_client.chat.completions.create(
                model=model,
                messages=llm_messages,
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw_response = completion.choices[0].message.content or ""
            logger.debug("Raw LLM response (%d chars): %r", len(raw_response), raw_response[:200])
            _last_exc = None
            break
        except RateLimitError as exc:
            _last_exc = exc
            if _attempt < _max_retries - 1:
                _wait = (_attempt + 1) * 15
                logger.warning(
                    "Rate limit hit (attempt %d/%d) — retrying in %ds",
                    _attempt + 1, _max_retries, _wait,
                )
                time.sleep(_wait)
            else:
                logger.error("Rate limit exceeded after %d attempts", _max_retries)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise
    if _last_exc is not None:
        raise _last_exc

    # ── 6. Parse + validate ───────────────────────────────────────────────────
    valid_urls = vector_store.get_all_urls()
    result = response_parser.parse_response(raw_response, valid_urls)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_turn_count(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return ""
