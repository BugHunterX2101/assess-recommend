"""
POST /chat — main conversation endpoint.

Validates the request schema, enforces a 28-second timeout around
the agent call, and returns a structured ChatResponse.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from agent import agent
from app.models.request import ChatRequest
from app.models.response import ChatResponse, RecommendationItem

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_settings(request: Request):
    return request.app.state.settings


def _get_llm_client(request: Request):
    return request.app.state.llm_client


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    body: ChatRequest,
    request: Request,
) -> ChatResponse:
    """
    Submit the full conversation history and receive the agent's next turn.

    - **messages**: Full conversation history (role + content pairs).
    - Returns: reply text, 0–10 catalog-grounded recommendations, and end_of_conversation flag.
    """
    cfg = request.app.state.settings
    llm_client = request.app.state.llm_client

    messages = [m.model_dump() for m in body.messages]

    # Count user turns (each user message = 1 turn)
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > cfg.max_turns:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation exceeds maximum of {cfg.max_turns} turns.",
        )

    # Enforce timeout around the agent call
    timeout = cfg.request_timeout_s - 2  # Leave 2s for marshalling
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.run,
                messages,
                llm_client,
                cfg.llm_model,
                cfg.max_turns,
                cfg.top_k_retrieval,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error("Agent call timed out after %d seconds.", timeout)
        raise HTTPException(status_code=408, detail="Request timed out. Please try again.")
    except Exception as exc:
        logger.exception("Agent call failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"[DEBUG] {type(exc).__name__}: {exc}")

    # Build validated response
    recommendations = [
        RecommendationItem(
            name=rec["name"],
            url=rec["url"],
            test_type=rec.get("test_type", []),
        )
        for rec in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recommendations,
        end_of_conversation=result.get("end_of_conversation", False),
    )
