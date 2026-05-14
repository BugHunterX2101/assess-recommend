"""Pydantic response models."""

from __future__ import annotations

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: list[str]


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool
