"""Pydantic request models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content must not be empty.")
        if len(v) > 4000:
            raise ValueError("Message content exceeds maximum length of 4000 characters.")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def check_turn_limit(cls, v: list[Message]) -> list[Message]:
        if len(v) == 0:
            raise ValueError("messages array must not be empty.")
        if len(v) > 16:
            raise ValueError("Conversation exceeds maximum of 8 turns (16 messages).")
        if v[0].role != "user":
            raise ValueError("The first message must have role 'user'.")
        return v
