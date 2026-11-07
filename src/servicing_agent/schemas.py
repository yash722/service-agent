from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "fde-servicing-agent"
    milestone: int = 1


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)


class ChatResponse(BaseModel):
    reply: str
    intent: Literal["ticket_status", "complex_request"]
    execution_id: str
    steps: list[str]

