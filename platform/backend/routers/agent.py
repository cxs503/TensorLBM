"""Conversational LLM-agent endpoints for the TensorLBM platform.

Exposes ``/api/agent/*`` routes that wrap :mod:`backend.agent_core`.
The agent is intentionally stateless – the client supplies the full
conversation history on every call.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import agent_core

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field("", description="Plain text content of the turn")
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Tool calls executed by the assistant during this turn. "
            "Echo this back to the agent so it can resolve 'the job' "
            "references."
        ),
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    actions: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    used_llm: bool = False
    intent: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send one user turn; receive the assistant reply plus any tool actions."""
    resp = agent_core.chat(
        message=req.message,
        history=[t.model_dump() for t in req.history],
    )
    return ChatResponse(
        reply=resp.reply,
        actions=resp.actions,
        suggestions=resp.suggestions,
        used_llm=resp.used_llm,
        intent=resp.intent,
    )


@router.get("/capabilities")
async def capabilities() -> dict:
    """List the tools and scenarios the agent can invoke."""
    return {
        "tools": agent_core.list_tools(),
        "scenarios": [scenario for scenario, _ in agent_core._SCENARIOS],
    }


@router.get("/info")
async def info() -> dict:
    """Return runtime info about the agent backend (LLM enabled / model)."""
    return {
        "llm_enabled": agent_core._llm_enabled(),
        "llm_model": os.environ.get("TENSORLBM_LLM_MODEL", "")
        if agent_core._llm_enabled() else "",
        "fallback": "rule-based intent parser (Chinese + English)",
        "tools_count": len(agent_core.list_tools()),
    }
