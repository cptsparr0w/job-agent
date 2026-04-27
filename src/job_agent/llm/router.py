"""LLM router. Picks Qwen tier per task, records cost (zero for local) to
the DB. Stages call this; they don't talk to clients directly.

Doctrine: split work by inference passes, not model capability.
- QWEN_FAST     single pass for parse/filter/score
- QWEN_REFINED  draft → critique → rewrite for review/tailor
- QWEN_VISION   vision-capable Qwen for unknown form fields

Anthropic SDK is an optional extra (`pip install -e ".[claude]"`) for the
case where you want to flip a tier to a cloud model later. By default
no cloud calls are made.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import structlog

from ..db import DB
from ..models import LLMCall
from .qwen import QwenClient

# Anthropic is optional. If the dep isn't installed (or no key set), we
# silently disable any Claude tiers; calling them raises a helpful error.
try:
    from .claude import ClaudeClient

    _CLAUDE_AVAILABLE = True
except ImportError:
    ClaudeClient = None  # type: ignore[assignment, misc]
    _CLAUDE_AVAILABLE = False

log = structlog.get_logger(__name__)


class ModelTier(StrEnum):
    """The tiers callers use. Mapped to concrete dispatch in the router."""

    QWEN_FAST = "qwen_fast"          # was QWEN — parse, filter, score
    QWEN_REFINED = "qwen_refined"    # was CLAUDE_SMART — review, tailor (multi-pass)
    QWEN_VISION = "qwen_vision"      # was CLAUDE_VISION — unknown form fields


class LLMRouter:
    """Routes structured calls. Stages call .complete() with a tier, a
    system prompt, and messages. The router picks the dispatch path,
    runs the call, logs cost, and returns the response text.
    """

    def __init__(self, db: DB):
        self.db = db
        self.qwen = QwenClient(
            base_url=os.environ.get("QWEN_BASE_URL", "http://localhost:1234/v1"),
            model=os.environ.get("QWEN_MODEL", "qwen3-30b-a3b-instruct"),
        )
        # Vision model is loaded under a different model name in LM Studio.
        # Allow override; fall back to the same model for users running a
        # multimodal model as primary.
        self.vision_model = os.environ.get("QWEN_VISION_MODEL", self.qwen.model)
        self.refined_model = os.environ.get("QWEN_REFINED_MODEL", self.qwen.model)
        self.refined_model = os.environ.get("QWEN_REFINED_MODEL", self.qwen.model)

    async def complete(
        self,
        *,
        tier: ModelTier,
        stage: str,
        system: str,
        messages: list[dict[str, Any]],
        application_id: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        # For QWEN_REFINED only — overrides the default uniform `system`.
        # If not provided, REFINED reuses `system` for all three roles.
        draft_system: str | None = None,
        critique_system: str | None = None,
        rewrite_system: str | None = None,
    ) -> tuple[str, LLMCall]:
        """Run a completion. Returns (text, LLMCall) where the call has
        been logged to the cost ledger.

        For tier=QWEN_REFINED, runs a three-pass draft/critique/rewrite.
        Pass-specific system prompts can be supplied; if omitted, the
        default uniform `system` is used for all three.
        """
        t0 = time.perf_counter()

        if tier is ModelTier.QWEN_FAST:
            text, usage = await self.qwen.complete(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                json_schema=json_schema,
                temperature=temperature,
            )
            model_name = self.qwen.model
        elif tier is ModelTier.QWEN_REFINED:
            # Multi-pass. Stages can supply per-pass system prompts; if
            # not, fall back to the uniform `system` for all three.
            user_message = _last_user_message(messages)
            # Temporarily swap the client's model for refined_model
            original_model = self.qwen.model
            self.qwen.model = self.refined_model
            try:
                text, usage = await self.qwen.refined_complete(
                    draft_system=draft_system or system,
                    critique_system=critique_system or _default_critique_system(),
                    rewrite_system=rewrite_system or system,
                    user_message=user_message,
                    max_tokens=max_tokens,
                )
            finally:
                self.qwen.model = original_model
            model_name = self.refined_model
        elif tier is ModelTier.QWEN_VISION:
            # Same client, but the loaded model must be vision-capable.
            # The caller is expected to put image content in messages.
            text, usage = await self.qwen.complete(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                json_schema=json_schema,
                temperature=temperature,
            )
            model_name = self.vision_model
        else:
            raise ValueError(f"unknown tier: {tier}")

        latency_ms = int((time.perf_counter() - t0) * 1000)

        call = LLMCall(
            application_id=application_id,
            stage=stage,
            tier=tier.value,
            model=model_name,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=0,
            cache_create_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,  # local; free
            called_at=datetime.now(timezone.utc),
        )
        await self.db.log_llm_call(call)

        log.info(
            "llm_call",
            stage=stage,
            tier=tier.value,
            model=model_name,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            latency_ms=latency_ms,
        )
        return text, call


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message as a string. Used by REFINED tier
    when the caller passed messages instead of a bare user_message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            # If content is a list of blocks (multimodal), concatenate text.
            return "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
    return ""


def _default_critique_system() -> str:
    """Default critic prompt. Stages can override with one tuned to their
    output type (cover letter critic vs review critic etc.)."""
    return (
        "You are a strict editor. Read the draft and list specific issues: "
        "AI tells (em-dashes, 'I'm thrilled to', 'leveraging', etc.), "
        "generic phrases, weak openings, and structural problems. "
        "Be concrete — quote the exact text in question. "
        "Do not rewrite. Output a numbered list, one issue per line."
    )


class BudgetExceeded(Exception):
    """Compat shim. Local-only Qwen has no budget to exceed."""
    pass
