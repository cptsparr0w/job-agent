"""Qwen client via LM Studio's OpenAI-compatible endpoint.

Two completion methods:

- complete()         single pass. Used by the FAST tier (parse, filter, score).
                      Supports JSON-schema-constrained generation when LM Studio
                      and the loaded model support it.

- refined_complete() three passes — draft, critique, rewrite. Used by the
                      REFINED tier (review, tailor). Recovers most of the
                      quality gap a smart cloud model would have provided,
                      at the cost of ~3x latency.

LM Studio caches KV state for identical prefixes automatically. Keep the
stable content (résumé, voice guidelines, candidate brief) at the start
of the system prompt and it stays cached across calls.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI


class QwenClient:
    def __init__(self, base_url: str, model: str):
        # LM Studio doesn't validate the API key, but the SDK requires one.
        self.client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Single-pass completion. Returns (text, usage_dict).

        If `json_schema` is provided, request constrained generation via
        OpenAI's response_format. Most current LM Studio models support
        this; if not, the server falls back to free-form and the system
        prompt's JSON instruction has to carry the load.
        """
        full_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        resp = await self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return text, usage

    async def refined_complete(
        self,
        *,
        draft_system: str,
        critique_system: str,
        rewrite_system: str,
        user_message: str,
        max_tokens: int = 2048,
    ) -> tuple[str, dict[str, int]]:
        """Three-pass refinement. Returns (final_text, total_usage).

        Pass 1 — draft:    higher temperature, prose-friendly
        Pass 2 — critique: low temperature, finds AI tells and weak phrases
        Pass 3 — rewrite:  medium temperature, integrates critique into draft

        Total usage is summed across the three passes for cost accounting
        parity with the single-call API. (Local cost is still $0; the
        tokens count is for observability only.)
        """
        # Pass 1: draft
        draft, u1 = await self.complete(
            system=draft_system,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        # Pass 2: critique. We give the critic the original task AND the
        # draft, then ask for specific issues — not a rewrite, just a list.
        critique, u2 = await self.complete(
            system=critique_system,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<original_task>\n{user_message}\n</original_task>\n\n"
                        f"<draft>\n{draft}\n</draft>\n\n"
                        "Critique this draft. Be specific. List concrete AI tells, "
                        "weak phrases, generic openers, and structural issues. "
                        "Each item should reference the exact text it's about. "
                        "Do not rewrite — only critique."
                    ),
                }
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        # Pass 3: rewrite. The rewriter sees task + draft + critique.
        final, u3 = await self.complete(
            system=rewrite_system,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<original_task>\n{user_message}\n</original_task>\n\n"
                        f"<draft>\n{draft}\n</draft>\n\n"
                        f"<critique>\n{critique}\n</critique>\n\n"
                        "Rewrite the draft, addressing every item in the critique. "
                        "Keep what was good. Output the rewritten content only — "
                        "no preamble, no explanation."
                    ),
                }
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )

        usage = {
            "input_tokens": u1["input_tokens"] + u2["input_tokens"] + u3["input_tokens"],
            "output_tokens": (
                u1["output_tokens"] + u2["output_tokens"] + u3["output_tokens"]
            ),
        }
        return final, usage
