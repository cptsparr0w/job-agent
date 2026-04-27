"""Claude client. Wraps the Anthropic SDK with two things the router needs:
prompt caching and per-model cost estimation.

Prompt caching design: for the tailor stage (cover letters), the system
prompt + the long context (résumé, voice guidelines, candidate brief) is
~5K tokens of stable content. We mark it as `cache_control` and Anthropic
caches it for ~5 minutes between calls. Across a batch of 20 cover
letters that runs in 5 minutes, calls 2-20 read from cache at ~10% the
input price.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic


# Pricing per 1M tokens (USD). Update from /v1/models or the pricing page
# when models change. These are placeholders — verify before going live.
# Format: (input, output, cache_write_5m, cache_read)
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-opus-4-7": (15.00, 75.00, 18.75, 1.50),
}


class ClaudeClient:
    def __init__(self, api_key: str, fast_model: str, smart_model: str):
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None
        self.fast_model = fast_model
        self.smart_model = smart_model

    async def complete(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        cache_system: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """Run a completion. Returns (text, usage_dict).

        If `cache_system=True` and `system` is a string, we wrap it in a
        cache-control block. If `system` is already a list, we trust the
        caller and pass it through.
        """
        if self.client is None:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        sys_param: list[dict[str, Any]] | str
        if cache_system and isinstance(system, str):
            sys_param = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            sys_param = system

        resp = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=sys_param,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
        )

        # Concatenate text blocks (we don't use tools yet)
        text = "".join(
            block.text  # type: ignore[union-attr]
            for block in resp.content
            if block.type == "text"
        )

        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": getattr(
                resp.usage, "cache_read_input_tokens", 0
            ) or 0,
            "cache_creation_input_tokens": getattr(
                resp.usage, "cache_creation_input_tokens", 0
            ) or 0,
        }
        return text, usage

    @staticmethod
    def estimate_cost(model: str, usage: dict[str, int]) -> float:
        """Cost in USD. Conservative — when model isn't in the table we
        fall back to Sonnet pricing."""
        rates = _PRICING.get(model, _PRICING["claude-sonnet-4-6"])
        in_rate, out_rate, cache_write_rate, cache_read_rate = rates
        # input_tokens already EXCLUDES cached/created tokens in the API,
        # so we charge them separately.
        cost = (
            usage["input_tokens"] * in_rate
            + usage["output_tokens"] * out_rate
            + usage.get("cache_creation_input_tokens", 0) * cache_write_rate
            + usage.get("cache_read_input_tokens", 0) * cache_read_rate
        ) / 1_000_000
        return cost
