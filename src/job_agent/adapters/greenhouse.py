"""Greenhouse adapter.

This is a SCAFFOLD. The interface is fixed by base.py; the methods below
are TODO stubs for Qwen to flesh out. Greenhouse is the easiest ATS:
predictable URL pattern, public job board JSON, mostly stable selectors.

Discover: GET https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
Fill:     each posting's apply form lives at boards.greenhouse.io/{slug}/jobs/{id}/apply

Qwen brief for filling this in:
  1. Implement discover() using httpx to hit the JSON API. Map fields to
     DiscoveredJob. The `content` field is HTML — pass through to raw_html.
  2. Implement fill_form() with Playwright. Greenhouse forms have stable
     `aria-label` attributes; prefer `page.get_by_label()`. File inputs
     use a hidden <input type=file>; use page.set_input_files().
  3. After each fill, return a FillResult. Don't click submit.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from .base import ATSAdapter, DiscoveredJob, FieldFillRequest, FillResult, register


@register
class GreenhouseAdapter(ATSAdapter):
    name = "greenhouse"

    async def discover(self, board: str, **kwargs: Any) -> AsyncIterator[DiscoveredJob]:
        # TODO(qwen): implement. See module docstring for API details.
        # Hint: yield from a list comprehension over the JSON API response.
        raise NotImplementedError("Qwen: implement greenhouse discover()")
        if False:  # pragma: no cover
            yield  # type: ignore

    async def fill_form(
        self,
        application_url: str,
        fields: list[FieldFillRequest],
        **kwargs: Any,
    ) -> list[FillResult]:
        # TODO(qwen): implement with Playwright. See module docstring.
        raise NotImplementedError("Qwen: implement greenhouse fill_form()")
