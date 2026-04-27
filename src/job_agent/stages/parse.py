"""Stage 2: parse. Qwen-driven.

Reads `job.raw_html`, asks Qwen to produce a ParsedJob, persists it, and
transitions the application from DISCOVERED to PARSED.

Why Qwen: this runs on every discovered job (potentially hundreds per
day). It's structured extraction — Qwen's strength. Cost-of-error is
low because the next stage (filter) re-reads the parsed output and a
bad parse just means a low fit score.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ..db import DB
from ..llm.router import LLMRouter, ModelTier
from ..models import ApplicationRow, ApplicationState, ParsedJob

log = structlog.get_logger(__name__)


SYSTEM_PROMPT = """\
You are a data extractor. Given a job posting's HTML, produce a strict JSON
object matching the schema below. Do not include any prose, markdown, or
code fences — output JSON only.

Schema:
{
  "title": string,                 // exact role title
  "company": string,
  "location": string,              // primary office or "Remote"
  "remote_policy": "Remote" | "Hybrid" | "Onsite" | null,
  "seniority": "Junior" | "Mid" | "Senior" | "Staff" | "Principal" | "Lead" | "Director" | null,
  "comp_min_usd": integer | null,  // base salary min, USD, null if not stated
  "comp_max_usd": integer | null,
  "summary": string,               // 2-3 sentences in your own words
  "must_have": string[],           // explicit "required" qualifications
  "nice_to_have": string[],        // explicit "preferred"/"bonus"
  "domain_tags": string[],         // short slugs: "saas-security", "iam", "m-and-a", etc.
  "apply_url": string | null,
  "raw_text": string               // first 2000 chars of cleaned posting text
}

Rules:
- If a field is genuinely not in the posting, use null (or [] for lists).
  Do not guess.
- Keep `summary` factual and dry. No marketing fluff.
- `domain_tags` should be 2-6 short kebab-case strings reflecting the
  domain, not generic words like "software" or "technology".
"""


async def run(
    db: DB,
    router: LLMRouter,
    app_row: ApplicationRow,
) -> None:
    """Advance one application from DISCOVERED to PARSED."""
    if app_row.state is not ApplicationState.DISCOVERED:
        return  # idempotency

    job = await db.get_job(app_row.job_id)
    if job is None or not job.raw_html:
        await db.transition(
            app_row.id,
            ApplicationState.ERRORED,
            error="parse: job missing or no raw_html",
        )
        return

    user_msg = (
        "Extract the job posting below. Output JSON only.\n\n"
        f"<posting>\n{job.raw_html[:30_000]}\n</posting>"
    )

    text, _ = await router.complete(
        tier=ModelTier.QWEN_FAST,
        stage="parse",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        application_id=app_row.id,
        max_tokens=2048,
    )

    try:
        parsed = ParsedJob.model_validate(_clean_json(text))
    except Exception as e:
        log.warning("parse_failed", app_id=app_row.id, error=str(e), output=text[:300])
        await db.transition(
            app_row.id,
            ApplicationState.ERRORED,
            error=f"parse: {e}",
        )
        return

    await db.set_parsed_json(app_row.job_id, parsed)
    await db.transition(app_row.id, ApplicationState.PARSED)
    log.info("parsed", app_id=app_row.id, title=parsed.title, company=parsed.company)


def _clean_json(text: str) -> dict[str, Any]:
    """Qwen sometimes wraps JSON in fences despite instructions. Strip them."""
    s = text.strip()
    if s.startswith("```"):
        # ```json\n...\n```
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    return json.loads(s)
