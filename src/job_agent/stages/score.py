"""Stage 3a: filter. Qwen scores fit. Below cutoff → FILTERED_OUT (terminal),
above → SCORED.

Single-pass, low temperature, JSON-schema-constrained. Runs on every parsed
job so it has to be cheap; that's why it's QWEN_FAST not QWEN_REFINED.
"""

from __future__ import annotations

import json

import structlog

from ..db import DB
from ..llm.router import LLMRouter, ModelTier
from ..models import ApplicationRow, ApplicationState, FitScore

log = structlog.get_logger(__name__)


FIT_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "title_match": {"type": "number", "minimum": 0, "maximum": 100},
        "domain_match": {"type": "number", "minimum": 0, "maximum": 100},
        "seniority_match": {"type": "number", "minimum": 0, "maximum": 100},
        "location_match": {"type": "number", "minimum": 0, "maximum": 100},
        "comp_match": {"type": "number", "minimum": 0, "maximum": 100},
        "rationale": {"type": "string", "maxLength": 80},
    },
    "required": [
        "score", "title_match", "domain_match",
        "seniority_match", "location_match", "comp_match", "rationale",
    ],
    "additionalProperties": False,
}


def _build_system(profile: dict) -> str:
    targets = profile.get("targets", {})
    weights = profile.get("scoring", {}).get("weights", {})
    return (
        "You score job postings against a candidate's search profile. "
        "Output JSON only — no prose, no markdown, no fences.\n\n"
        "Each sub-score is 0-100. The overall `score` is the weighted sum "
        "of the five sub-scores using the weights provided. Round to 1 decimal.\n\n"
        f"<candidate_targets>\n{json.dumps(targets, indent=2)}\n</candidate_targets>\n\n"
        f"<scoring_weights>\n{json.dumps(weights, indent=2)}\n</scoring_weights>\n\n"
        "Sub-score guidance:\n"
        "- title_match: how closely the posting's title matches one of the target titles. "
        "Exact match = 100, related senior role = 70-90, off-target = 0-40.\n"
        "- domain_match: alignment of domain_tags with the candidate's stated domains "
        "(security, trust, SaaS posture, M&A). Strong overlap = 80-100.\n"
        "- seniority_match: 100 if seniority is in the target band; lower for over- or under-target.\n"
        "- location_match: 100 if listed location is one of the targets, else 0-50 based on remote policy.\n"
        "- comp_match: 100 if comp_max >= comp_floor; partial credit if comp_min is close; "
        "50 if comp not stated.\n"
        "- rationale: ONE sentence explaining the overall score."
    )


async def run(
    db: DB,
    router: LLMRouter,
    app_row: ApplicationRow,
    profile: dict,
) -> None:
    """Advance from PARSED to SCORED (or FILTERED_OUT)."""
    if app_row.state is not ApplicationState.PARSED:
        return

    job = await db.get_job(app_row.job_id)
    if job is None or job.parsed_json is None:
        await db.transition(
            app_row.id, ApplicationState.ERRORED, error="score: missing parsed job"
        )
        return

    parsed = job.parsed_json
    user_msg = (
        "Score this posting:\n\n"
        f"{json.dumps(parsed.model_dump(), indent=2)}"
    )

    text, _ = await router.complete(
        tier=ModelTier.QWEN_FAST,
        stage="score",
        system=_build_system(profile),
        messages=[{"role": "user", "content": user_msg}],
        application_id=app_row.id,
        max_tokens=1024,
        json_schema=FIT_SCORE_SCHEMA,
        temperature=0.4,
    )

    try:
        fit = FitScore.model_validate_json(_strip_fences(text))
    except Exception as e:
        log.warning("score_failed", app_id=app_row.id, error=str(e), output=text[:300])
        await db.transition(
            app_row.id, ApplicationState.ERRORED, error=f"score: {e}"
        )
        return

    cutoff = profile.get("scoring", {}).get("filter_cutoff", 65)
    if fit.score < cutoff:
        await db.transition(
            app_row.id,
            ApplicationState.FILTERED_OUT,
            fit_score=fit.score,
            review_notes=fit.rationale,
        )
        log.info(
            "filtered_out", app_id=app_row.id, score=fit.score, cutoff=cutoff,
            rationale=fit.rationale,
        )
        return

    await db.transition(
        app_row.id,
        ApplicationState.SCORED,
        fit_score=fit.score,
        review_notes=fit.rationale,
    )
    log.info(
        "scored", app_id=app_row.id, score=fit.score, cutoff=cutoff,
        rationale=fit.rationale,
    )


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()
