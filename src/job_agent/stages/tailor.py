"""Stage 4: tailor. Now Qwen-only via three-pass refinement.

The doctrine shift: instead of one Claude call with a long cached system
prompt, we run three Qwen calls — drafter, critic, rewriter — each with
its own system prompt tuned for that role. The stable context (résumé,
voice guidelines, candidate brief) appears in all three system prompts,
so LM Studio's KV cache can dedupe it across passes and across calls.

Quality recovers most of the Claude→Qwen gap; latency goes from ~2s to
~30s per cover letter; cost goes to zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from ..db import DB
from ..llm.router import LLMRouter, ModelTier
from ..models import ApplicationRow, ApplicationState, TailoredApplication

log = structlog.get_logger(__name__)


def _stable_context(profile: dict, candidate_assets: dict[str, str]) -> str:
    """The shared prefix across all three passes. Identical bytes →
    LM Studio reuses KV cache automatically."""
    return "\n\n".join(
        [
            "<candidate_resume>",
            candidate_assets.get("resume", ""),
            "</candidate_resume>",
            "<voice_guidelines>",
            candidate_assets.get("voice", ""),
            "</voice_guidelines>",
            "<career_brief>",
            candidate_assets.get("brief", ""),
            "</career_brief>",
            f"<targets>{json.dumps(profile.get('targets', {}), indent=2)}</targets>",
        ]
    )


def _draft_system(stable: str) -> str:
    return (
        "You are drafting a job application cover letter on the candidate's behalf. "
        "Write in the candidate's voice. Three short paragraphs. Specific, "
        "concrete, dry where possible. Avoid AI tells: no em-dashes, no "
        "'I'm thrilled to', no 'leveraging synergies', no listy bullet-spam. "
        "Sign with first name only.\n\n"
        f"{stable}\n\n"
        "Output the cover letter as plain prose. No JSON yet — that comes later."
    )


def _critique_system(stable: str) -> str:
    return (
        "You are a strict editor reviewing a draft cover letter for the "
        "candidate. List specific issues — AI tells, generic phrases, "
        "weak openings, structural problems, places where the voice "
        "doesn't match the candidate's guidelines below. Quote exact text. "
        "Do not rewrite — only critique. Output a numbered list.\n\n"
        f"{stable}"
    )


def _rewrite_system(stable: str) -> str:
    return (
        "You are rewriting a cover letter to address specific critiques. "
        "Three short paragraphs. Keep what was good in the draft; fix every "
        "issue raised in the critique. Maintain the candidate's voice "
        "throughout. Sign with first name only.\n\n"
        f"{stable}\n\n"
        "After the rewritten cover letter, on a new line beginning with "
        "'---JSON---', output a JSON object:\n"
        '{"cover_letter": <the rewritten letter as a string>, '
        '"custom_answers": {<question>: <answer>}, '
        '"resume_deltas": [<0-3 suggested resume bullet edits>]}'
    )


async def run(
    db: DB,
    router: LLMRouter,
    app_row: ApplicationRow,
    profile: dict,
    candidate_assets: dict[str, str],
) -> None:
    """Advance one application from REVIEWED to TAILORED via 3-pass Qwen."""
    if app_row.state is not ApplicationState.REVIEWED:
        return

    job = await db.get_job(app_row.job_id)
    if job is None or job.parsed_json is None:
        await db.transition(
            app_row.id, ApplicationState.ERRORED, error="tailor: missing parsed job"
        )
        return

    parsed = job.parsed_json
    custom_questions = profile.get("custom_questions", [])

    user_msg = (
        f"Draft a cover letter for this role:\n\n"
        f"Title: {parsed.title}\n"
        f"Company: {parsed.company}\n"
        f"Summary: {parsed.summary}\n"
        f"Must-have: {', '.join(parsed.must_have)}\n"
        f"Nice-to-have: {', '.join(parsed.nice_to_have)}\n"
        f"Domain tags: {', '.join(parsed.domain_tags)}\n"
        f"Strategic notes from review: {app_row.review_notes or '(none)'}\n\n"
        f"Custom questions to answer (if any): {json.dumps(custom_questions)}"
    )

    stable = _stable_context(profile, candidate_assets)

    text, _ = await router.complete(
        tier=ModelTier.QWEN_REFINED,
        stage="tailor",
        system=_draft_system(stable),  # default; overridden per-pass below
        messages=[{"role": "user", "content": user_msg}],
        application_id=app_row.id,
        max_tokens=2048,
        draft_system=_draft_system(stable),
        critique_system=_critique_system(stable),
        rewrite_system=_rewrite_system(stable),
    )

    try:
        tailored = _parse_rewrite_output(text)
    except Exception as e:
        log.warning("tailor_failed", app_id=app_row.id, error=str(e), output=text[:400])
        await db.transition(
            app_row.id, ApplicationState.ERRORED, error=f"tailor: {e}"
        )
        return

    await db.transition(
        app_row.id,
        ApplicationState.TAILORED,
        cover_letter=tailored.cover_letter,
        custom_answers=tailored.custom_answers,
    )
    log.info("tailored", app_id=app_row.id, length=len(tailored.cover_letter))


def _parse_rewrite_output(text: str) -> TailoredApplication:
    """The rewriter outputs the rewritten letter, then '---JSON---', then
    a JSON object. Split on the marker; if Qwen forgot the marker, fall
    back to treating the whole thing as the letter and producing empty
    custom_answers/deltas."""
    marker = "---JSON---"
    if marker in text:
        _, json_part = text.split(marker, 1)
        json_part = _strip_fences(json_part).strip()
        return TailoredApplication.model_validate_json(json_part)
    # Marker missing — try parsing the whole thing as JSON; if that
    # fails, treat as a bare cover letter.
    s = _strip_fences(text).strip()
    if s.startswith("{"):
        return TailoredApplication.model_validate_json(s)
    return TailoredApplication(cover_letter=text.strip(), custom_answers={}, resume_deltas=[])


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def load_candidate_assets(profile: dict) -> dict[str, str]:
    """Reads résumé / voice / brief from disk per the profile. Called once
    at startup so we keep the exact same string across calls (LM Studio
    KV cache hit requires byte-identical content)."""
    cand = profile.get("candidate", {})
    out = {}
    for key, path_key in (
        ("resume", "resume_path"),
        ("voice", "voice_guidelines_path"),
        ("brief", "context_brief_path"),
    ):
        p = cand.get(path_key)
        if p and Path(p).exists():
            out[key] = Path(p).read_text()
        else:
            out[key] = ""
    return out
