"""Data models. Every row in `application` has a `state`; the orchestrator
advances rows through states by running the matching stage.

Design note: states are explicit strings, not Pydantic enums in the DB
column, so we can add new states in migrations without breaking old rows.
The Python enum is the source of truth for the orchestrator dispatch.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ApplicationState(StrEnum):
    """The state machine. Order matters — the orchestrator dispatches in
    this order, but each stage is idempotent so re-runs are safe."""

    DISCOVERED = "discovered"      # row exists, raw HTML stored
    PARSED = "parsed"              # Qwen extracted structured JD
    FILTERED_OUT = "filtered_out"  # terminal: didn't pass Qwen filter
    SCORED = "scored"              # Qwen filter passed
    REVIEW_REJECTED = "review_rejected"  # terminal: Claude said no
    REVIEWED = "reviewed"          # Claude said yes
    TAILORED = "tailored"          # cover letter + answers drafted
    STAGED = "staged"              # form filled, in review queue
    SUBMITTED = "submitted"        # terminal: human clicked submit
    ERRORED = "errored"            # needs human intervention

    @property
    def is_terminal(self) -> bool:
        return self in {
            ApplicationState.FILTERED_OUT,
            ApplicationState.REVIEW_REJECTED,
            ApplicationState.SUBMITTED,
        }


# Stages dispatch on the *current* state. After a stage runs successfully,
# it transitions the row to the next state.
NEXT_STATE: dict[ApplicationState, ApplicationState] = {
    ApplicationState.DISCOVERED: ApplicationState.PARSED,
    ApplicationState.PARSED: ApplicationState.SCORED,  # may go FILTERED_OUT
    ApplicationState.SCORED: ApplicationState.REVIEWED,  # may go REVIEW_REJECTED
    ApplicationState.REVIEWED: ApplicationState.TAILORED,
    ApplicationState.TAILORED: ApplicationState.STAGED,
    ApplicationState.STAGED: ApplicationState.SUBMITTED,  # human-driven
}


# --- Source data shapes ---


class JobRecord(BaseModel):
    """A discovered job posting. `parsed_json` is the structured form Qwen
    produces from `raw_html` in the parse stage."""

    id: str
    source: str
    source_id: str
    url: str
    company: str | None = None
    title: str | None = None
    location: str | None = None
    raw_html: str | None = None
    parsed_json: ParsedJob | None = None
    discovered_at: datetime


class ParsedJob(BaseModel):
    """Qwen's structured extraction. Schema is deliberately flat so Qwen
    can fill it without nested-object reasoning."""

    title: str
    company: str
    location: str
    remote_policy: str | None = None  # "Remote" | "Hybrid" | "Onsite"
    seniority: str | None = None       # "IC4" | "Senior" | "Principal" | ...
    comp_min_usd: int | None = None
    comp_max_usd: int | None = None
    summary: str = Field(description="2-3 sentence role summary")
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(
        default_factory=list,
        description="e.g. ['saas-security', 'iam', 'm-and-a']",
    )
    apply_url: str | None = None
    raw_text: str | None = None  # preserved for full-text search later


class FitScore(BaseModel):
    """Qwen's first-pass score. Cheap, runs on every parsed job."""

    score: float = Field(ge=0, le=100)
    title_match: float = Field(ge=0, le=100)
    domain_match: float = Field(ge=0, le=100)
    seniority_match: float = Field(ge=0, le=100)
    location_match: float = Field(ge=0, le=100)
    comp_match: float = Field(ge=0, le=100)
    rationale: str  # one sentence


class ReviewVerdict(BaseModel):
    """Claude's strategic review. Only top-N parsed jobs get this."""

    score: float = Field(ge=0, le=100)
    verdict: str  # 'apply' | 'skip' | 'flag'
    rationale: str
    concerns: list[str] = Field(default_factory=list)


class TailoredApplication(BaseModel):
    """Claude's tailored draft. The cover letter is the main artifact;
    custom_answers covers ATS short-answer prompts."""

    cover_letter: str
    custom_answers: dict[str, str] = Field(default_factory=dict)
    resume_deltas: list[str] = Field(
        default_factory=list,
        description="Suggested résumé bullet edits to emphasize for this role",
    )


# --- DB row dataclasses (lightweight; we don't ORM) ---


class ApplicationRow(BaseModel):
    """Mirrors the `application` table."""

    id: str
    job_id: str
    state: ApplicationState
    fit_score: float | None = None
    review_score: float | None = None
    review_notes: str | None = None
    cover_letter: str | None = None
    custom_answers: dict[str, str] = Field(default_factory=dict)
    submitted_at: datetime | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


# Resolves forward reference
JobRecord.model_rebuild()


# --- LLM accounting ---


class LLMCall(BaseModel):
    """One row in `llm_call`. Used for cost dashboards and exam writeup."""

    application_id: str | None
    stage: str
    tier: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    latency_ms: int
    cost_usd: float
    called_at: datetime
    extra: dict[str, Any] | None = None  # not persisted, useful for tests
