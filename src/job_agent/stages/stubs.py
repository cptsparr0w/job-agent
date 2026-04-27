"""Stub stages. Qwen fills these in following the patterns in parse.py
and tailor.py.

Each function takes (db, router, app_row, profile) and advances state
on success.
"""

from __future__ import annotations

import structlog

from ..db import DB
from ..llm.router import LLMRouter
from ..models import ApplicationRow, ApplicationState

log = structlog.get_logger(__name__)


async def discover(db: DB, profile: dict) -> int:
    """Stage 1: discover. Iterate over enabled sources, run their adapters,
    upsert jobs and create applications in DISCOVERED state.

    Returns the number of new applications created.

    TODO(qwen): implement. Iterate `profile['sources']`, look up the
    adapter in REGISTRY, await adapter.discover(board), upsert each job,
    create one application row per (new) job in DISCOVERED state.
    """
    raise NotImplementedError("Qwen: implement discover stage")


async def score(
    db: DB,
    router: LLMRouter,
    app_row: ApplicationRow,
    profile: dict,
) -> None:
    """Stage 3a: filter. Qwen scores fit. If below cutoff, transition to
    FILTERED_OUT (terminal); otherwise to SCORED.

    TODO(qwen): implement. Pattern from parse.py — JSON-schema-constrained
    output asking for a FitScore, parse, transition. Cutoff lives in
    profile['scoring']['filter_cutoff']. Use ModelTier.QWEN_FAST.
    """
    raise NotImplementedError("Qwen: implement score stage")


async def review(
    db: DB,
    router: LLMRouter,
    app_row: ApplicationRow,
    profile: dict,
    candidate_assets: dict[str, str],
) -> None:
    """Stage 3b: strategic review. Three-pass Qwen looks at top-N scored
    jobs and decides apply/skip/flag. Below cutoff → REVIEW_REJECTED, else
    REVIEWED.

    TODO: implement using ModelTier.QWEN_REFINED with role-specialized
    system prompts. Pattern from tailor.py — drafter writes the
    assessment, critic finds weak points, rewriter integrates. Output
    a ReviewVerdict JSON with score and rationale.
    """
    raise NotImplementedError("Implement review stage with QWEN_REFINED")


async def fill_form(db: DB, router: LLMRouter, app_row: ApplicationRow) -> None:
    """Stage 5: form fill. Look up the adapter for the job's source,
    build FieldFillRequest list from the application + candidate profile,
    call adapter.fill_form(). Transition to STAGED on success.

    Vision fallback: if any FillResult has success=False, run the vision
    sub-stage with ModelTier.QWEN_VISION (requires a multimodal model
    loaded in LM Studio, e.g. Qwen2.5-VL).

    TODO: implement.
    """
    raise NotImplementedError("Implement fill_form stage")
