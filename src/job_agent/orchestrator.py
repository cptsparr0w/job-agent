"""Orchestrator. The state machine loop.

Design: each tick, the orchestrator queries `application` for rows in
each non-terminal state, runs the matching stage on each, and moves on.
Stages are async, so a tick can run many in parallel up to a per-stage
concurrency cap.

This is intentionally simple: no LangGraph, no Celery, no message bus.
Just a function and a dispatch table. For a single-user job-search
pipeline that's the right scale, and it's the right scale for the cert
exam writeup too — every architectural decision is legible.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import structlog

from .db import DB
from .llm.router import BudgetExceeded, LLMRouter
from .models import ApplicationRow, ApplicationState
from .stages import parse, score, tailor, stubs

log = structlog.get_logger(__name__)


# Concurrency caps per state. Qwen calls are local — no point in massive
# parallelism, the GPU is the bottleneck. Claude calls hit a remote API
# with rate limits, but Anthropic's defaults are generous.
CONCURRENCY: dict[ApplicationState, int] = {
    ApplicationState.DISCOVERED: 4,   # parse with Qwen
    ApplicationState.PARSED: 4,        # filter with Qwen
    ApplicationState.SCORED: 2,        # review with Claude
    ApplicationState.REVIEWED: 2,      # tailor with Claude
    ApplicationState.TAILORED: 1,      # form fill — browser, serial
}


async def tick(
    db: DB,
    router: LLMRouter,
    profile: dict,
    candidate_assets: dict[str, str],
) -> dict[str, int]:
    """One pass through every state. Returns counts of advances by state."""
    counts: dict[str, int] = {}

    # Map states to bound stage runners. Each bound runner takes only
    # the application row, since db/router/profile are closed over.
    runners: dict[ApplicationState, Callable[[ApplicationRow], Awaitable[None]]] = {
        ApplicationState.DISCOVERED: lambda a: parse.run(db, router, a),
        ApplicationState.PARSED: lambda a: score.run(db, router, a, profile),
        ApplicationState.SCORED: lambda a: stubs.review(
            db, router, a, profile, candidate_assets
        ),
        ApplicationState.REVIEWED: lambda a: tailor.run(
            db, router, a, profile, candidate_assets
        ),
        ApplicationState.TAILORED: lambda a: stubs.fill_form(db, router, a),
    }

    for state, runner in runners.items():
        rows: list[ApplicationRow] = []
        async for row in db.applications_in_state(state, limit=50):
            rows.append(row)
        if not rows:
            continue

        sem = asyncio.Semaphore(CONCURRENCY.get(state, 1))

        async def _run_one(a: ApplicationRow, runner=runner) -> None:
            async with sem:
                try:
                    await runner(a)
                except BudgetExceeded as e:
                    log.warning("budget_exceeded", app_id=a.id, error=str(e))
                    raise  # bubble up; halts the tick
                except NotImplementedError as e:
                    log.info("stage_not_implemented", state=state.value, error=str(e))
                    # Don't transition — stage is a stub
                except Exception as e:
                    log.exception("stage_failed", app_id=a.id, state=state.value)
                    await db.transition(
                        a.id, ApplicationState.ERRORED, error=f"{state.value}: {e}"
                    )

        try:
            await asyncio.gather(*[_run_one(r) for r in rows])
        except BudgetExceeded:
            log.warning("halting_tick", reason="budget")
            break
        counts[state.value] = len(rows)

    return counts


async def run_loop(
    db: DB,
    router: LLMRouter,
    profile: dict,
    candidate_assets: dict[str, str],
    interval_s: float = 60.0,
    once: bool = False,
) -> None:
    """Main entry point. Runs ticks forever, or once if requested."""
    while True:
        # Discover step (cross-cutting; runs once per tick rather than
        # per-state because it creates new rows).
        try:
            await stubs.discover(db, profile)
        except NotImplementedError:
            log.info("discover_stub", note="implement to enable phase 1")

        counts = await tick(db, router, profile, candidate_assets)
        log.info("tick_complete", counts=counts)

        if once:
            return
        await asyncio.sleep(interval_s)
