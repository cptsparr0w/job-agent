"""Smoke tests. Mocks the router so they run offline.

Run: pytest tests/
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from job_agent.db import DB
from job_agent.models import (
    ApplicationRow,
    ApplicationState,
    JobRecord,
    LLMCall,
)


@pytest.fixture
async def db(tmp_path):
    d = DB(tmp_path / "test.db")
    await d.init_schema(Path(__file__).parents[1] / "migrations")
    return d


async def test_schema_applies(db):
    """Migration runs cleanly."""
    async with db.connect() as conn:
        cur = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r[0] for r in await cur.fetchall()}
    assert "job" in names
    assert "application" in names
    assert "form_field_resolution" in names
    assert "llm_call" in names


async def test_application_lifecycle(db):
    """Insert a job, create an app in DISCOVERED, transition through states."""
    job = JobRecord(
        id="j1",
        source="greenhouse",
        source_id="123",
        url="https://example.com/j/123",
        company="Acme",
        title="Senior Security PM",
        location="Remote",
        raw_html="<html>...</html>",
        discovered_at=datetime.now(timezone.utc),
    )
    await db.upsert_job(job)

    now = datetime.now(timezone.utc)
    app = ApplicationRow(
        id="a1",
        job_id="j1",
        state=ApplicationState.DISCOVERED,
        created_at=now,
        updated_at=now,
    )
    await db.upsert_application(app)

    rows = []
    async for r in db.applications_in_state(ApplicationState.DISCOVERED):
        rows.append(r)
    assert len(rows) == 1
    assert rows[0].id == "a1"

    await db.transition("a1", ApplicationState.PARSED, fit_score=None)
    rows = []
    async for r in db.applications_in_state(ApplicationState.PARSED):
        rows.append(r)
    assert len(rows) == 1


async def test_parse_stage_with_mocked_qwen(db):
    """Parse stage with a fake router. Verifies the dispatch + DB
    transition path without hitting any model."""
    from job_agent.stages import parse

    # seed
    job = JobRecord(
        id="j1",
        source="greenhouse",
        source_id="123",
        url="https://example.com/j/123",
        raw_html="<html>Senior Security PM at Acme</html>",
        discovered_at=datetime.now(timezone.utc),
    )
    await db.upsert_job(job)
    now = datetime.now(timezone.utc)
    app = ApplicationRow(
        id="a1",
        job_id="j1",
        state=ApplicationState.DISCOVERED,
        created_at=now,
        updated_at=now,
    )
    await db.upsert_application(app)

    # mock router
    fake_qwen_output = json.dumps(
        {
            "title": "Senior Security PM",
            "company": "Acme",
            "location": "Remote",
            "remote_policy": "Remote",
            "seniority": "Senior",
            "comp_min_usd": None,
            "comp_max_usd": None,
            "summary": "Security PM role.",
            "must_have": [],
            "nice_to_have": [],
            "domain_tags": ["saas-security"],
            "apply_url": None,
            "raw_text": "Senior Security PM at Acme",
        }
    )
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=(
            fake_qwen_output,
            LLMCall(
                application_id="a1",
                stage="parse",
                tier="qwen_fast",
                model="fake",
                input_tokens=10,
                output_tokens=10,
                latency_ms=1,
                cost_usd=0.0,
                called_at=datetime.now(timezone.utc),
            ),
        )
    )

    await parse.run(db, router, app)

    refreshed = []
    async for r in db.applications_in_state(ApplicationState.PARSED):
        refreshed.append(r)
    assert len(refreshed) == 1
    refreshed_job = await db.get_job("j1")
    assert refreshed_job.parsed_json.title == "Senior Security PM"
