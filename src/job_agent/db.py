"""DB access layer. Thin wrapper over aiosqlite, no ORM.

Why no ORM: the schema is simple, query patterns are predictable, and the
exam writeup is cleaner when you can point at the SQL. The cost is some
boilerplate; we keep it in one file.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from .models import (
    ApplicationRow,
    ApplicationState,
    JobRecord,
    LLMCall,
    ParsedJob,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DB:
    """Async SQLite handle. One instance per process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a connected aiosqlite Connection. Used as:

            async with db.connect() as conn:
                ...

        Note: NO `await` in front of `db.connect()`. The aiosqlite
        Connection is both an awaitable and an async context manager;
        doing both starts its background thread twice and raises
        RuntimeError("threads can only be started once").
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            conn.row_factory = aiosqlite.Row
            yield conn

    async def init_schema(self, migrations_dir: str | Path) -> None:
        """Apply all migrations in order. Idempotent."""
        migrations_dir = Path(migrations_dir)
        files = sorted(migrations_dir.glob("*.sql"))
        async with self.connect() as conn:
            for f in files:
                sql = f.read_text()
                await conn.executescript(sql)
            await conn.commit()

    # --- Job ---

    async def upsert_job(self, job: JobRecord) -> None:
        async with self.connect() as conn:
            await conn.execute(
                """
                INSERT INTO job (id, source, source_id, url, company, title,
                                 location, raw_html, parsed_json, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id) DO UPDATE SET
                    url = excluded.url,
                    company = excluded.company,
                    title = excluded.title,
                    location = excluded.location,
                    raw_html = excluded.raw_html
                """,
                (
                    job.id,
                    job.source,
                    job.source_id,
                    job.url,
                    job.company,
                    job.title,
                    job.location,
                    job.raw_html,
                    job.parsed_json.model_dump_json() if job.parsed_json else None,
                    job.discovered_at.isoformat(),
                ),
            )
            await conn.commit()

    async def set_parsed_json(self, job_id: str, parsed: ParsedJob) -> None:
        async with self.connect() as conn:
            await conn.execute(
                "UPDATE job SET parsed_json = ? WHERE id = ?",
                (parsed.model_dump_json(), job_id),
            )
            await conn.commit()

    async def get_job(self, job_id: str) -> JobRecord | None:
        async with self.connect() as conn:
            cur = await conn.execute("SELECT * FROM job WHERE id = ?", (job_id,))
            row = await cur.fetchone()
            return _row_to_job(row) if row else None

    # --- Application ---

    async def upsert_application(self, app: ApplicationRow) -> None:
        async with self.connect() as conn:
            await conn.execute(
                """
                INSERT INTO application (id, job_id, state, fit_score, review_score,
                                          review_notes, cover_letter, custom_answers,
                                          submitted_at, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state = excluded.state,
                    fit_score = excluded.fit_score,
                    review_score = excluded.review_score,
                    review_notes = excluded.review_notes,
                    cover_letter = excluded.cover_letter,
                    custom_answers = excluded.custom_answers,
                    submitted_at = excluded.submitted_at,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    app.id,
                    app.job_id,
                    app.state.value,
                    app.fit_score,
                    app.review_score,
                    app.review_notes,
                    app.cover_letter,
                    json.dumps(app.custom_answers),
                    app.submitted_at.isoformat() if app.submitted_at else None,
                    app.error,
                    app.created_at.isoformat(),
                    _now(),
                ),
            )
            await conn.commit()

    async def applications_in_state(
        self, state: ApplicationState, limit: int = 50
    ) -> AsyncIterator[ApplicationRow]:
        async with self.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM application WHERE state = ? ORDER BY created_at LIMIT ?",
                (state.value, limit),
            )
            rows = await cur.fetchall()
        for row in rows:
            yield _row_to_application(row)

    async def transition(
        self,
        app_id: str,
        new_state: ApplicationState,
        **fields,
    ) -> None:
        """Move an application to a new state, optionally updating fields.

        Allowed fields: fit_score, review_score, review_notes, cover_letter,
        custom_answers, submitted_at, error.
        """
        allowed = {
            "fit_score", "review_score", "review_notes", "cover_letter",
            "custom_answers", "submitted_at", "error",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"transition() got bad fields: {bad}")
        sets = ["state = ?", "updated_at = ?"]
        args: list[object] = [new_state.value, _now()]
        for k, v in fields.items():
            sets.append(f"{k} = ?")
            if k == "custom_answers":
                args.append(json.dumps(v))
            elif isinstance(v, datetime):
                args.append(v.isoformat())
            else:
                args.append(v)
        args.append(app_id)
        async with self.connect() as conn:
            await conn.execute(
                f"UPDATE application SET {', '.join(sets)} WHERE id = ?",
                tuple(args),
            )
            await conn.commit()

    # --- Cost log ---

    async def log_llm_call(self, call: LLMCall) -> None:
        async with self.connect() as conn:
            await conn.execute(
                """
                INSERT INTO llm_call (application_id, stage, tier, model,
                                      input_tokens, output_tokens,
                                      cache_read_tokens, cache_create_tokens,
                                      latency_ms, cost_usd, called_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.application_id,
                    call.stage,
                    call.tier,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.cache_read_tokens,
                    call.cache_create_tokens,
                    call.latency_ms,
                    call.cost_usd,
                    call.called_at.isoformat(),
                ),
            )
            await conn.commit()

    async def daily_spend_usd(self, tier_prefix: str = "claude") -> float:
        """For the budget guard."""
        async with self.connect() as conn:
            cur = await conn.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM llm_call
                WHERE tier LIKE ? || '%'
                  AND called_at >= date('now', 'start of day')
                """,
                (tier_prefix,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0


# --- row -> model adapters ---


def _row_to_job(row: aiosqlite.Row) -> JobRecord:
    parsed = None
    if row["parsed_json"]:
        parsed = ParsedJob.model_validate_json(row["parsed_json"])
    return JobRecord(
        id=row["id"],
        source=row["source"],
        source_id=row["source_id"],
        url=row["url"],
        company=row["company"],
        title=row["title"],
        location=row["location"],
        raw_html=row["raw_html"],
        parsed_json=parsed,
        discovered_at=datetime.fromisoformat(row["discovered_at"]),
    )


def _row_to_application(row: aiosqlite.Row) -> ApplicationRow:
    return ApplicationRow(
        id=row["id"],
        job_id=row["job_id"],
        state=ApplicationState(row["state"]),
        fit_score=row["fit_score"],
        review_score=row["review_score"],
        review_notes=row["review_notes"],
        cover_letter=row["cover_letter"],
        custom_answers=json.loads(row["custom_answers"]) if row["custom_answers"] else {},
        submitted_at=(
            datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None
        ),
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def get_db() -> DB:
    """Module-level convenience. Reads $DB_PATH or defaults."""
    path = os.environ.get("DB_PATH", "./data/job-agent.db")
    return DB(path)
