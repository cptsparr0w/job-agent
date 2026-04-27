"""CLI. `python -m job_agent.cli <subcommand>`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import structlog
import yaml
from dotenv import load_dotenv

from .db import get_db
from .llm.router import LLMRouter
from .orchestrator import run_loop, tick
from .stages.tailor import load_candidate_assets

# Auto-load .env on import
load_dotenv()


def _setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


@click.group()
def main() -> None:
    """job-agent — draft-and-stage job application pipeline."""
    _setup_logging()


@main.command("init-db")
def cmd_init_db() -> None:
    """Create the schema if it doesn't exist."""
    db = get_db()
    asyncio.run(db.init_schema(Path(__file__).parents[2] / "migrations"))
    click.echo(f"Schema applied to {db.path}")


def _load_profile() -> dict:
    profile_path = Path("config/profile.yaml")
    if not profile_path.exists():
        raise click.UsageError(
            "config/profile.yaml not found. "
            "Copy config/profile.yaml.example and edit."
        )
    return yaml.safe_load(profile_path.read_text())


@main.command()
@click.option("--once", is_flag=True, help="Run one tick and exit.")
@click.option("--interval", type=float, default=60.0, help="Seconds between ticks.")
def run(once: bool, interval: float) -> None:
    """Run the orchestrator loop."""
    db = get_db()
    router = LLMRouter(db)
    profile = _load_profile()
    candidate_assets = load_candidate_assets(profile)
    asyncio.run(run_loop(db, router, profile, candidate_assets, interval, once))


@main.command()
def status() -> None:
    """Show counts of applications by state and today's spend."""

    async def _go() -> None:
        from .models import ApplicationState

        db = get_db()
        click.echo("Applications by state:")
        for state in ApplicationState:
            count = 0
            async for _ in db.applications_in_state(state, limit=10_000):
                count += 1
            click.echo(f"  {state.value:20s}  {count}")
        spend = await db.daily_spend_usd("claude")
        click.echo(f"\nToday's Claude spend: ${spend:.4f}")

    asyncio.run(_go())


if __name__ == "__main__":
    main()
