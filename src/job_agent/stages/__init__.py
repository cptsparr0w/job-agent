"""Pipeline stages. Each stage is a function:

    async def run(db, router, profile, app_row) -> None

It reads the application's current state, does its work, and transitions
the state. Idempotent — re-running on a row in a later state is a no-op.

The orchestrator imports stages by their state name; see orchestrator.py.
"""
