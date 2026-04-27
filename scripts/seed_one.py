"""Seed one Greenhouse posting. Tries the JSON API first; falls back to
scraping the public HTML page if the API doesn't have the job."""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
from datetime import datetime, timezone

import httpx

from job_agent.db import get_db
from job_agent.models import ApplicationRow, ApplicationState, JobRecord


def parse_url(url: str) -> tuple[str, str]:
    m = re.search(r"greenhouse\.io/([^/]+)/jobs/(\d+)", url)
    if not m:
        raise ValueError(f"Not a recognizable Greenhouse URL: {url}")
    return m.group(1), m.group(2)


async def fetch_job(client, slug, job_id, fallback_url):
    api = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{job_id}?content=true"
    print(f"Trying API: {api}")
    try:
        resp = await client.get(api)
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", ""),
            "location": (data.get("location") or {}).get("name", ""),
            "raw_html": data.get("content", ""),
            "apply_url": data.get("absolute_url", fallback_url),
        }
    except httpx.HTTPStatusError as e:
        print(f"  API returned {e.response.status_code}; scraping {fallback_url}")

    resp = await client.get(fallback_url)
    resp.raise_for_status()
    html = resp.text
    title_m = re.search(r"<h1[^>]*>([^<]+)</h1>", html)
    return {
        "title": title_m.group(1).strip() if title_m else "Unknown",
        "location": "",
        "raw_html": html,
        "apply_url": fallback_url,
    }


async def main(url: str) -> None:
    slug, job_id = parse_url(url)
    company = slug.replace("-", " ").title()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        )
    }
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
        info = await fetch_job(client, slug, job_id, url)

    cjid = hashlib.sha256(f"greenhouse:{job_id}".encode()).hexdigest()[:16]
    caid = hashlib.sha256(f"{cjid}:ninaad".encode()).hexdigest()[:16]
    now = datetime.now(timezone.utc)

    db = get_db()
    await db.upsert_job(JobRecord(
        id=cjid, source="greenhouse", source_id=job_id, url=info["apply_url"],
        company=company, title=info["title"], location=info["location"],
        raw_html=info["raw_html"], discovered_at=now,
    ))
    print(f"Inserted job: {company} - {info['title']}")
    print(f"  raw_html length: {len(info['raw_html'])} chars")

    await db.upsert_application(ApplicationRow(
        id=caid, job_id=cjid, state=ApplicationState.DISCOVERED,
        created_at=now, updated_at=now,
    ))
    print(f"Created application {caid} in state DISCOVERED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/seed_one.py greenhouse-url", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
