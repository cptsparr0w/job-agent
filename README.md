# job-agent

Draft-and-stage job application pipeline. Discovers security PM listings,
parses them, scores them, drafts tailored applications, fills the forms, and
queues each application for human review before submit. Built as a portfolio
project for the Claude Certified Architect cert.

**Runs entirely on your local LM Studio.** No cloud API keys required.

## Architecture

Six stages, each a row in `applications` with a `state` column. The
orchestrator picks up jobs in each state and advances them.

```
1. discover   →  Playwright crawls ATS pages, stores raw HTML
2. parse      →  Qwen extracts structured job data (single pass)
3a. filter    →  Qwen scores fit (single pass), top 20 forwarded
3b. review    →  Qwen 3-pass strategic review on top 20
4. tailor     →  Qwen 3-pass cover letter draft → critique → rewrite
5. fill       →  Playwright fills the form (Qwen vision fallback)
6. review     →  You click submit
```

## The doctrine: split work by inference passes, not by model

Without a smarter cloud model to lean on, we recover quality through
**test-time compute** — multiple specialized passes by the same local
model. The drafter writes, the critic finds AI tells and weak phrases,
the rewriter integrates the critique. Three Qwen calls instead of one
Claude call, ~30s instead of ~2s, $0 instead of $0.05.

| Tier             | Use                              | Pattern             |
|------------------|----------------------------------|---------------------|
| `QWEN_FAST`      | parse, filter, score             | single pass         |
| `QWEN_REFINED`   | review, tailor, custom answers   | draft → critique → rewrite |
| `QWEN_VISION`    | unknown form fields              | single pass, vision-capable model |

LM Studio's KV cache deduplicates identical prefixes across calls, so
the stable context (résumé, voice guidelines, candidate brief) gets the
same effective prompt-caching benefit Anthropic provides — for free,
with no special API.

## Build phases

- [x] **Phase 0 — Skeleton**: models, DB, LLM router, orchestrator, multi-pass
- [ ] **Phase 1 — Discovery + parse**: Greenhouse only, end-to-end
- [ ] **Phase 2 — Score + tailor**: filter and 3-pass tailor
- [ ] **Phase 3 — Form fill**: Greenhouse + Lever adapters, Qwen vision fallback
- [ ] **Phase 4 — Review UI + observability**: HTMX queue, dashboard

## Running

Prereqs: Python 3.11+, LM Studio running an instruct model on `localhost:1234`.

```bash
pip install -e ".[dev]"
playwright install chromium
cp .env.example .env  # tweak QWEN_MODEL if needed
cp config/profile.yaml.example config/profile.yaml  # edit for your search
python -m job_agent.cli init-db
pytest tests/
python -m job_agent.cli run --once
```

## Optional cloud fallback

If you later want a Claude tier — for the highest-stakes cover letters,
or for Anthropic-only vision quirks — install the optional extra and set
your key:

```bash
pip install -e ".[claude]"
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
```

The router auto-detects the optional dep at import time and unlocks
`CLAUDE_*` tiers. With the dep absent or the key unset, only Qwen
tiers are available.

## Layout

```
src/job_agent/
├── models.py       # Pydantic + state enum
├── db.py           # aiosqlite wrapper
├── orchestrator.py # state machine loop
├── llm/
│   ├── router.py   # picks tier per task; logs cost
│   ├── qwen.py     # OpenAI-compatible client + multi-pass refine
│   └── claude.py   # optional, behind extras_require
├── adapters/
│   ├── base.py     # ATS adapter interface
│   └── greenhouse.py
└── stages/
    ├── parse.py    # QWEN_FAST
    └── tailor.py   # QWEN_REFINED — draft, critique, rewrite
```
