-- Phase 0 schema. Future migrations go in 002_*.sql, etc.

CREATE TABLE IF NOT EXISTS job (
    id              TEXT PRIMARY KEY,             -- hash(source, source_id)
    source          TEXT NOT NULL,                -- 'greenhouse' | 'lever' | ...
    source_id       TEXT NOT NULL,                -- ATS-native job id
    url             TEXT NOT NULL,
    company         TEXT,
    title           TEXT,
    location        TEXT,
    raw_html        TEXT,                         -- kept for debugging / re-parsing
    parsed_json     TEXT,                         -- structured JD (Qwen output)
    discovered_at   TEXT NOT NULL,                -- ISO 8601
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_job_source ON job(source);
CREATE INDEX IF NOT EXISTS idx_job_company ON job(company);

CREATE TABLE IF NOT EXISTS application (
    id              TEXT PRIMARY KEY,             -- hash(job_id, candidate_id)
    job_id          TEXT NOT NULL REFERENCES job(id),
    state           TEXT NOT NULL,                -- see ApplicationState enum
    fit_score       REAL,                         -- Qwen 0-100
    review_score    REAL,                         -- Claude 0-100
    review_notes    TEXT,                         -- Claude rationale
    cover_letter    TEXT,                         -- Claude draft
    custom_answers  TEXT,                         -- JSON: {question_text: answer}
    submitted_at    TEXT,
    error           TEXT,                         -- last error if state=errored
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_application_state ON application(state);
CREATE INDEX IF NOT EXISTS idx_application_job ON application(job_id);

-- Selector-learning store. When the deterministic adapter fails on an unknown
-- form field, Claude vision identifies it and we cache the resolution here.
CREATE TABLE IF NOT EXISTS form_field_resolution (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ats             TEXT NOT NULL,                -- 'workday' | 'icims' | ...
    url_pattern     TEXT NOT NULL,                -- e.g. host + path prefix
    field_label     TEXT NOT NULL,                -- "Years of experience" etc.
    selector_kind   TEXT NOT NULL,                -- 'css' | 'xpath' | 'role'
    selector_value  TEXT NOT NULL,
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    last_used_at    TEXT,
    UNIQUE (ats, url_pattern, field_label)
);

-- Cost / observability log. Every LLM call.
CREATE TABLE IF NOT EXISTS llm_call (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id  TEXT,                         -- nullable; some calls aren't per-app
    stage           TEXT NOT NULL,                -- 'parse' | 'filter' | 'review' | 'tailor' | 'vision'
    tier            TEXT NOT NULL,                -- 'qwen' | 'claude_fast' | 'claude_smart'
    model           TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_create_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    called_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_call_stage ON llm_call(stage);
CREATE INDEX IF NOT EXISTS idx_llm_call_called_at ON llm_call(called_at);
