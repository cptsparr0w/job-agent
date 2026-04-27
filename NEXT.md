# job-agent — pickup notes

## State (end of session 1)
- Phase 0 skeleton: ✓
- parse stage (QWEN_FAST):  ✓ working, ~13s on 9KB HTML
- score stage (QWEN_FAST):  ✓ working, ~5s avg
- review stage (QWEN_REFINED):  stub
- tailor stage (QWEN_REFINED):  implemented, untested
- discover, fill_form: stubs

## Doctrine
Split work by inference passes, not model capability. Where local model
options give a real performance gradient, also split by capability.

| Tier          | Model                        | Use                       |
|---------------|------------------------------|---------------------------|
| QWEN_FAST     | google/gemma-4-26b-a4b       | parse, filter, score      |
| QWEN_REFINED  | qwen/qwen3-coder-next        | review, tailor (3-pass)   |
| QWEN_VISION   | (unset; Qwen 2.5-VL when needed) | unknown form fields  |

## Bake-off (docs/eval-day1.txt)
parse  gemma   1 call   13.0s  727 tokens
score  gemma   5 calls   5.0s  207 tokens
score  coder   1 call   20.8s   62 tokens

Coder reasoned about tradeoffs (45.5 score, partial credit for domain match).
Gemma applied rules (0.0 score, hard exclude on title).
Latency ~4x for Coder; correct call to keep Gemma on FAST tier.

## Next single move
Implement src/job_agent/stages/review.py — same shape as score.py but with
tier=ModelTier.QWEN_REFINED and three role-specialized system prompts
(drafter, critic, rewriter). Reads SCORED rows, transitions to REVIEWED
or REVIEW_REJECTED based on cutoff in profile['review']['review_cutoff'].

## Resume sequence
cd ~/code/job-agent
source .venv/bin/activate
python -m job_agent.cli status

LM Studio must have BOTH models loaded:
- google/gemma-4-26b-a4b
- qwen/qwen3-coder-next

## Open questions
- Greenhouse boards-api may not serve newer job IDs. seed_one.py falls back
  to scraping HTML — proven working.
- Free-text degeneration on rationale fields needs maxLength in JSON schema.
  Same pattern will apply to review/tailor outputs.
- For real job hunting: seed 5+ Senior Security PM postings to validate the
  score stage pushes them forward rather than rejecting everything.

## Bugs survived (notes for the cert writeup)
1. aiosqlite Connection is awaitable AND async-context-manager. Doing both
   double-starts the thread. Fixed with @asynccontextmanager on db.connect().
2. Optional Anthropic dep — llm/__init__.py imported ClaudeClient eagerly;
   removed import. orchestrator.py imports BudgetExceeded; added no-op shim.
3. Gemma 4 token-degeneration in free-text fields: max_tokens raised, schema
   maxLength added, temperature lifted to 0.4. All three together.
