# Changelog

All notable changes to `llm-lab` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/) once v1.0.0 is cut.

> **Pre-1.0 note (≤ v0.x):** Until v1.0.0 is released, minor versions (0.Y.0) may
> contain breaking changes. Patch versions (0.Y.Z) are backwards-compatible.

> **Release process note:** v0.9.0 → v0.9.6 are kept in git history but were
> not CI-green; the first CI-green release is **v0.9.7**. The earlier
> commits document the iteration that got us there. Releases page on
> GitHub shows v0.9.7 as Latest.

---

## [0.9.7] — 2026-07-13 — first CI-green release

First release where every CI job is green end-to-end on a fresh
GitHub Actions runner (no manual cleanup, no local-only verification).

### Highlights

- 7 governance docs (CHANGELOG, GOVERNANCE, CONTRIBUTING, SECURITY, THREAT_MODEL, CODEOWNERS, ADOPTERS)
- 9 ADRs (sync runner, two LLM paths, sha16, audit chain, structured logging, Prometheus metrics, SBOM, …)
- `llm-lab verify` CLI subcommand — tamper-evident event-log hash chain
- `GET /metrics` endpoint — Prometheus exposition format
- 381 tests pass / 1 skipped / 9/9 benchmark fault scenarios
- ruff + mypy + bandit clean
- macOS + 3 Python versions CI matrix
- CycloneDX SBOM per release

### Added

- **Audit chain (ADR-0006).** `event_log` gains `prev_hash` and `row_hash`
  columns; each new row chains to the previous via
  `sha256(prev_hash || canonical_json(row_content))`. `db.verify_log()`
  walks the chain in `id` order and reports the first tamper.
- **`llm-lab verify` CLI subcommand.** Exits 0 on success / 1 on the
  first break. Supports `--json` and `--limit`. False-positive CodeQL
  alerts on `planner/engine.py` are documented with the security
  contract in `_safe_template_path`'s docstring (regex-validated
  `template_id` + `os.path.commonpath` containment check).
- **Observability (`observability.py`).** Structured JSON logging via
  stdlib, `ContextVar`-based trace-id propagation, in-process
  Prometheus metrics store with bespoke text-exposition renderer.
- **HTTP middleware.** Sets / honours `x-trace-id`, records
  `llm_lab_requests_total` and `llm_lab_request_duration_seconds` with
  path-template cardinality collapse (`/result/abc123...` → `/result/:id`).
- **`worker.call_llm` instrumentation.** Per-call metrics
  (`llm_lab_llm_calls_total`, `llm_lab_tokens_total`,
  `llm_lab_llm_call_duration_seconds`) + one structured log line per call.
- **Single OpenAI client source.** `worker.build_openai_client` is the
  single source of truth; `promptfoo_provider` delegates to it.
- **CycloneDX SBOM (ADR-0009).** New `sbom` CI job produces
  `sbom.cdx.json` per push / PR, attached as a release artifact.
- **CI matrix.** `ubuntu-latest` + `macos-latest` × Python 3.10 / 3.11
  / 3.12; separate `security` and `benchmark-smoke` jobs; concurrency-group
  cancellation of in-flight runs on the same ref.
- **Content + growth scaffolding.** `docs/CONTENT-CALENDAR.md` (12-month
  plan), `docs/PERSONAS.md` (3 buyer personas), `docs/OUTREACH-TEMPLATES.md`
  (4 cold messages), `docs/blog-posts/{01-04}-*.md` (4 skeletons),
  `docs/talks/{2026-local-ml-meetup,cfp-abstract,meetup-list-2026}.md`,
  `docs/FINDING-FIRST-USER.md`, `docs/NEXT-STEPS.md`.

### Fixed

- **Multi-process / multi-writer concurrency in the audit chain.**
  `db.append_event` wraps the read-insert-update in `BEGIN IMMEDIATE`
  + `PRAGMA busy_timeout=5000`, so concurrent writers across processes
  serialise cleanly. Regression test
  (`test_concurrent_appends_produce_valid_chain`) runs 4 writers ×
  25 rows and asserts `verify_log` reports `ok=True`. (Without the
  fix the chain would break at the first concurrent writer — verified
  by temporarily reverting the fix during development.)
- **Workflow `permissions` posture.** Workflow-level
  `permissions: contents: read`; codecov step additionally grants
  `pull-requests: read` for PR comment posting. Addresses the 4 ×
  Medium "Workflow does not contain permissions" alerts from CodeQL.
- **CI workflow syntax.** `permissions:` is only valid at workflow
  or job level (not on a step); the `pip-audit` step was indented
  too deep and is now a sibling of `bandit`. Verified locally with
  `yaml.safe_load` before every push from v0.9.3 onward.
- **CI test flakes.** `test_serve_accepts_flags`,
  `test_watch_command_help` previously substring-checked option names
  (`"--port" in result.stdout`); rich's help renderer wraps long
  flag names across lines on narrower terminal widths (CI vs local
  dev). Replaced with stable help-text substring checks
  (`"Server port" in result.stdout`).
- **`test_build_client_openai_defaults` env var.** The test was
  setting `OPENAI_API_KEY`, but `_build_client` reads `LLM_API_KEY`
  (the dispatcher env var). Original test only passed locally
  because a previous test leaked `LLM_API_KEY=ollama` into the env.
  Fix: set the right env var, rewrite with `monkeypatch` for proper
  isolation.
- **pip-audit in CI.** `pip install -e .` made `llm-lab` an
  editable install; pip-audit under `--strict` refused to audit an
  editable without a PyPI release. Fix: don't install `llm-lab` in
  the security job at all; extract the runtime deps from
  `pyproject.toml` into a requirements-style file and audit that.
- **bandit B608 vs ruff S608.** Added `# noqa: B608, S608` on the
  two f-string SQL queries; bandit doesn't honour ruff's S608 ID.

### Security

- The audit chain is a **detection** mechanism, not a **prevention**
  one. `THREAT_MODEL.md §P3` and the docstring on
  `_safe_template_path` document what it does and does not defend
  against. Multi-host deployments are unsupported; an attacker with
  file-system write access can rewrite the chain forward. For
  high-stakes deployments ship a periodic hash snapshot of the
  SQLite file to a write-once store (S3 Object Lock, immutable
  syslog, etc.) and compare.

### Removed

- **v0.9.0 → v0.9.6 release tags** were deleted from the Releases
  page after v0.9.7 was published. Those git commits remain in
  history for traceability. None of those tags passed CI end-to-end
  on first push; v0.9.7 is the first that did.

---

## [Unreleased]

Nothing yet. The next planned change is the v1.0.0 scope:
- Cut the 0.x → 1.0 transition (commit to SemVer)
- Third-party security audit (budget pending)
- LF AI & Data Sandbox application (if traction supports it)

---

## [0.1.0] — initial internal release

Initial codebase as audited. Includes:

- FastAPI service (`llm_lab/main.py`) with multi-tenant isolation,
  authentication on 8 endpoints, security headers middleware, and
  `/health` exemptions.
- CLI (`llm_lab/cli.py`, Typer) with
  `run / compare / serve / history / export / report / watch / diff`.
- `llm_lab/runner.py` — sync planner → LLM → verifier pipeline,
  ThreadPoolExecutor concurrency.
- `llm_lab/worker.py` — multi-provider LLM caller (OpenAI / Anthropic /
  Gemini / Ollama / vLLM / llama.cpp / TGI / LocalAI) with lazy SDK
  imports and graceful fallback.
- `llm_lab/promptfoo_provider.py` — parallel LLM path mirroring
  promptfoo semantics (YAML config, SQLite cache, exponential-backoff
  retry).
- `llm_lab/planner/engine.py` — template engine with `_TEMPLATE_ID_RE`
  + `_safe_template_path` to block path-traversal.
- `llm_lab/verifier.py` — `structural` / `keyword` / `deepeval` (opt-in)
  verifiers.
- `llm_lab/tracer.py` — Langfuse integration with SQLite fallback.
- `llm_lab/db.py` — SQLite + WAL + `busy_timeout=5000`, `event_log`
  schema with `_sha16` input/output hashes and a `verdict` column.
- `llm_lab/pricing.py` — `_PRICE_PER_1K` per-model costs; local
  providers always $0.
- `llm_lab/export.py` — JSON / CSV / XLSX / HTML exporters; all HTML
  output escaped via `_esc()` (XSS-safe).

Security audit fixes shipped prior to this changelog (commits
`f31d0e4` `be72f17` `2e47df1` `1456178` `a709577`):

1. HTML report XSS → `_esc()` everywhere.
2. `intent_id` regex + `validate_path_param` against path traversal.
3. Constant-time API key comparison (`hmac.compare_digest`).
4. Auth on 8 endpoints; `/health` and `/promptfoo/health` exempt.
5. Security response headers middleware.
6. Template path-traversal hard stop in `planner._safe_template_path`.
7. `event_log.verdict` column added via idempotent `ALTER TABLE`.
8. Output artefact hashes written to `event_log` for audit
   traceability.

**Test status:** 349 passed / 1 skipped.
**Static checks:** `ruff` (incl. `S` rules) + `mypy` clean.
