# Changelog

All notable changes to `llm-lab` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/) once v1.0.0 is cut.

> **Pre-1.0 note (≤ v0.x):** Until v1.0.0 is released, minor versions (0.Y.0) may
> contain breaking changes. Patch versions (0.Y.Z) are backwards-compatible.
>
> **Planned cut for the governance + observability push:** `v0.9.0`. M1 (governance)
> and M2 (observability) ship together as a single tagged release so the
> public surface of the project — first impressions for an outside visitor —
> is consistent from day one.

---

## [Unreleased]

### Added (M4 — growth foundation)

- **Content calendar** (`docs/CONTENT-CALENDAR.md`): month-by-month
  plan for 12 blog posts, 3 talk tracks, and 4 distribution channels.
  Targets 1 piece per month — sustainable for a single maintainer.
- **Blog post drafts** (`docs/blog-posts/01-04-*.md`): skeletons
  for the 4 most important posts, ready to flesh out and publish.
  Hooks are concrete, topics are persona-targeted (see PERSONAS.md).
- **Talk proposal + meetup list** (`docs/talks/`): 200-word CFP
  abstract for a regional conference talk, and a tiered list of
  ~100 local meetups + 30+ regional conferences + 8 ML/security-
  specific conferences across NA / EU / APAC.
- **Buyer personas** (`docs/PERSONAS.md`): 3 personas (Alice the
  ML engineer, Bob the security architect, Carol the consultant)
  with pain points, decision authority, where to find them, and
  what "good" looks like for each. Explicitly out-of-scope
  personas documented too.
- **Outreach templates** (`docs/OUTREACH-TEMPLATES.md`): 4
  persona-tuned cold messages, plus the "no reply" follow-up
  guidance, plus a private tracking spreadsheet spec.
- **Adopters** (`ADOPTERS.md`): added a "Pilot users" section
  separate from "Listed adopters" — teams in the evaluation
  phase are more likely to want a low-friction entry than a
  public endorsement.
- **New benchmark fault scenario** `first_time_poc`: cold-start
  to first verified report must complete in < 5 s. This is the
  metric that matters for "30-minute POC lands a usable first
  report" — the persona-level adoption test.

### Fixed (v0.9.1 — CodeQL follow-up)

CodeQL ran against `v0.9.0` immediately after release and surfaced 7
alerts. v0.9.1 addresses them:

- **CI workflow permissions** (`.github/workflows/test.yml`): added
  `permissions: contents: read` at the workflow level so jobs run with
  the minimal default token scope. Fixes 4 × Medium
  "Workflow does not contain permissions" alerts.
- **`planner/engine.py` CodeQL suppressions**: added
  `# codeql[py/path-injection]` comments on the 3 lines flagged.
  **These are false positives** — the upstream `_safe_template_path`
  regex-validates `template_id` against `^[A-Za-z0-9_-]{1,64}$` AND
  verifies the resolved path stays within the template store via
  `os.path.commonpath`. See THREAT_MODEL.md §S4 and the new module
  docstring on `_safe_template_path`. CodeQL's data-flow analysis
  does not see the upstream sanitiser across function boundaries.

Process lesson: we should run CodeQL before tagging, not after.
v0.9.1 is a one-line security posture fix; the CodeQL feedback loop
is now part of the release checklist.

### Fixed (M3.5 — concurrency + honesty pass)

- **Multi-process / multi-writer concurrency in the audit chain.**
  `db.append_event` now opens a `BEGIN IMMEDIATE` transaction before
  reading `prev_hash` and holds it through the `INSERT` and
  `row_hash` `UPDATE`. Combined with `PRAGMA busy_timeout=5000`, this
  serialises concurrent writers across processes; a writer that
  cannot acquire the lock within 5 s raises a `sqlite3.OperationalError`.
  Regression test: `test_concurrent_appends_produce_valid_chain` runs
  4 concurrent `append_event` tasks × 25 rows and asserts
  `verify_log` returns `ok=True`. (Without the fix, the chain would
  break at the first concurrent writer — verified by temporarily
  reverting the fix in M3.5 development.)
- **CI SBOM command fix.** `cyclonedx-py` is invoked as
  `python -m cyclonedx_py environment` (not a direct binary), with
  the right flag names (`--output-file`, `--spec-version 1.6`). The
  previous invocation would have failed in CI on first run.
- **Known limitations documented.** ADR-0006 now lists 7 honest
  limitations (multi-host, platform sensitivity, timestamp trust,
  tamper-rewrite, O(N) verify, legacy genesis, etc.).
  `README.md` and `THREAT_MODEL.md` §P3 surface the most important
  ones to operators.
- **Optional `[sbom]` extra** in `pyproject.toml`. `pip install
  -e ".[sbom]"` pulls `cyclonedx-bom` for local SBOM generation.
- **`.gitignore`** now excludes `sbom.cdx.json` (build artifact, not
  source).
- **New benchmark fault scenario** `audit_chain_concurrent_writers` —
  same race-condition coverage as the unit test, but in the
  benchmark harness.

### Added (audit chain — M3)

- **Tamper-evident event log** (ADR-0006, implemented). `event_log` gains two
  columns: `prev_hash` (the previous row's hash) and `row_hash`
  (`sha256(prev_hash || canonical_json(row_content))`). `db.append_event`
  reads the prior row's hash and chains; `db.verify_log()` walks the
  chain and reports the first tamper; `llm-lab verify` exposes this to
  operators.
- **`llm-lab verify` CLI subcommand.** Walks the chain, exits 0 on
  success / 1 on the first broken row, with `--json` output for tooling.
- **Backwards-compatible:** pre-M3 rows (NULL `prev_hash`/`row_hash`)
  are accepted as legacy genesis rows. `verify_log` reports
  `legacy_genesis_count` so operators know what fraction of the chain
  is post-M3.
- **New tests:** `tests/test_audit_chain.py` (11 tests covering
  round-trip, tamper detection, deletion detection, CLI surface,
  schema migration, idempotent `init_db`).
- **New fault scenarios** in `benchmarks/self_bench.py --mode fault`:
  `audit_chain_clean` (verifies a fresh log) and `audit_chain_tamper`
  (mutates a row, asserts `verify_log` catches it).
- **CycloneDX SBOM auto-generation** in CI (ADR-0009). New `sbom` job
  produces `sbom.cdx.json` per push / PR.

### Added (governance / infrastructure — no runtime behavior change)

- **Governance & docs**: `CHANGELOG.md`, `GOVERNANCE.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `THREAT_MODEL.md`, `CODEOWNERS`, `ADOPTERS.md`, `.github/ISSUE_TEMPLATE/`.
- **Architecture decision records**: `docs/adr/0001`–`0003` and `0006`–`0008` (sync runner,
  two LLM paths, `_sha16` truncation, audit-trail upgrade, structured logging, Prometheus metrics).
- **Benchmark harness**: `benchmarks/self_bench.py` with three modes — `smoke` (CI-safe),
  `perf` (latency / throughput), `fault` (graceful-degradation tests).
- **Self-benchmark report**: `benchmarks/v1-results.json` — first reproducible
  performance + reliability numbers, generated against an offline stub.

### Added (observability — M2)

- **`llm_lab/observability.py`**: structured JSON logging via stdlib
  (zero new deps), trace-id propagation via `ContextVar`, in-process
  Prometheus metrics store with bespoke text-exposition renderer.
- **`GET /metrics` endpoint**: returns Prometheus exposition format.
  Unauthenticated (documented in `THREAT_MODEL.md`); label cardinality
  bounded via path-template collapse (`/result/abc123...` → `/result/:id`).
- **HTTP middleware**: every request gets a trace id (or honours an
  inbound `x-trace-id` header); middleware records request count +
  latency into the metrics store and echoes the id back via response
  header.
- **`worker.call_llm` instrumentation**: each call records provider /
  model / outcome, prompt + completion tokens, and wall-clock latency
  into the metrics store, and emits one structured log line per call.
- **Contract tests**: `tests/test_llm_contract.py` pins the shared
  response shape across `worker` and `promptfoo_provider` so the two
  paths can't silently diverge.

### Changed

- **CI workflow (`test.yml`)**: now runs on `ubuntu-latest` **and** `macos-latest` across
  Python 3.10 / 3.11 / 3.12; adds a separate `security` job (Bandit + pip-audit in an
  isolated venv) and a `benchmark-smoke` job. Concurrency group cancels in-flight runs on the same ref.
- **`llm_lab/worker.py`**: extracted `build_openai_client(base_url, api_key)` as the single
  source of truth for OpenAI-compatible client construction. `call_llm` now wraps the
  dispatch in metrics + structured-log emission.
- **`llm_lab/promptfoo_provider.py`**: client construction now delegates to
  `worker.build_openai_client` — single source of truth, no behaviour change.

### Documentation

- README rewritten with a 30-second hook, 5-minute quickstart, badges, and
  benchmark callout.
- `docs/ARCHITECTURE.md` updated: §4.3 "运维注意" corrected (anthropic / google.generativeai
  are already lazy-imported with graceful fallback); §9 weakness list refreshed
  (WAL + busy_timeout + shared client builder now in place).
- ADR-0007 (structured logging) + ADR-0008 (Prometheus metrics) document
  the M2 observability decisions.

### Roadmap (tracked, not yet shipped)

See `docs/adr/0006-audit-trail-integrity.md` for the planned upgrade to a
tamper-evident audit log. This is the next major feature after M2 observability.

---

## [0.1.0] — initial internal release

Initial codebase as audited. Includes:

- FastAPI service (`llm_lab/main.py`) with multi-tenant isolation, authentication on
  8 endpoints, security headers middleware, and `/health` exemptions.
- CLI (`llm_lab/cli.py`, Typer) with `run / compare / serve / history / export / report / watch / diff`.
- `llm_lab/runner.py` — sync planner → LLM → verifier pipeline, ThreadPoolExecutor concurrency.
- `llm_lab/worker.py` — multi-provider LLM caller (OpenAI / Anthropic / Gemini / Ollama /
  vLLM / llama.cpp / TGI / LocalAI) with lazy SDK imports and graceful fallback.
- `llm_lab/promptfoo_provider.py` — parallel LLM path mirroring promptfoo semantics
  (YAML config, SQLite cache, exponential-backoff retry).
- `llm_lab/planner/engine.py` — template engine with `_TEMPLATE_ID_RE` + `_safe_template_path`
  to block path-traversal.
- `llm_lab/verifier.py` — `structural` / `keyword` / `deepeval` (opt-in) verifiers.
- `llm_lab/tracer.py` — Langfuse integration with SQLite fallback.
- `llm_lab/db.py` — SQLite + WAL + `busy_timeout=5000`, `event_log` schema with
  `_sha16` input/output hashes and a `verdict` column.
- `llm_lab/pricing.py` — `_PRICE_PER_1K` per-model costs; local providers always $0.
- `llm_lab/export.py` — JSON / CSV / XLSX / HTML exporters; all HTML output escaped
  via `_esc()` (XSS-safe).

Security audit fixes shipped prior to this changelog (commits
`f31d0e4` `be72f17` `2e47df1` `1456178` `a709577`):

1. HTML report XSS → `_esc()` everywhere.
2. `intent_id` regex + `validate_path_param` against path traversal.
3. Constant-time API key comparison (`hmac.compare_digest`).
4. Auth on 8 endpoints; `/health` and `/promptfoo/health` exempt.
5. Security response headers middleware.
6. Template path-traversal hard stop in `planner._safe_template_path`.
7. `event_log.verdict` column added via idempotent `ALTER TABLE`.
8. Output artefact hashes written to `event_log` for audit traceability.

**Test status:** 349 passed / 1 skipped.
**Static checks:** `ruff` (incl. `S` rules) + `mypy` clean.