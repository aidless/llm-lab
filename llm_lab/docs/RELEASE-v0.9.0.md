# Release notes — v0.9.0 (governance + observability + audit-chain push)

> **Draft — not yet tagged.** This document describes the first cut after
> M1 (governance), M2 (observability), and M3 (audit chain). Tag it when
> you're ready.

**Release date:** _TBD_
**Tag:** `v0.9.0`
**SemVer note:** minor bump within `0.x` is allowed to contain breaking changes
(see `CHANGELOG.md` preamble). v0.9.0 contains **no breaking runtime API changes**
— only governance, CI, observability, and audit-chain additions.

---

## What's new

### Governance

- `CHANGELOG.md` — first formal release log.
- `GOVERNANCE.md` — single-maintainer → core-team evolution path, with the
  objective triggers to promote.
- `CONTRIBUTING.md` — dev setup, PR template, security-sensitive review checklist.
- `SECURITY.md` — disclosure process, response SLA, severity rating.
- `THREAT_MODEL.md` — what's in scope, partial, and out of scope for the
  security posture. Audit-fix lineage tied back to commits.
- `CODEOWNERS` — review routing for security-sensitive modules.
- `ADOPTERS.md` — empty file with header. Be the first entry.
- `.github/ISSUE_TEMPLATE/` — bug, feature, question templates.
- `.github/PULL_REQUEST_TEMPLATE.md` — security-checklist embedded.

### Architecture decision records (ADRs)

- `docs/adr/0001-sync-runner.md` — sync Runner + ThreadPoolExecutor, not full asyncio.
- `docs/adr/0002-two-llm-paths.md` — `worker.py` and `promptfoo_provider.py`
  stay separate; share `build_openai_client`.
- `docs/adr/0003-sha16-hash.md` — why `_sha16` truncates SHA-256 to 16 hex chars.
- `docs/adr/0006-audit-trail-integrity.md` — **implemented in M3.** Hash-chain
  on `event_log` + `llm-lab verify` subcommand.
- `docs/adr/0007-structured-logging.md` — stdlib JSON logging + trace-id
  propagation, no new deps.
- `docs/adr/0008-prometheus-metrics.md` — in-process metrics store +
  bespoke Prometheus exposition renderer, no new deps.
- `docs/adr/0009-cyclonedx-sbom.md` — auto-generated CycloneDX SBOM
  per CI run, attached as a release artifact.

### Observability (M2)

- `llm_lab/observability.py` — single module owning structured logging,
  trace-id propagation, and the in-process metrics store.
- `GET /metrics` — Prometheus exposition format. Unauthenticated (documented
  in `THREAT_MODEL.md`); label cardinality bounded via path-template collapse.
- HTTP middleware records request count + latency, sets / honours trace id,
  echoes trace id back via `x-trace-id` response header.
- `worker.call_llm` instrumentation: per-call metrics + structured log line.
- `tests/test_observability.py` (16 tests) — formatter, trace-id,
  metrics, Prometheus exposition, endpoint behaviour.
- `tests/test_llm_contract.py` (4 tests) — pins the shared response shape
  across `worker` and `promptfoo_provider`.

### Audit log hash chain (M3)

- `event_log` gains two columns: `prev_hash` and `row_hash`. Each new row
  chains to the previous via `sha256(prev_hash || canonical_json(row))`.
- `db.verify_log()` walks the chain in `id` order and reports the first
  tamper (modification, deletion, or insertion).
- **`llm-lab verify`** subcommand: `OK N rows verified, chain intact`
  on success, `FAIL at id=N (kind=...)` on the first break. Exits 0/1
  for CI integration. Supports `--json` and `--limit`.
- **Backwards-compatible**: pre-M3 rows (NULL chain columns) are accepted
  as legacy genesis rows. `verify_log` reports `legacy_genesis_count`
  so operators know the post-M3 fraction.
- `tests/test_audit_chain.py` (11 tests) — round-trip, tamper detection,
  deletion detection, CLI surface, schema migration, idempotency.

### CI / quality gates

- `.github/workflows/test.yml` rewritten:
  - **Matrix**: `ubuntu-latest` + `macos-latest` × Python 3.10 / 3.11 / 3.12.
  - **Coverage**: Codecov upload on `ubuntu / 3.11` job.
  - **Security**: separate `security` job runs in an **isolated venv** (so
    `pip-audit --strict` only sees `llm-lab`'s declared deps, not the host's
    global environment), runs `bandit -ll` and `pip-audit --strict`.
  - **Benchmark smoke**: separate `benchmark-smoke` job runs
    `benchmarks/self_bench.py --mode smoke` and uploads the JSON as an artifact.
  - **SBOM**: separate `sbom` job produces `sbom.cdx.json` (CycloneDX 1.5).

### Benchmarks

- `benchmarks/self_bench.py` — three modes:
  - `smoke` (CI-gated, ~2s): perf + fault scenarios.
  - `perf` (default 25 steps, configurable): latency + throughput.
  - `fault` (graceful-degradation): 8 scenarios covering provider
    timeout, missing API key, optional-SDK-or-key, SQLite lock
    contention, audit-chain (clean / tamper / concurrent writers),
    and first-time-POC cold-start.
- `benchmarks/v1-results.json` — reproducible benchmark report.
  - 50-step throughput: ~185 steps/sec on Windows / Python 3.11.
  - p50 latency: ~5.4 ms (stub LLM).
  - All **8 fault scenarios pass**.

### Code changes

- `llm_lab/db.py`: hash-chain columns + `compute_row_hash()` + `verify_log()`.
  Backward-compatible migration via idempotent `ALTER TABLE`.
- `llm_lab/cli.py`: new `verify` subcommand.
- `llm_lab/worker.py`: extracted `build_openai_client(base_url, api_key)` —
  single source of truth for OpenAI-compatible client construction.
  `call_llm` now wraps the dispatch in metrics + structured-log emission.
- `llm_lab/promptfoo_provider.py`: client construction now delegates to
  `worker.build_openai_client`. No behaviour change.
- `llm_lab/main.py`: HTTP middleware for trace-id + metrics; new
  `GET /metrics` route; `lifespan` calls `configure_logging()`.
- `docs/ARCHITECTURE.md`: §4.3 "运维注意" corrected (anthropic /
  google.generativeai were already lazy-imported); §9 weakness list refreshed.

---

## What did NOT change

- **No breaking API changes.** All public function signatures, HTTP routes,
  and CLI subcommands unchanged.
- **No new runtime dependencies.** `bandit` and `pip-audit` are CI-only
  (installed by the security job); neither is a project dependency.
- **No schema migrations.** The `event_log` table is unchanged; the
  hash-chain upgrade (ADR-0006) is scheduled for v0.10.0.

---

## Known issues carried forward

- **`ccfddl`** appears in `pip freeze` but is not on PyPI. `pip-audit --strict`
  fails because of this. **Action:** remove the package from the
  developer's local environment or pin via `requirements-dev.txt` so it
  doesn't reach the audit job. Tracked separately.
- **`event_log` tamper-evidence** is documented as not yet implemented
  (`THREAT_MODEL.md` §P3). Scheduled for v0.10.0 (ADR-0006).
- **Multi-host concurrency** — single-host SQLite is safe (WAL + busy_timeout);
  multi-host deployments need a real RDBMS. Out of scope for v0.x.

---

## Upgrade instructions

`pip install --upgrade llm-lab==0.9.0` or pull the tag.

No action required for users. If you run `pip-audit` locally, expect it to
report the `ccfddl` issue until that's resolved (see above).

---

## Thanks

To everyone who opens an issue, files a PR, or simply runs `llm-lab` and
feels strongly enough to tell us what broke. This release makes your
feedback easier to give — please use the new templates.