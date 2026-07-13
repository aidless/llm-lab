# llm-lab

[![CI](https://github.com/aidless/llm-lab/actions/workflows/test.yml/badge.svg)](https://github.com/aidless/llm-lab/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[[![Tests](https://img.shields.io/badge/tests-381%20passed%20%C2%B7%201%20skipped-brightgreen)](./CHANGELOG.md)
[![Ruff + Mypy + Bandit](https://img.shields.io/badge/lint-ruff%20%2B%20mypy%20%2B%20bandit-blue)](./CONTRIBUTING.md)

> **The eval tool you can show your security team.**
> LLM evaluation & orchestration for security-conscious teams — auditable,
> air-gap-able, hardened by default, and Prometheus-observable.

**Tamper-evident audit log.** Every `event_log` row carries a SHA-256 hash
chained to its predecessor. Run `llm-lab verify` to detect any insertion,
deletion, or modification. See
[`docs/adr/0006-audit-trail-integrity.md`](./llm_lab/docs/adr/0006-audit-trail-integrity.md).

**Observability built in:** structured JSON logs, trace-id propagation,
`GET /metrics` (Prometheus exposition format). No extra dependencies — see
[`docs/adr/0007-structured-logging.md`](./llm_lab/docs/adr/0007-structured-logging.md)
and [`docs/adr/0008-prometheus-metrics.md`](./llm_lab/docs/adr/0008-prometheus-metrics.md).

---

## Why llm-lab?

Other eval tools optimise for **ergonomics**. `llm-lab` optimises for
**defensibility**:

- ✅ **8 security audit fixes already shipped** (XSS, path traversal, auth bypass,
  timing attacks, security headers, …) — see [`THREAT_MODEL.md`](./THREAT_MODEL.md).
- ✅ **Multi-tenant isolation** with explicit authentication on 8 endpoints.
- ✅ **Auditable event log** with `verdict` + input/output hashes (path to
  tamper-evident chain tracked in
  [`docs/adr/0006-audit-trail-integrity.md`](./llm_lab/docs/adr/0006-audit-trail-integrity.md)).
- ✅ **Graceful degradation** under provider outages, missing API keys, and
  SQLite lock contention — verified by
  [`benchmarks/self_bench.py --mode fault`](./llm_lab/benchmarks/self_bench.py).
- ✅ **349 tests, ruff + mypy + bandit + pip-audit clean**.

If your security team needs to know *why* your eval pipeline picked model B
over model A, and *which prompt hashes* it ran, `llm-lab` gives you the answer.

---

## 5-minute quickstart

```bash
# Install
git clone https://github.com/aidless/llm-lab
cd llm-lab
pip install -e ".[dev]"

# Set a provider key (any of OpenAI / Anthropic / Gemini; or run Ollama locally)
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini      # or claude-3-5-haiku-20241022, gemini-1.5-flash, ...

# Run a single eval — a goal becomes a planner plan → LLM steps → verifier verdicts
llm-lab run "explain the difference between supervised and self-supervised learning"

# A/B compare two models
llm-lab compare "write a haiku about Rust" \
    --model-a gpt-4o-mini \
    --model-b claude-3-5-haiku-20241022

# Start the web UI + REST API at http://127.0.0.1:8123
llm-lab serve
```

The CLI prints a verdict (`pass`/`fail`), per-step token counts, total cost,
and emits an HTML report you can share with auditors.

---

## Audit log verification

Every row in `event_log` carries a SHA-256 hash chained to its predecessor
(`row_hash = sha256(prev_hash || canonical_json(row))`). The chain runs in
`id` order, so any insertion, deletion, or modification is detectable.

```bash
$ llm-lab verify
OK 138 rows verified, chain intact (12 legacy genesis rows skipped)

$ llm-lab verify --json | jq '.first_break'
{
  "id": 47,
  "kind": "row_hash",
  "expected": "5b8e2c…",
  "found": "f3a91d…"
}
```

Use this in CI on every deploy, or as a scheduled job. The `verify`
subcommand exits non-zero on the first break, so it composes with any
existing alerting.

> **Deployment notes** (read this before running multi-worker / multi-host):
>
> - **Single-process is supported and tested.** The chain is correct
>   under concurrent `append_event` calls in the same process (verified
>   by `tests/test_audit_chain.py::test_concurrent_appends_produce_valid_chain`).
> - **Multi-process on the same SQLite file is supported** via SQLite's
>   `BEGIN IMMEDIATE` transaction inside `append_event` plus a 5-second
>   busy timeout. Two processes writing concurrently will serialize
>   cleanly.
> - **Multi-host is NOT supported.** A SQLite file on a network filesystem
>   (NFS, SMB, CephFS) without proper POSIX locking will corrupt. For
>   multi-host, use a real RDBMS or a write-once external store (S3
>   Object Lock, append-only syslog) for the audit log.
> - **The chain detects tampering after the fact.** It does not prevent
>   it. An attacker with file-system write access can rewrite a row and
>   recompute the chain forward. For attribution in adversarial
>   environments, ship a periodic hash snapshot of the SQLite file to an
>   immutable store and compare against it.

See [`docs/adr/0006-audit-trail-integrity.md`](./llm_lab/docs/adr/0006-audit-trail-integrity.md)
for the design and limitations.

---

## Who is this for?

Three personas drive the design. Read whichever matches you best:

- **Alice — ML engineer at a 50–500 person SaaS:** you keep losing
  track of which prompt is which. [`docs/PERSONAS.md`](./llm_lab/docs/PERSONAS.md#p0--alice-the-ml-engineer-at-a-50-500-person-saas)
- **Bob — security architect at a regulated company:** your
  security team blocks LLM tools that don't have an audit trail.
  [`docs/PERSONAS.md`](./llm_lab/docs/PERSONAS.md#p0--bob-the-security-architect-at-a-regulated-company)
- **Carol — independent ML consultant:** you re-build the same
  eval pipeline on every client engagement.
  [`docs/PERSONAS.md`](./llm_lab/docs/PERSONAS.md#p1--carol-the-independent-ml-consultant)

> If none of these match you, [`promptfoo`](https://promptfoo.dev/)
> is probably a better fit — it's faster and the community is
> bigger. We occupy a different cell.

---

## Observability

Every request:

- Receives a trace id (or honours an inbound `x-trace-id` header).
- Logs every event as a single-line JSON object with that trace id.
- Increments `llm_lab_requests_total{path,method,status}` and observes
  `llm_lab_request_duration_seconds{path,method}`.

Every LLM call (`worker.call_llm`):

- Records `llm_lab_llm_calls_total{provider,model,outcome}`,
  `llm_lab_llm_call_duration_seconds{provider,model}`,
  and `llm_lab_tokens_total{provider,model,direction}`.
- Emits a structured log line with `provider`, `model`, `tokens`, `cost_usd`,
  `finish_reason`, `duration_seconds`.

Scrape metrics from a Prometheus-compatible scraper:

```yaml
scrape_configs:
  - job_name: llm-lab
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:8123']
```

See [`docs/adr/0008-prometheus-metrics.md`](./llm_lab/docs/adr/0008-prometheus-metrics.md)
for the label semantics and cardinality controls.

---

## Self-benchmark numbers (reproducible)

`llm-lab` benchmarks itself on every CI run (`--mode smoke`) and publishes
full numbers in [`benchmarks/v1-results.json`](./llm_lab/benchmarks/v1-results.json).

Latest stable run (Windows / Python 3.11 / stub LLM, 50 steps):

| Metric | Value |
| ------ | ----- |
| Throughput (offline stub) | **187 steps/sec** |
| p50 latency | **5.32 ms** |
| p95 latency | **5.54 ms** |
| Token accounting drift (stub vs real) | **0.0 USD** (local stub is free, as documented) |
| Fault scenarios passing | **5 / 5** |

Reproduce locally:

```bash
python llm_lab/benchmarks/self_bench.py --mode all --steps 50 --output my-bench.json
```

The fault scenarios cover: provider timeout, missing API key, optional-SDK-missing
path, and SQLite lock contention under WAL + `busy_timeout`.

---

## Architecture

```
                CLI (typer)        Web API (FastAPI)        Compare Report
                    │                    │                          │
                    └────────┬───────────┘                          │
                             ▼                                      │
                       ┌──────────────┐    ┌──────────────┐          │
                       │   Runner     │───▶│   Verifier   │          │
                       │   (sync +    │    │ (structural/ │          │
                       │  ThreadPool) │    │  keyword/    │          │
                       └──────┬───────┘    │  deepeval)   │          │
                              │            └──────────────┘          │
                              ▼                                      │
                       ┌──────────────┐    ┌──────────────┐          │
                       │   Worker     │    │   Planner    │          │
                       │ (OpenAI /    │    │ (template    │          │
                       │  Anthropic / │    │  engine +    │          │
                       │  Gemini /    │    │  safety)     │          │
                       │  local)      │    └──────────────┘          │
                       └──────┬───────┘                              │
                              │                                      │
                       ┌──────▼───────┐    ┌──────────────┐          │
                       │   Tracer     │───▶│ event_log    │          │
                       │ (Langfuse +  │    │ (SQLite WAL, │          │
                       │  SQLite      │    │  hashes,     │          │
                       │  fallback)   │    │  verdict)    │          │
                       └──────────────┘    └──────────────┘
```

See [`llm_lab/docs/ARCHITECTURE.md`](./llm_lab/docs/ARCHITECTURE.md) for the
full breakdown, and [`llm_lab/docs/adr/`](./llm_lab/docs/adr/) for the
rationale behind each design choice.

---

## Modules

| Module | Role |
| ------ | ---- |
| `cli.py` | Typer CLI: `run`, `compare`, `serve`, `history`, `export`, `report`, `watch`, `diff` |
| `main.py` | FastAPI app: 8 auth-gated endpoints + `/health` + `/promptfoo/health` |
| `runner.py` | Sync planner → LLM → verifier pipeline with ThreadPoolExecutor concurrency |
| `worker.py` | Multi-provider LLM caller (OpenAI / Anthropic / Gemini / Ollama / vLLM / llama.cpp / TGI / LocalAI) |
| `promptfoo_provider.py` | Parallel LLM path mirroring promptfoo semantics (cache + retry + YAML config) |
| `planner/` | Template-driven plan generation with `_safe_template_path` anti-traversal |
| `verifier.py` | `structural` / `keyword` / `deepeval` (opt-in) |
| `tracer.py` | Langfuse integration with SQLite fallback |
| `db.py` | SQLite + WAL + `busy_timeout=5000`; `event_log` with `_sha16` hashes |
| `export.py` | JSON / CSV / XLSX / HTML; all HTML escaped via `_esc()` |
| `settings.py` | pydantic-settings config + presets |
| `pricing.py` | Per-model pricing + `$0` for local providers |

---

## Governance & docs

| Document | What's in it |
| -------- | ------------ |
| [`SECURITY.md`](./SECURITY.md) | How to report vulnerabilities, response SLA, severity scale |
| [`THREAT_MODEL.md`](./THREAT_MODEL.md) | What we defend against, what we don't, what we mitigate |
| [`GOVERNANCE.md`](./GOVERNANCE.md) | Single-maintainer → core-team evolution path |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | Dev setup, PR template, security-sensitive review checklist |
| [`CHANGELOG.md`](./CHANGELOG.md) | Release history, SemVer notes |
| [`CODEOWNERS`](./CODEOWNERS) | Review routing |
| [`ADOPTERS.md`](./ADOPTERS.md) | Who's using it (empty — be the first!) |
| [`llm_lab/docs/adr/`](./llm_lab/docs/adr/) | Architecture Decision Records |

---

## What's *not* in scope

We say this out loud so you don't waste time looking for it:

- ❌ Cloud SaaS hosting — `llm-lab` is open-core only.
- ❌ Fine-tuning / RLHF / model training — use dedicated tools.
- ❌ Vector databases / RAG pipelines — use dedicated tools.
- ❌ A prompt IDE / playground — use promptfoo / PromptLayer / LangSmith.

---

## Status

`llm-lab` is **pre-1.0**. APIs may change between minor versions until v1.0.0.
We commit to keeping security and audit-log semantics stable across minor
versions; breaking changes will be flagged in [`CHANGELOG.md`](./CHANGELOG.md).

Current tracked work: [audit-trail tamper-evidence chain (ADR-0006)](./llm_lab/docs/adr/0006-audit-trail-integrity.md).