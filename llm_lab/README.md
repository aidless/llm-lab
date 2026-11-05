# llm-lab

**Local-first LLM evaluation workbench.** Plan a task, call a model, verify the
output, and inspect a hash-linked audit trail without sending run data to a
hosted evaluation service.

[![CI](https://github.com/aidless/llm-lab/actions/workflows/test.yml/badge.svg)](https://github.com/aidless/llm-lab/actions/workflows/test.yml)

This directory is the installable package (`pyproject.toml` version 0.9.7).
Python 3.10+. MIT. There is no PyPI release. The repository also holds
unevaluated notes and draft planning; those are not this package. See
[`SCOPE.md`](../SCOPE.md) and [`FAILURE_MODES.md`](../FAILURE_MODES.md).

The CI badge reports [`.github/workflows/test.yml`](../.github/workflows/test.yml).
There is no `codeql.yml` in this repo, so this file does not carry a CodeQL
badge. Collection counts are checked against
[`docs/test-inventory.json`](../docs/test-inventory.json). A green workflow
does not mean a real model was called.

**Tamper-evident audit log.** Every `event_log` row carries a SHA-256 hash
chained to its predecessor. Run `llm-lab verify` to detect any insertion,
deletion, or modification. See
  [`docs/adr/0006-audit-trail-integrity.md`](./docs/adr/0006-audit-trail-integrity.md).

**Observability built in:** structured JSON logs, trace-id propagation,
`GET /metrics` (Prometheus exposition format). No extra dependencies — see
  [`docs/adr/0007-structured-logging.md`](./docs/adr/0007-structured-logging.md)
and [`docs/adr/0008-prometheus-metrics.md`](./docs/adr/0008-prometheus-metrics.md).

---

## Why llm-lab?

Other eval tools optimise for **ergonomics**. `llm-lab` optimises for
**defensibility**:

- The threat model names specific controls (path checks, HTML escaping,
  auth on selected endpoints, security headers) and the cases they do not
  cover. See [`THREAT_MODEL.md`](../THREAT_MODEL.md) and
  [`FAILURE_MODES.md`](../FAILURE_MODES.md).
- SQLite is one file shared by every `intent_id`. That is not tenant
  isolation. [`THREAT_MODEL.md`](../THREAT_MODEL.md) says to run one process
  per tenant if isolation matters.
- `event_log` stores a SHA-256 hash chain. `llm-lab verify` detects an edit
  that does not recompute later hashes. It does not detect a forward
  recompute or a replaced database file. See
  [`docs/adr/0006-audit-trail-integrity.md`](./docs/adr/0006-audit-trail-integrity.md).
- Fault-path coverage for timeouts, missing keys, and SQLite lock waits lives
  in [`benchmarks/self_bench.py`](./benchmarks/self_bench.py). CI runs
  `--mode smoke` only.

No organisation is listed in [`ADOPTERS.md`](../ADOPTERS.md).

---

## Offline path

No API key and no network. CI runs this after installing the package.

```bash
git clone https://github.com/aidless/llm-lab
cd llm-lab
python3 -m venv .venv
.venv/bin/pip install -e './llm_lab[dev]'
.venv/bin/llm-lab demo
```

`pyproject.toml` lives in this directory, so the install path is `./llm_lab`
from the repository root, not `.` . The demo writes `.llm-lab-demo/audit.db`,
`result.json`, and `report.html`. Exit 0 means the fixture passed the verifier
and the chain was internally consistent. The provider id is `offline-demo-v1`.

## Run with a real model

```bash
cp llm_lab/.env.example .env
# Edit .env: LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY
.venv/bin/llm-lab run "explain the difference between supervised and self-supervised learning"
.venv/bin/llm-lab compare "write a haiku about Rust" --model-a gpt-4o-mini --model-b claude-3-5-haiku-20241022
.venv/bin/llm-lab serve
```

`serve` listens on `http://127.0.0.1:8123`. Real calls can cost money. The CLI
prints a verdict, token counts, cost, and an HTML report. That report is not
an external audit.

---

## Auditable batch evaluation

`llm-lab evalops` adds a separate batch-evaluation contract for frozen task
sets.  Unlike a single planner run, it records one final result per task,
strictly parses multiple-choice answers, separates provider and evaluator
failures, and makes the denominator for every metric explicit.  The task-set
builder currently handles the bundled C-Eval multiple-choice schema; the
runner/report contract can be exercised with any compatible frozen JSONL set.

```bash
# Freeze and verify a task set; the manifest stores the task-file hash.
llm-lab evalops verify evalops/ceval-201

# Run an OpenAI-compatible local model (Ollama or vLLM endpoint).
llm-lab evalops run evalops/ceval-201/tasks-v1.jsonl \
  --output runs/ceval-201-c8 \
  --adapter ollama \
  --model qwen25-3b-16k:latest \
  --concurrency 8 \
  --json

# Rebuild reports from raw records and run metadata.
llm-lab evalops report runs/ceval-201-c8/records.jsonl \
  --config-json runs/ceval-201-c8/run-meta.json \
  --wall-seconds 50.4 \
  --output runs/ceval-201-c8-recomputed
```

The contract includes:

- SHA-256 hashes for the task set, effective config, and answer set.
- Bounded concurrency, timeout, retry/backoff, and crash-safe resume.
- Config/task/task-count drift rejection before a resume.
- Atomic replacement of a retried task, so one task never appears twice.
- Explicit `timeout`, `transport`, `model_error`, `parse_error`, and
  `evaluator_error` failure classes.
- JSON, Markdown, HTML, and OpenCompass-compatible dataset exports.

The checked-in [`evalops/evidence-20261027`](../evalops/evidence-20261027/README.md)
package contains a verified 201-task local C-Eval subset run and a standalone
`verify_evidence.py`.  It documents both the result and its boundaries:
the subset is not an official leaderboard, local CPU timings are not a
production SLO, and the deterministic fixture is not model-quality evidence.

See [ADR-0010](./docs/adr/0010-frozen-batch-evalops.md) for the record and
resume contract.

---

## Audit log verification

Every row in `event_log` carries a SHA-256 hash chained to its predecessor
(`row_hash = sha256(prev_hash || canonical_json(row))`). The chain runs in
`id` order, so any insertion, deletion, or modification is detectable.

```bash
$ llm-lab verify
OK 138 rows verified, chain intact (12 legacy genesis rows skipped)
WARNING legacy rows have no hash. Edits inside them do not fail this command. See FAILURE_MODES.md.

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

See [`docs/adr/0006-audit-trail-integrity.md`](./docs/adr/0006-audit-trail-integrity.md)
for the design and limitations.

---

## Who this was designed around

[`ADOPTERS.md`](../ADOPTERS.md) lists no organisations and no anonymous users.
The three personas below are hypothetical composites used while writing the
tool. They are not interviews and not customers.

- **Alice — ML engineer at a 50–500 person SaaS:** you keep losing
  track of which prompt is which. [`docs/PERSONAS.md`](./docs/PERSONAS.md#p0--alice-the-ml-engineer-at-a-50-500-person-saas)
- **Bob — security architect at a regulated company:** your
  security team blocks LLM tools that don't have an audit trail.
  [`docs/PERSONAS.md`](./docs/PERSONAS.md#p0--bob-the-security-architect-at-a-regulated-company)
- **Carol — independent ML consultant:** you re-build the same
  eval pipeline on every client engagement.
  [`docs/PERSONAS.md`](./docs/PERSONAS.md#p1--carol-the-independent-ml-consultant)

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

See [`docs/adr/0008-prometheus-metrics.md`](./docs/adr/0008-prometheus-metrics.md)
for the label semantics and cardinality controls.

---

## Self-benchmark numbers

CI runs `benchmarks/self_bench.py --mode smoke` and uploads the JSON. It does
not re-run the real-provider file
[`benchmarks/v2-results.json`](./benchmarks/v2-results.json). That file is one
recorded Windows / Python 3.11 session. It is not a service level and not a
number CI checks.

### Stub run (offline, what CI runs)

Windows / Python 3.11 / offline stub, 50 steps:

| Metric | Value |
| ------ | ----- |
| Throughput | **187 steps/sec** |
| p50 latency | **5.32 ms** |
| p95 latency | **5.54 ms** |
| Cost | **$0.00** (local stub is free) |
| Fault scenarios passing | **9 / 9** |

### Real-provider run (DeepSeek v4-flash, non-thinking)

Windows / Python 3.11 / DeepSeek v4-flash, 50 steps (`benchmarks/v2-results.json`):

| Metric | Value | Notes |
| ------ | ----- | ----- |
| Throughput | **0.179 steps/sec** | real network, 50 sequential calls |
| p50 latency | **3,756 ms** | including TLS, DeepSeek queue, model |
| p95 latency | **17,597 ms** | tail latency from large-context responses |
| Total tokens | **20,740** | input + output, all 50 calls |
| Cost | **$0.062** | at DeepSeek v4-flash pricing (¥1/¥2 per 1M tokens) |
| Fault scenarios passing | **9 / 9** | identical to stub run |

> **Note on thinking mode:** the `DISABLE_LLM_THINKING=1` env var forwards
> `extra_body={thinking: {type: disabled}}` to the chat-completions
> endpoint. The numbers above are with that env var set. Without
> it, DeepSeek v4-flash defaults to internal reasoning and the
> numbers roughly **double** (50 steps in ~400s instead of ~280s,
> 33k tokens instead of 21k). For benchmark runs, set the env var.

Reproduce locally:

```bash
# Stub run (no credentials, fast)
python llm_lab/benchmarks/self_bench.py --mode all --steps 50 --output my-bench.json

# Real-provider run (DeepSeek v4-flash example)
export OPENAI_API_KEY=sk-...           # or set via $env: in PowerShell
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-v4-flash
export DISABLE_LLM_THINKING=1          # faster, cheaper
python llm_lab/benchmarks/self_bench.py --mode all --steps 50 --real --output my-bench.json
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

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the
full breakdown, and [`docs/adr/`](./docs/adr/) for the
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
| [`SECURITY.md`](../SECURITY.md) | How to report vulnerabilities, response SLA, severity scale |
| [`THREAT_MODEL.md`](../THREAT_MODEL.md) | What we defend against, what we don't, what we mitigate |
| [`GOVERNANCE.md`](../GOVERNANCE.md) | Single-maintainer → core-team evolution path |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Dev setup, PR template, security-sensitive review checklist |
| [`CHANGELOG.md`](../CHANGELOG.md) | Release history, SemVer notes |
| [`CODEOWNERS`](../CODEOWNERS) | Review routing |
| [`ADOPTERS.md`](../ADOPTERS.md) | Who's using it (empty — be the first!) |
| [`docs/adr/`](./docs/adr/) | Architecture Decision Records |

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
versions; breaking changes will be flagged in [`CHANGELOG.md`](../CHANGELOG.md).

Current tracked work: [audit-trail tamper-evidence chain (ADR-0006)](./docs/adr/0006-audit-trail-integrity.md).
