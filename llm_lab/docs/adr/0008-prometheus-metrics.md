# ADR-0008: Prometheus metrics, in-process store, zero deps

- **Status:** Accepted
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

For the "eval tool you can show your security team" wedge, we need to
prove two things in production:

1. **The service is healthy** — request rate, error rate, p95 latency.
2. **The LLM calls are working** — provider / model / outcome breakdown,
   per-call latency, total tokens consumed (for cost control).

Operators expect Prometheus-style scraping. We need a `/metrics` endpoint
that emits the Prometheus text exposition format, with stable label
semantics and bounded cardinality.

## Decision drivers

- **Zero new dependencies.** `prometheus_client` is the standard, but
  pulls a non-trivial dep tree (protobuf, etc.) for what we need.
- **Bounded label cardinality.** Path templates collapse dynamic segments
  (`/result/abc123def456` → `/result/:id`) so a flood of unique intent
  ids doesn't blow up the metric store.
- **Thread-safe.** The FastAPI app is async but the runner / worker do
  sync I/O in `concurrent.futures.ThreadPoolExecutor`. A single process-
  wide store with a `threading.Lock` is sufficient — we do not need
  cross-process aggregation (single-host deployment only).
- **Fail-loud on labels.** A label collision with a reserved word would
  silently corrupt the exposition. We escape special chars
  (`"`, `\`, newline) in every label value.

## Considered options

1. **In-process store + bespoke Prometheus renderer** (chosen). ~250 LOC,
   zero deps.
2. **`prometheus_client`** — standard, but adds the dependency and
   changes our middleware shape (we'd need its `Counter` / `Histogram`
   types instead of method calls).
3. **Push to a metrics backend (StatsD / OTLP).** Adds an external
   dependency and a network call we don't otherwise need.

## Decision outcome

**Option 1 — bespoke in-process store.**

`llm_lab.observability.Metrics` exposes:

- **Counters**:
  - `llm_lab_requests_total{path, method, status}` — every HTTP request.
  - `llm_lab_llm_calls_total{provider, model, outcome}` — every LLM call
    made via `worker.call_llm`.
  - `llm_lab_tokens_total{provider, model, direction}` — prompt vs
    completion tokens, summed.
- **Histograms** (Prometheus-style cumulative buckets + count + sum):
  - `llm_lab_request_duration_seconds{path, method}` — HTTP request
    wall-clock.
  - `llm_lab_llm_call_duration_seconds{provider, model}` — per-call
    latency, including retry / fallback paths.

`/metrics` returns the rendered text. The endpoint is **intentionally
unauthenticated** — Prometheus scrapers don't carry credentials, and the
data is non-sensitive (counters over our own request volume, not user
data). Documented in `THREAT_MODEL.md`.

### Path-template normalisation

`/result/{intent_id}`, `/status/{task_id}`, `/export/{kind}/{id}`, etc.,
would otherwise explode label cardinality. `observability._collapse_path_params`
replaces any path segment that is ≥ 8 hex characters with `:id`. This
keeps cardinality bounded (~30 known paths, not 30 + N_unique_intents).

### `/metrics` access policy

The endpoint is unauthenticated, but the response contains only counters
and histograms — no request bodies, no PII, no secrets. Acceptable for
scraping by Prometheus or Victoria Metrics inside a trusted network.
Operators that want auth can front `llm-lab` with a reverse proxy.

### Consequences

**Positive:**

- Zero new deps.
- Path-template collapse keeps label cardinality predictable.
- Renderer is small enough that operators can audit it (one file).

**Negative:**

- We do not get the broader Prometheus ecosystem (pushgateway, exemplars,
  OpenMetrics) — only the text exposition format.
- Single-process aggregation only. Multi-worker deployments (e.g.,
  uvicorn `--workers 4`) each have their own store; Prometheus needs to
  scrape each worker's port.

**Neutral:**

- Histogram buckets are fixed at `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
  0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]` seconds. Good enough for HTTP and
  LLM-call latency distributions; revisit if we add long-running jobs.

## Validation

- `tests/test_observability.py::test_metrics_*` — unit tests for the
  counters, histograms, and renderer.
- `tests/test_observability.py::test_metrics_endpoint_*` — end-to-end
  HTTP tests using `TestClient`.
- `benchmarks/self_bench.py --mode all` exercises the metrics path under
  load and renders the snapshot in `benchmarks/v1-results.json`.

## Links / references

- `llm_lab/observability.py` — implementation.
- ADR-0007 — companion ADR for the structured logs that share the same
  trace id.