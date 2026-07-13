# ADR-0007: Structured JSON logging via stdlib, not structlog

- **Status:** Accepted
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

`llm-lab` runs as a long-lived service. Operators need to:

1. Reconstruct a single request's trace across multiple log lines (API entry →
   runner → worker → verifier → tracer).
2. Ship logs to a centralised aggregator without bespoke parsing.
3. Diagnose silent failures — when the verifier reports `pass` but the
   model output is clearly nonsense, the question is "what happened
   between submit and result?".

Stdlib `logging` calls scattered across the codebase answer none of these
questions. We need **structured JSON logs** with a stable shape.

## Decision drivers

- **Zero new dependencies.** `llm-lab` already has 16 declared deps. Adding
  `structlog` (or `python-json-logger`) for one feature is a poor trade.
- **Cross-process consistency.** The CLI, the FastAPI service, the runner,
  and the verifier all need the same log shape. `structlog` makes this
  easy but is overkill if `logging.Formatter` with a `json.dumps` does it.
- **Stdlib `ContextVar`** gives us per-task trace-id propagation with no
  extra plumbing. It works across `asyncio` and sync boundaries (because
  `asyncio.to_thread` copies the context).

## Considered options

1. **Stdlib `logging` + custom `JsonFormatter`** (chosen). One new module
   (`llm_lab.observability`) that owns the formatter, the context-var for
   trace id, and the JSON-line shape.
2. **`structlog`** — industry-standard, plays well with stdlib logging as a
   sink. Adds one dep + a new API for every log call.
3. **Plain `print(..., file=sys.stderr)` with manual JSON** — works, but
   loses all the `logging` ecosystem benefits (level filtering, handler
   fan-out, library interop).

## Decision outcome

**Option 1 — stdlib `logging` with a JSON formatter.**

We add `llm_lab.observability` with three primitives:

- `configure_logging()` — idempotent root-logger setup. One handler emits
  one JSON object per line. Default level `INFO`; override via
  `LLM_LAB_LOG_LEVEL`.
- `set_trace_id() / current_trace_id() / with_trace_id()` — a
  `ContextVar` propagates the trace id across async / sync boundaries.
  Middleware sets a fresh id per HTTP request, or honours an inbound
  `x-trace-id` header.
- `JsonFormatter` — emits a stable JSON shape. Standard keys
  (`ts`, `level`, `logger`, `msg`) always present; `trace_id` included
  when in scope; extra kwargs from `logger.info(..., extra={...})` merged
  in.

Log shape:

```json
{"ts": "2026-07-13T12:00:00.000+00:00",
 "level": "INFO",
 "logger": "llm_lab.worker",
 "msg": "llm call complete",
 "trace_id": "abc123def456",
 "provider": "openai",
 "model": "gpt-4o-mini",
 "tokens": 412,
 "cost_usd": 0.00023,
 "finish_reason": "stop",
 "duration_seconds": 0.412}
```

### Consequences

**Positive:**

- Zero new deps.
- Stable shape across all log calls.
- `trace_id` round-trips end-to-end (set in middleware → visible in
  worker / runner / verifier / tracer).
- Backwards-compatible: callers using `logging.warning(...)` keep working,
  their output just becomes JSON.

**Negative:**

- Custom `extra=` kwargs must use a known set of keys to appear in the
  JSON (`provider`, `model`, `tokens`, `cost_usd`, `verdict`, `intent_id`,
  `step`). Anything outside this set is included but may collide with
  LogRecord internals; the formatter filters those out.

**Neutral:**

- Existing log strings like `"tracer.trace_call_sync failed: %s"` continue
  to work; we leave them in place until we have evidence operators want
  more structure.

## Validation

- `tests/test_observability.py::test_json_formatter_*` — unit tests on the
  formatter's output shape.
- `tests/test_observability.py::test_trace_id_*` — context propagation.
- `tests/test_observability.py::test_trace_id_echoed_in_response_header` —
  end-to-end trace-id propagation through FastAPI.

## Links / references

- `llm_lab/observability.py` — implementation.
- ADR-0008 — companion ADR for the Prometheus metrics store.