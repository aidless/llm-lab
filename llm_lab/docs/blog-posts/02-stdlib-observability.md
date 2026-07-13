# Blog post M2 — "How I shipped structured JSON logging + Prometheus metrics with zero new dependencies"

> **Status:** Skeleton. Tutorial-ish. ~1000-1200 words when fleshed out.
> Target: dev.to + r/Python.

## Hook

> I almost added `structlog` and `prometheus_client` to my
> `pyproject.toml`. Then I looked at what they actually did.
> Three of my four one-job libraries were one-job libraries I
> didn't actually need.

## The reflex: "I need a logger, so I import a logger library"

The Python observability ecosystem is excellent. It's also
excessive for a small project. `structlog` is the right call when
you have a 30-engineer team shipping 50 services; it's the wrong
call when you have a single-author project that needs one JSON
line per log record.

## What I actually need

1. Every log line is one JSON object.
2. Every request has a trace id; every log line in that request
   carries it.
3. Every log line goes to stderr (so `journald` / Docker /
   `kubectl logs` pick it up).
4. No new dependencies.

## The 80-line `JsonFormatter`

[code block: 30 lines of `class JsonFormatter(logging.Formatter)`]

The interesting parts:

- `ContextVar` for trace id, so it works across `asyncio` and
  `asyncio.to_thread` (Python copies context vars across thread
  boundaries).
- `extra=` kwargs from `logger.info(..., extra={...})` get merged
  into the JSON object via the formatter, not the call site.
- LogRecord internals (`name`, `pathname`, `lineno`, …) are
  filtered out so they don't leak.

[link to `llm_lab/observability.py:124`]

## The 200-line Prometheus store

[code block: the `Metrics` class, ~80 lines]

The interesting parts:

- Thread-safe with a single `threading.Lock`. Don't need more.
- Bounded label cardinality: `path = _collapse_path_params(path)`
  replaces `/result/abc123` with `/result/:id` so we don't blow up
  time-series count.
- Render produces the Prometheus text exposition format manually
  — no `prometheus_client` dependency.

[link to `llm_lab/observability.py:226`]

## The integration

Middleware:

```python
@app.middleware("http")
async def observability_middleware(request, call_next):
    set_trace_id(request.headers.get("x-trace-id") or new_trace_id())
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        metrics().inc_request(path=_collapse_path_params(request.url.path),
                              method=request.method,
                              status=response.status_code)
        metrics().observe_request(...)
        set_trace_id(None)  # don't leak into the next request
```

`worker.call_llm` wrapper:

[code block: 30 lines showing the metrics + log emission]

## What I gave up

- `structlog`'s processor pipeline (contextvars, exception
  formatting, log levels per logger, etc.). I use ~3 of these
  features; hand-rolling 80 lines is cheaper than the dep.
- `prometheus_client`'s exemplars, pushgateway, OpenMetrics. I
  have one service; the simple text format is enough.
- TLS / auth for `/metrics`. Documented in `THREAT_MODEL.md`:
  unauth is fine for non-sensitive counters; operators who care
  can put `llm-lab` behind a reverse proxy.

## What I kept

- Real trace-id propagation end-to-end (API → runner → worker →
  verifier → tracer).
- Real Prometheus output that scrapes cleanly.
- Zero `pyproject.toml` changes (other than `prometheus_client`
  would have added).

## The 80/20

A 250-line module replaced ~16 MB of dependencies for a 1000-LOC
project. The trade is "I'll write the integration code" vs
"I'll learn their API and hope they don't break it next release".

For a single-author project, the first is the right trade.

## What's next

Next month: a hash chain for the audit log, and the multi-process
race I caught the day I wrote the regression test. (Blog post M3.)

---

**Tags:** `python` `observability` `logging` `prometheus`
**Length target:** 1000-1200 words
**Read time target:** 6-7 minutes
**CTA:** "What would you have done differently? Open an issue or
DM me."