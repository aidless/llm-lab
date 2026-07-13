How I shipped structured JSON logging + Prometheus metrics with zero new dependencies

I almost added `structlog` and `prometheus_client` to my `pyproject.toml`. Then I read what they actually do.

Both libraries are excellent. `structlog` is the right call when you have a 30-engineer team shipping 50 services. `prometheus_client` is the right call when you have five teams of consumers scraping different metrics. For a single-author Python project with one process and one user, both are over-engineered. The 80 lines of code I would have pulled in, I can write in 200. The result: zero new runtime dependencies, full control over the output, and a smaller `pip install` footprint for every user.

Here is what I did instead.

## The minimum useful observability surface

A small Python service needs four things, in order of importance:

1. Every log line is one JSON object. (No parsing for downstream tools.)
2. Every request has a trace id. Every log line in that request carries the same trace id. (So you can grep by id and see the whole story.)
3. Every log line goes to stderr. (So `journald`, Docker, and `kubectl logs` all see it without any extra configuration.)
4. Every metric is exposed in Prometheus text format at a stable URL.

`structlog` gives you #1, #2, #3 with a lot of flexibility. `prometheus_client` gives you #4 with a lot of flexibility. Both are about 16 MB of transitive dependencies combined. For a service that runs in a single process and exports maybe 20 metric names, the libraries are doing more work than the project.

## The 80-line JsonFormatter

The custom logging formatter is the simplest part. The whole thing is here:

```python
import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        tid = _trace_id_var.get()
        if tid:
            payload["trace_id"] = tid
        # Merge any extra={...} kwargs the logger call passed in.
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in payload:
                continue
            if k in ("args", "msg", "levelname", "levelno", "pathname",
                    "filename", "module", "exc_info", "exc_text",
                    "stack_info", "lineno", "funcName", "created",
                    "msecs", "relativeCreated", "thread", "threadName",
                    "processName", "process", "message", "asctime"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
```

Every log call becomes JSON. To attach structured fields, use `logger.info("...", extra={"request_id": "abc", "tokens": 412})`. The formatter merges them into the output. Three lines of filter logic keep the noise out (LogRecord internals, unjsonable values).

## Trace IDs that survive `asyncio.to_thread`

The interesting part is the trace id. Python's `ContextVar` propagates across `asyncio` boundaries — including into threads spawned by `asyncio.to_thread`, because the context is copied at the task boundary. So:

```python
@app.middleware("http")
async def observability_middleware(request, call_next):
    set_trace_id(request.headers.get("x-trace-id") or new_trace_id())
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        m.inc_request(
            path=_collapse_path_params(request.url.path),
            method=request.method,
            status=response.status_code,
        )
        set_trace_id(None)  # don't leak to the next request
```

The middleware sets the trace id at the start of every request. Every log line emitted during that request — including from background tasks, from `asyncio.to_thread` workers, from anywhere — picks up the same id. After the request, we clear it so the next request on the same thread starts fresh.

The corresponding `set_trace_id("...")` is a one-liner:

```python
def set_trace_id(trace_id: str | None) -> None:
    _trace_id_var.set(trace_id)
```

That's it. No middleware in every function. No thread-local hacks. The contextvar is propagated automatically.

## A 200-line Prometheus store

For metrics, the same logic applies. `prometheus_client` is a 16 MB package that gives you `Counter`, `Histogram`, `Gauge`, and a registry. I need exactly those three types. Here is the whole class:

```python
from collections import defaultdict
from dataclasses import dataclass, field
import threading


@dataclass
class _Histogram:
    buckets: list[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0, 60.0,
    ])
    counts: list[int] = field(default_factory=lambda: [0] * 13)
    total: int = 0
    sum: float = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum += value
        for i, ub in enumerate(self.buckets):
            if value <= ub:
                self.counts[i] += 1


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, frozenset], int] = defaultdict(int)
        self._histograms: dict[tuple[str, frozenset], _Histogram] = {}

    def inc_request(self, *, path, method, status) -> None:
        with self._lock:
            self._counters[("llm_lab_requests_total",
                            frozenset({"path": path, "method": method,
                                       "status": str(status).__class__(status)}))] += 1
```

A real implementation has more counters, but the structure is the same. Each counter is a key in a dict, indexed by `(metric_name, frozen_label_set)`. The lock is held only for the increment; reads are lock-free because we always replace the integer atomically in CPython.

The render function is the part that matters most to operators. It produces the Prometheus text format directly:

```
# HELP llm_lab_requests_total Total HTTP requests
# TYPE llm_lab_requests_total counter
llm_lab_requests_total{method="POST",path="/submit",status="200"} 1
llm_lab_requests_total{method="GET",path="/metrics",status="200"} 1
# HELP llm_lab_request_duration_seconds HTTP request duration
# TYPE llm_lab_request_duration_seconds histogram
llm_lab_request_duration_seconds_bucket{le="0.005",method="POST",path="/submit"} 0
llm_lab_request_duration_seconds_bucket{le="0.01",method="POST",path="/submit"} 0
llm_lab_request_duration_seconds_count{method="POST",path="/submit"} 1
llm_lab_request_duration_seconds_sum{method="POST",path="/submit"} 0.012
```

This is the format Prometheus scrapers expect. We feed it from `GET /metrics` and call it done.

## The label cardinality discipline

The most important thing this small store forces on you: **label cardinality**. With `prometheus_client`, you can declare `Counter("foo", "label1")` and pass any string at increment time, with no bound. That is how you accidentally create 10 million time series and get your scraper banned by Prometheus.

My `Metrics` class does not enforce this, but the call sites do. `path` is collapsed before use:

```python
def _collapse_path_params(path: str) -> str:
    parts = path.strip("/").split("/")
    out = []
    for p in parts:
        if len(p) >= 8 and all(c in "0123456789abcdefABCDEF" for c in p):
            out.append(":id")
        else:
            out.append(p)
    return "/" + "/".join(out)
```

This replaces any path segment that looks like a 12-hex id with `:id`. So `/result/abc123def456` becomes `/result/:id`. Every request to `/result/abc...` now collapses to the same series. **Cardinality is bounded by code, not by data.** This single design choice is the difference between a metrics store that scales and one that takes down your monitoring.

## When you should actually use `structlog` and `prometheus_client`

The first 200 lines I wrote are enough for a single-service project that exports a few dozen metrics and doesn't have 30 engineers shipping it. You should reach for the libraries when:

- **You need `structlog`**: distributed tracing (OpenTelemetry), log shipping to ELK / Loki, or the processor pipeline (cryptographic signing, redacting, sampling). The `ContextVar` + custom formatter pattern won't get you that.

- **You need `prometheus_client`**: pull-based gauges (memory / CPU / queue depth), exemplars (linking a metric data point to a specific trace), multi-process mode (worker pool with shared memory), or exposition formats beyond the text format (protobuf, OpenMetrics).

If you don't need any of those, **you're paying the libraries' tax without getting the benefit.** For a small service, the cost is 16 MB of transitive dependencies, an import-time penalty of about 50ms, and an API surface that hides your project's actual observability shape behind a layer of generic abstraction.

## What I have

For `llm-lab` — a single-process LLM evaluation framework — the 280 lines of `observability.py` give me:

- One JSON object per log line, with trace id + any structured fields I want
- Three metric families: HTTP requests (counter + histogram), LLM calls (counter + histogram), tokens (counter, by direction)
- A `GET /metrics` endpoint that scrapers consume natively
- A `Metrics.snapshot()` for tests, so I can assert "after this operation, the `llm_lab_tokens_total{provider="openai",direction="prompt"}` counter incremented by exactly 412"

The full code is in `llm_lab/observability.py` on GitHub. It's 280 lines including comments, type hints, and a render function. It depends on `ContextVar` (stdlib) and `dataclasses` (stdlib). Nothing else.

The same pattern would work for any service with maybe 20-50 metric series, one or two background workers, and a request flow that needs to be traceable end-to-end. If your service fits that description, try this approach first. You can always reach for `structlog` and `prometheus_client` later when the requirements actually demand it.
