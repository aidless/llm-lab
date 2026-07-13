"""Structured JSON logging + Prometheus metrics for ``llm-lab``.

This module is the single source of truth for cross-process observability:

* :func:`configure_logging` wires the root logger to a JSON formatter that
  emits one JSON object per line, with a ``trace_id`` (carried via
  :class:`TraceContext` / :func:`current_trace_id`) and standard fields
  (``ts``, ``level``, ``logger``, ``msg``, ``exc_info``).
* :class:`Metrics` provides an in-process counter / histogram store. It is
  deliberately tiny — no third-party deps. The :func:`render_prometheus`
  helper serialises the snapshot to the Prometheus text exposition format,
  which the ``GET /metrics`` endpoint serves.

Trace-id propagation
--------------------

Within a single request, set the trace id via :func:`set_trace_id`. Use
:func:`with_trace_id` as a context manager for short-lived scopes (e.g., a
background task). All log records emitted inside the scope will carry the
id.

JSON log shape
--------------

::

    {"ts": "2026-01-01T12:00:00.000000+00:00",
     "level": "INFO",
     "logger": "llm_lab.runner",
     "msg": "step complete",
     "trace_id": "abc123def456",
     "intent_id": "abc123def456",
     "step": 3,
     "model": "gpt-4o-mini",
     "tokens": 412,
     "cost_usd": 0.00023,
     "verdict": "pass"}

Every record carries ``trace_id`` if one is in scope; extra keyword arguments
passed to the logger are merged into the JSON.

Metrics
-------

Counters: ``llm_lab_requests_total{path,method,status}``,
``llm_lab_llm_calls_total{provider,model,outcome}``,
``llm_lab_tokens_total{provider,model,direction}`` (``direction`` is
``prompt`` / ``completion``).

Histograms: ``llm_lab_request_duration_seconds{path,method}``,
``llm_lab_llm_call_duration_seconds{provider,model}``.

Use :func:`metrics_snapshot` to get a serialisable snapshot for tests.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Trace-id context
# ---------------------------------------------------------------------------

_trace_id_var: ContextVar[str | None] = ContextVar("llm_lab_trace_id", default=None)


def current_trace_id() -> str | None:
    """Return the trace id in scope, or ``None`` if none has been set."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None) -> None:
    """Set the trace id for the current context. ``None`` clears it."""
    _trace_id_var.set(trace_id)


def new_trace_id() -> str:
    """Generate a new 12-char trace id and set it as current."""
    tid = uuid.uuid4().hex[:12]
    set_trace_id(tid)
    return tid


@contextmanager
def with_trace_id(trace_id: str | None = None) -> Iterator[str]:
    """Scope a trace id. Yields the id; restores the previous one on exit."""
    token = _trace_id_var.set(trace_id or new_trace_id())
    try:
        yield _trace_id_var.get() or ""
    finally:
        _trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

# Keys that the formatter treats as "standard" — they always appear in the
# output (with a default of null) when in scope. Extra kwargs from the logger
# call are merged in verbatim.
_STD_KEYS = ("trace_id", "intent_id", "step", "model", "provider", "tokens", "cost_usd", "verdict")


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record.

    Honours :func:`current_trace_id` and merges ``extra`` kwargs from the
    logger call into the top-level JSON object.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        trace_id = current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id

        # Merge known extra fields first so caller-supplied ones win on
        # collision. Logger kwargs land on record.__dict__ as plain attrs.
        for k in _STD_KEYS:
            v = getattr(record, k, None)
            if v is not None:
                payload.setdefault(k, v)

        # All other custom kwargs the caller passed via ``logger.info(..., extra={...})``.
        # Skip the standard LogRecord attributes to avoid leaking internals.
        _log_record_internal = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "asctime",
        }
        for k, v in record.__dict__.items():
            if k in _log_record_internal or k in payload or k in _STD_KEYS:
                continue
            try:
                json.dumps(v)
            except TypeError:
                v = repr(v)
            payload[k] = v

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Stable serialisation so log lines diff cleanly across runs.
        return json.dumps(payload, ensure_ascii=False, sort_keys=False)


def configure_logging(level: str | int | None = None, *, stream=None) -> None:
    """Configure the root logger to emit JSON lines.

    Idempotent: removes any previously installed JSON handler before adding a
    fresh one. ``level`` defaults to ``$LLM_LAB_LOG_LEVEL`` or ``INFO``.
    """
    if level is None:
        level = os.getenv("LLM_LAB_LOG_LEVEL", "INFO")

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers we previously installed (idempotency).
    for h in list(root.handlers):
        if getattr(h, "_llm_lab_json", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._llm_lab_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel("WARNING")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class _Histogram:
    """Fixed-bucket histogram. Keeps buckets + count + sum."""
    buckets: list[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
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

    def render(self, name: str, labels: dict[str, str]) -> str:
        # Prometheus convention: cumulative buckets end with ``+Inf``.
        lines = [f"# TYPE {name} histogram"]
        cumulative = 0
        for ub, c in zip(self.buckets, self.counts, strict=False):
            cumulative += c
            label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
            label_str = "{" + label_str + "}" if label_str else ""
            lines.append(f'{name}_bucket{{le="{ub}",{label_str[1:-1] if label_str else ""}}} {cumulative}')
        # +Inf bucket = total observations.
        label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
        if label_str:
            lines.append(f'{name}_bucket{{le="+Inf",{label_str}}} {self.total}')
            lines.append(f'{name}_count{{{label_str}}} {self.total}')
            lines.append(f'{name}_sum{{{label_str}}} {self.sum:.6f}')
        else:
            lines.append(f'{name}_bucket{{le="+Inf"}} {self.total}')
            lines.append(f"{name}_count {self.total}")
            lines.append(f"{name}_sum {self.sum:.6f}")
        return "\n".join(lines)


def _escape(value: str) -> str:
    """Escape a label value for Prometheus exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metrics:
    """Thread-safe in-process metrics store.

    Three counter families (request, llm_calls, tokens) and two histograms
    (request duration, llm call duration). Labels are passed as kwargs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, frozenset], int] = defaultdict(int)
        self._histograms: dict[tuple[str, frozenset], _Histogram] = {}

    # --- counters --------------------------------------------------------

    def inc_request(self, *, path: str, method: str, status: int) -> None:
        self._counter("llm_lab_requests_total", path=path, method=method, status=str(status))

    def inc_llm_call(self, *, provider: str, model: str, outcome: str) -> None:
        self._counter(
            "llm_lab_llm_calls_total", provider=provider, model=model, outcome=outcome
        )

    def inc_tokens(self, *, provider: str, model: str, direction: str, count: int) -> None:
        if count <= 0:
            return
        self._counter(
            "llm_lab_tokens_total",
            provider=provider,
            model=model,
            direction=direction,
            _=str(count),
        )

    def _counter(self, name: str, **labels: str) -> None:
        # ``_`` is the value-packed label for tokens (workaround for the lack
        # of explicit counter-with-value helpers).
        key = (name, frozenset(labels.items()))
        with self._lock:
            self._counters[key] += 1 if "_" not in labels else int(labels["_"])

    # --- histograms ------------------------------------------------------

    def observe_request(self, *, path: str, method: str, seconds: float) -> None:
        self._observe("llm_lab_request_duration_seconds", path=path, method=method, v=seconds)

    def observe_llm_call(self, *, provider: str, model: str, seconds: float) -> None:
        self._observe(
            "llm_lab_llm_call_duration_seconds", provider=provider, model=model, v=seconds
        )

    def _observe(self, name: str, v: float, **labels: str) -> None:
        key = (name, frozenset(labels.items()))
        with self._lock:
            hist = self._histograms.get(key)
            if hist is None:
                hist = _Histogram()
                self._histograms[key] = hist
            hist.observe(v)

    # --- rendering -------------------------------------------------------

    def render(self) -> str:
        """Render the full snapshot in Prometheus exposition format."""
        out: list[str] = []
        with self._lock:
            # Group counters by metric name for TYPE headers.
            counter_groups: dict[str, list[tuple[dict[str, str], int]]] = defaultdict(list)
            for (name, label_fs), value in self._counters.items():
                labels = dict(label_fs)
                if "_" in labels:
                    del labels["_"]
                counter_groups[name].append((labels, value))

            for name, entries in sorted(counter_groups.items()):
                out.append(f"# TYPE {name} counter")
                # Sort by canonicalised label string for stable output.
                entries_sorted = sorted(
                    entries,
                    key=lambda pair: ",".join(f"{k}={v}" for k, v in sorted(pair[0].items())),
                )
                for labels, value in entries_sorted:
                    label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
                    out.append(f"{name}{{{label_str}}} {value}")

            for (name, label_fs), hist in sorted(self._histograms.items()):
                labels = dict(label_fs)
                if "v" in labels:
                    del labels["v"]
                out.append(hist.render(name, labels))

        return "\n".join(out) + "\n"

    def snapshot(self) -> dict[str, Any]:
        """Serialisable snapshot — for tests."""
        with self._lock:
            return {
                "counters": {
                    f"{name}|{dict(label_fs)}": value
                    for (name, label_fs), value in self._counters.items()
                },
                "histograms": {
                    f"{name}|{dict(label_fs)}": {
                        "count": hist.total,
                        "sum": hist.sum,
                    }
                    for (name, label_fs), hist in self._histograms.items()
                },
            }


# Global metrics instance — set up at process start.
_metrics = Metrics()


def metrics() -> Metrics:
    """Return the process-wide :class:`Metrics` instance."""
    return _metrics


def render_prometheus() -> str:
    """Render the global metrics in Prometheus exposition format."""
    return _metrics.render()


def metrics_snapshot() -> dict[str, Any]:
    """Snapshot of the global metrics — used by tests."""
    return _metrics.snapshot()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Measure wall-clock for an LLM call. Populates ``elapsed`` in the dict.

    Use as::

        with timed() as t:
            result = worker.call_llm(...)
        metrics().observe_llm_call(provider=..., model=..., seconds=t["elapsed"])

    The ``provider`` / ``model`` labels are passed to :func:`metrics`
    separately — they are not the timer context's concern.
    """
    payload: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield payload
    finally:
        payload["elapsed"] = time.perf_counter() - t0