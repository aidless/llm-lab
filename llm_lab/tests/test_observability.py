"""Smoke tests for the observability module + Prometheus ``/metrics`` endpoint."""

from __future__ import annotations

import io
import json
import logging

from fastapi.testclient import TestClient

from llm_lab import observability as obs
from llm_lab.observability import (
    JsonFormatter,
    Metrics,
    metrics_snapshot,
    with_trace_id,
)

# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def _make_record(msg: str, **extra) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="llm_lab.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_formatter_basic():
    """A bare record produces a well-formed JSON line with the standard keys."""
    f = JsonFormatter()
    out = f.format(_make_record("hello"))
    payload = json.loads(out)
    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "llm_lab.test"
    assert "ts" in payload
    assert "trace_id" not in payload  # none in scope


def test_json_formatter_includes_extras():
    """Extra kwargs on the record are merged into the JSON object."""
    f = JsonFormatter()
    out = f.format(
        _make_record(
            "step complete",
            model="gpt-4o",
            tokens=412,
            cost_usd=0.00023,
            verdict="pass",
        )
    )
    payload = json.loads(out)
    assert payload["model"] == "gpt-4o"
    assert payload["tokens"] == 412
    assert payload["cost_usd"] == 0.00023
    assert payload["verdict"] == "pass"


def test_json_formatter_includes_trace_id():
    """A trace id set in scope appears in every record."""
    f = JsonFormatter()
    with with_trace_id("abc123def456"):
        out = f.format(_make_record("hi"))
    payload = json.loads(out)
    assert payload["trace_id"] == "abc123def456"


def test_json_formatter_includes_exception():
    """exc_info is serialised into the JSON object."""
    f = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord(
            name="llm_lab.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="oops",
            args=(),
            exc_info=sys.exc_info(),
        )
        payload = json.loads(f.format(rec))
    assert "exc_info" in payload
    assert "ValueError: boom" in payload["exc_info"]


# ---------------------------------------------------------------------------
# Trace-id context
# ---------------------------------------------------------------------------


def test_trace_id_round_trip():
    """``with_trace_id`` yields the id and restores the previous on exit."""
    obs.set_trace_id(None)
    with with_trace_id("outer"):
        assert obs.current_trace_id() == "outer"
        with with_trace_id("inner"):
            assert obs.current_trace_id() == "inner"
        # Restored on exit.
        assert obs.current_trace_id() == "outer"
    assert obs.current_trace_id() is None


def test_new_trace_id_is_unique():
    a = obs.new_trace_id()
    b = obs.new_trace_id()
    assert a != b
    assert len(a) == 12


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_counter_increments():
    m = Metrics()
    m.inc_request(path="/submit", method="POST", status=200)
    m.inc_request(path="/submit", method="POST", status=200)
    m.inc_request(path="/submit", method="POST", status=500)
    snap = m.snapshot()
    counter = snap["counters"]
    # The snapshot key format is ``f"{name}|{dict(label_fs)}"``; match by
    # substring against the repr-style dict (uses ``:`` not ``=``).
    matches_200 = [
        v
        for k, v in counter.items()
        if "'status': '200'" in k and "'method': 'POST'" in k
    ]
    matches_500 = [
        v
        for k, v in counter.items()
        if "'status': '500'" in k and "'method': 'POST'" in k
    ]
    assert sum(matches_200) == 2
    assert sum(matches_500) == 1


def test_metrics_histogram_observe():
    m = Metrics()
    for v in [0.01, 0.02, 0.5, 5.0, 30.0]:
        m.observe_request(path="/x", method="GET", seconds=v)
    snap = m.snapshot()
    key = next(k for k in snap["histograms"] if "llm_lab_request_duration_seconds" in k)
    assert snap["histograms"][key]["count"] == 5
    assert abs(snap["histograms"][key]["sum"] - 35.53) < 0.01


def test_metrics_render_is_valid_prometheus():
    """The rendered output looks like a Prometheus exposition payload."""
    m = Metrics()
    m.inc_request(path="/submit", method="POST", status=200)
    m.observe_request(path="/submit", method="POST", seconds=0.123)
    out = m.render()
    # TYPE headers
    assert "# TYPE llm_lab_requests_total counter" in out
    assert "# TYPE llm_lab_request_duration_seconds histogram" in out
    # Counter line — labels are sorted alphabetically when rendered.
    assert 'llm_lab_requests_total{method="POST",path="/submit",status="200"} 1' in out
    # Histogram has the +Inf bucket and a sum
    assert 'llm_lab_request_duration_seconds_count{method="POST",path="/submit"} 1' in out or \
           'llm_lab_request_duration_seconds_count{path="/submit",method="POST"} 1' in out
    assert 'le="+Inf"' in out


def test_metrics_render_escapes_quotes_in_labels():
    """Label values with quotes / backslashes are escaped."""
    m = Metrics()
    m.inc_request(path='/sub"path', method="GET", status=200)
    out = m.render()
    assert 'path="/sub\\"path"' in out


# ---------------------------------------------------------------------------
# /metrics HTTP endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_prometheus_text():
    """The /metrics route serves Prometheus exposition format."""
    from llm_lab.main import app

    with TestClient(app) as client:  # triggers lifespan
        # Make at least one request first so there's something to render.
        client.get("/")
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # The body should contain at least one TYPE header (we made a GET / above).
    assert "# TYPE" in resp.text


def test_metrics_endpoint_records_request():
    """A real request increments the requests counter for its path."""
    from llm_lab.main import app

    with TestClient(app) as client:  # triggers lifespan
        client.get("/")  # warm-up
        needle = "'status': '200'"
        before = _sum_request_counters(matches=needle)
        client.get("/")
        after = _sum_request_counters(matches=needle)
    assert after >= before + 1, f"expected increment but before={before} after={after}"


def _sum_request_counters(*, matches: str) -> int:
    """Sum all ``llm_lab_requests_total`` counter entries whose key contains ``matches``."""
    snap = metrics_snapshot()["counters"]
    total = 0
    for k, v in snap.items():
        if "llm_lab_requests_total" in k and matches in k:
            total += v
    return total


def test_metrics_endpoint_unauthenticated():
    """``/metrics`` is intentionally public — no API key needed."""
    from llm_lab.main import app

    with TestClient(app) as client:
        # No X-API-Key header.
        resp = client.get("/metrics")
    assert resp.status_code == 200


def test_trace_id_echoed_in_response_header():
    """Every request carries an ``x-trace-id`` header for client correlation."""
    from llm_lab.main import app

    with TestClient(app) as client:
        resp = client.get("/")
    assert "x-trace-id" in resp.headers
    assert len(resp.headers["x-trace-id"]) == 12


def test_inbound_trace_id_is_honoured():
    """If the caller supplies ``x-trace-id``, we propagate it through."""
    from llm_lab.main import app

    with TestClient(app) as client:
        resp = client.get("/", headers={"x-trace-id": "deadbeef0001"})
    assert resp.headers["x-trace-id"] == "deadbeef0001"


# ---------------------------------------------------------------------------
# configure_logging idempotency
# ---------------------------------------------------------------------------


def test_configure_logging_is_idempotent():
    """Calling configure_logging twice doesn't add two handlers."""
    obs.configure_logging(level="INFO", stream=io.StringIO())
    handlers_before = [h for h in logging.getLogger().handlers if getattr(h, "_llm_lab_json", False)]
    obs.configure_logging(level="INFO", stream=io.StringIO())
    handlers_after = [h for h in logging.getLogger().handlers if getattr(h, "_llm_lab_json", False)]
    assert len(handlers_before) == len(handlers_after) == 1