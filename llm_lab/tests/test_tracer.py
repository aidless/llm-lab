"""Tests for tracer.py — Langfuse-backed with local SQLite fallback."""

import os

import pytest

from llm_lab import tracer
from llm_lab.models import Verdict


def test_tracer_falls_back_to_local_without_langfuse():
    os.environ.pop("LANGFUSE_SECRET_KEY", None)
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    lf = tracer._get_lf()
    assert lf is None


def test_tracer_does_not_crash_without_langfuse():
    import asyncio

    from llm_lab import db as _local_db

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_local_db.init_db())
    loop.run_until_complete(tracer.trace_call("test-id", 1, "test-model", "hello", "world"))
    loop.run_until_complete(tracer.trace_event("test-id", 0, "plan", "test goal"))
    loop.run_until_complete(tracer.trace_verdict("test-id", 1, Verdict(label="pass", reason="ok")))
    loop.close()


def test_shutdown_no_langfuse():
    tracer.shutdown()


@pytest.mark.asyncio
async def test_trace_event_without_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from pathlib import Path

    from llm_lab import db as _local_db

    tmp = Path("_test_tracer.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    await _local_db.init_db()
    await tracer.trace_event("te-test", 1, "plan", "my detail")

    events = await _local_db.get_events("te-test")
    assert len(events) == 1
    assert events[0]["action"] == "plan"
    assert events[0]["seq"] == 1

    if tmp.exists():
        tmp.unlink()


@pytest.mark.asyncio
async def test_trace_event_without_detail(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from pathlib import Path

    from llm_lab import db as _local_db

    tmp = Path("_test_tracer2.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    await _local_db.init_db()
    await tracer.trace_event("te-nodetail", 1, "verify")

    events = await _local_db.get_events("te-nodetail")
    assert len(events) == 1
    assert events[0]["detail"] is None

    if tmp.exists():
        tmp.unlink()


@pytest.mark.asyncio
async def test_trace_verdict_all_labels(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from pathlib import Path

    from llm_lab import db as _local_db

    tmp = Path("_test_tracer3.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    await _local_db.init_db()
    for label in ("pass", "fail", "partial"):
        await tracer.trace_verdict("tv-labels", 1, Verdict(label=label, reason=f"because {label}"))

    events = await _local_db.get_events("tv-labels")
    assert len(events) == 3
    for e in events:
        assert e["action"] == "verify"

    if tmp.exists():
        tmp.unlink()


@pytest.mark.asyncio
async def test_get_trace_without_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from pathlib import Path

    from llm_lab import db as _local_db

    tmp = Path("_test_tracer4.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    await _local_db.init_db()
    await tracer.trace_call("gt-test", 1, "gpt-4o", "hello", "world")

    result = await tracer.get_trace("gt-test")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["intent_id"] == "gt-test"
    assert result[0]["action"] == "call"

    if tmp.exists():
        tmp.unlink()


@pytest.mark.asyncio
async def test_get_trace_empty_without_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from pathlib import Path

    from llm_lab import db as _local_db

    tmp = Path("_test_tracer5.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    await _local_db.init_db()
    result = await tracer.get_trace("nonexistent")
    assert isinstance(result, list)
    assert result == []

    if tmp.exists():
        tmp.unlink()


@pytest.mark.asyncio
async def test_get_summary_delegates_to_db(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from pathlib import Path

    from llm_lab import db as _local_db

    tmp = Path("_test_tracer6.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    await _local_db.init_db()
    await _local_db.append_event(
        intent_id="gs-test",
        seq=1,
        action="call",
        token_usage={"total_tokens": 200},
        cost_usd=0.01,
    )

    summary = await tracer.get_summary("gs-test")
    assert summary["intent_id"] == "gs-test"
    assert summary["total_tokens"] == 200
    assert summary["total_cost_usd"] == pytest.approx(0.01)

    if tmp.exists():
        tmp.unlink()


def test_shutdown_handles_none_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracer._lf = None
    tracer.shutdown()


def test_get_lf_returns_none_when_only_secret_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracer._lf = None
    assert tracer._get_lf() is None


def test_get_lf_returns_none_when_only_public_set(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    assert tracer._get_lf() is None


# ── Langfuse paths ────────────────────────────────────────────────────────


def test_get_lf_returns_instance_when_both_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    lf = tracer._get_lf()
    assert lf is not None


def test_get_lf_caches_instance(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    lf1 = tracer._get_lf()
    lf2 = tracer._get_lf()
    assert lf1 is lf2


def test_ensure_trace_creates_langfuse_trace(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    tracer._trace_cache.clear()

    mock_lf = MagicMock()
    mock_trace = MagicMock()
    mock_lf.trace.return_value = mock_trace
    tracer._lf = mock_lf

    tracer._ensure_trace("test-trace-intent", "test goal")
    mock_lf.trace.assert_called_once_with(name="test-trace-intent", input="test goal")
    assert tracer._trace_cache.get("test-trace-intent") is mock_trace


def test_trace_call_with_langfuse(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    tracer._trace_cache.clear()

    import asyncio

    mock_trace = MagicMock()
    tracer._lf = MagicMock()
    tracer._trace_cache["lf-call-intent"] = mock_trace

    asyncio.run(
        tracer.trace_call(
            "lf-call-intent", 1, "gpt-4o", "hello", "world", {"prompt_tokens": 5, "completion_tokens": 10}, 0.001
        )
    )
    mock_trace.generation.assert_called_once()
    _, kwargs = mock_trace.generation.call_args
    assert kwargs["name"] == "step-1"
    assert kwargs["model"] == "gpt-4o"


def test_trace_event_with_langfuse(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    tracer._trace_cache.clear()

    import asyncio

    mock_trace = MagicMock()
    tracer._lf = MagicMock()
    tracer._trace_cache["lf-event-intent"] = mock_trace

    asyncio.run(tracer.trace_event("lf-event-intent", 1, "plan", "detail"))
    mock_trace.span.assert_called_once()


def test_trace_verdict_with_langfuse(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    tracer._trace_cache.clear()

    import asyncio

    mock_trace = MagicMock()
    tracer._lf = MagicMock()
    tracer._trace_cache["lf-verdict-intent"] = mock_trace

    asyncio.run(tracer.trace_verdict("lf-verdict-intent", 1, Verdict(label="pass", reason="ok")))
    mock_trace.score.assert_called_once()
    _, kwargs = mock_trace.score.call_args
    assert kwargs["value"] == 1.0


def test_trace_verdict_with_langfuse_partial(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    tracer._trace_cache.clear()

    import asyncio

    mock_trace = MagicMock()
    tracer._lf = MagicMock()
    tracer._trace_cache["lf-verdict-intent"] = mock_trace

    asyncio.run(tracer.trace_verdict("lf-verdict-intent", 1, Verdict(label="partial", reason="maybe")))
    mock_trace.score.assert_called_once()
    _, kwargs = mock_trace.score.call_args
    assert kwargs["value"] == 0.5


def test_trace_verdict_with_langfuse_fail(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None
    tracer._trace_cache.clear()

    import asyncio

    mock_trace = MagicMock()
    tracer._lf = MagicMock()
    tracer._trace_cache["lf-verdict-intent"] = mock_trace

    asyncio.run(tracer.trace_verdict("lf-verdict-intent", 1, Verdict(label="fail", reason="bad")))
    mock_trace.score.assert_called_once()
    _, kwargs = mock_trace.score.call_args
    assert kwargs["value"] == 0.0


def test_get_trace_with_langfuse_returns_dict(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracer._lf = None

    import asyncio

    result = asyncio.run(tracer.get_trace("lf-get-trace"))
    assert isinstance(result, dict)
    assert result["source"] == "langfuse"


def test_get_lf_import_error(monkeypatch):
    """tracer.py 30-31: _get_lf returns None when langfuse import fails."""
    import sys
    import types

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    # Inject a dummy langfuse module with no Langfuse class -> ImportError
    dummy = types.ModuleType("langfuse")
    sys.modules["langfuse"] = dummy

    tracer._lf = None
    lf = tracer._get_lf()
    assert lf is None


def test_shutdown_with_langfuse(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    mock_lf = MagicMock()
    tracer._lf = mock_lf

    tracer.shutdown()
    mock_lf.flush.assert_called_once()
