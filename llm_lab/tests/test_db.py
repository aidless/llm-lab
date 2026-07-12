"""Tests for db.py — async SQLite-backed event log."""

import json
from pathlib import Path

import aiosqlite
import pytest

from llm_lab import db as database


@pytest.fixture(autouse=True)
def _clean_db(monkeypatch):
    tmp = Path("_test_llm_lab.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(database, "DB_PATH", str(tmp))
    yield
    if tmp.exists():
        tmp.unlink()


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    await database.init_db()
    async with aiosqlite.connect(database.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='event_log'"
        )
        row = await cursor.fetchone()
        assert row is not None
        cursor2 = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_event_log_intent'"
        )
        idx = await cursor2.fetchone()
        assert idx is not None


@pytest.mark.asyncio
async def test_append_and_get_events():
    await database.init_db()
    await database.append_event(
        intent_id="test-1", seq=1, action="call", model="gpt-4o", detail="hello"
    )
    await database.append_event(
        intent_id="test-1", seq=2, action="verify", model=None, detail={"key": "value"}
    )
    events = await database.get_events("test-1")
    assert len(events) == 2
    assert events[0]["seq"] == 1
    assert events[0]["action"] == "call"
    assert events[0]["model"] == "gpt-4o"
    assert json.loads(events[0]["detail"]) == "hello"
    assert events[1]["seq"] == 2
    assert events[1]["action"] == "verify"
    assert json.loads(events[1]["detail"]) == {"key": "value"}


@pytest.mark.asyncio
async def test_events_sorted_by_seq():
    await database.init_db()
    await database.append_event(intent_id="sorted", seq=3, action="step3")
    await database.append_event(intent_id="sorted", seq=1, action="step1")
    await database.append_event(intent_id="sorted", seq=2, action="step2")
    events = await database.get_events("sorted")
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert [e["action"] for e in events] == ["step1", "step2", "step3"]


@pytest.mark.asyncio
async def test_get_events_empty():
    await database.init_db()
    events = await database.get_events("nonexistent")
    assert events == []


@pytest.mark.asyncio
async def test_get_all_events():
    await database.init_db()
    await database.append_event(intent_id="a", seq=1, action="call", model="gpt-4o", cost_usd=0.01)
    await database.append_event(intent_id="b", seq=1, action="call", model="gpt-4o-mini", cost_usd=0.001)
    all_events = await database.get_all_events()
    assert len(all_events) == 2
    assert all_events[0]["intent_id"] == "b"
    assert all_events[1]["intent_id"] == "a"
    assert "token_usage" not in all_events[0]
    assert "id" in all_events[0]


@pytest.mark.asyncio
async def test_get_all_events_columns():
    await database.init_db()
    await database.append_event(intent_id="a", seq=1, action="call")
    all_events = await database.get_all_events()
    row = all_events[0]
    expected_keys = {"id", "intent_id", "seq", "timestamp", "action", "model", "detail", "cost_usd"}
    assert expected_keys.issubset(row.keys())


@pytest.mark.asyncio
async def test_get_all_events_empty_db():
    await database.init_db()
    all_events = await database.get_all_events()
    assert all_events == []


@pytest.mark.asyncio
async def test_list_intents():
    await database.init_db()
    await database.append_event(intent_id="alpha", seq=1, action="call", model="gpt-4o")
    await database.append_event(intent_id="alpha", seq=2, action="verify")
    await database.append_event(intent_id="beta", seq=1, action="call", model="gpt-4o-mini")
    entries = await database.list_intents(limit=20)
    assert len(entries) == 2
    assert entries[0]["intent_id"] == "beta"
    assert entries[1]["intent_id"] == "alpha"


@pytest.mark.asyncio
async def test_list_intents_respects_limit():
    await database.init_db()
    for i in range(5):
        await database.append_event(intent_id=f"id-{i}", seq=1, action="call")
    entries = await database.list_intents(limit=3)
    assert len(entries) == 3


@pytest.mark.asyncio
async def test_list_intents_empty():
    await database.init_db()
    entries = await database.list_intents()
    assert entries == []


@pytest.mark.asyncio
async def test_list_intents_only_seq_1():
    await database.init_db()
    await database.append_event(intent_id="first", seq=1, action="call", model="gpt-4o")
    await database.append_event(intent_id="first", seq=2, action="verify")
    await database.append_event(intent_id="second", seq=1, action="call", model="gpt-4o-mini")
    entries = await database.list_intents()
    assert len(entries) == 2
    for e in entries:
        assert e["intent_id"] in ("first", "second")


@pytest.mark.asyncio
async def test_get_run_summary():
    await database.init_db()
    await database.append_event(
        intent_id="summary-test",
        seq=1,
        action="call",
        token_usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        cost_usd=0.005,
    )
    await database.append_event(
        intent_id="summary-test",
        seq=2,
        action="verify",
        token_usage={"total_tokens": 50},
        cost_usd=0.001,
    )
    summary = await database.get_run_summary("summary-test")
    assert summary["intent_id"] == "summary-test"
    assert summary["events"] == 2
    assert summary["total_tokens"] == 150
    assert summary["total_cost_usd"] == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_get_run_summary_no_events():
    await database.init_db()
    summary = await database.get_run_summary("nonexistent")
    assert summary["intent_id"] == "nonexistent"
    assert summary["events"] == 0
    assert summary["total_tokens"] == 0
    assert summary["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_get_run_summary_null_token_usage():
    await database.init_db()
    await database.append_event(intent_id="null-tokens", seq=1, action="call", cost_usd=0.01)
    summary = await database.get_run_summary("null-tokens")
    assert summary["total_tokens"] == 0
    assert summary["total_cost_usd"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_get_run_summary_invalid_token_usage_json():
    await database.init_db()
    async with aiosqlite.connect(database.DB_PATH) as db:
        ts = "2025-01-01T00:00:00"
        await db.execute(
            """INSERT INTO event_log (intent_id, seq, timestamp, action, token_usage, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("bad-tokens", 1, ts, "call", "{not valid json}", 0.01),
        )
        await db.commit()
    summary = await database.get_run_summary("bad-tokens")
    assert summary["total_tokens"] == 0
    assert summary["total_cost_usd"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_append_event_with_input_output_hashes():
    await database.init_db()
    await database.append_event(
        intent_id="hash-test",
        seq=1,
        action="call",
        input_text="hello world",
        output_text="goodbye world",
    )
    events = await database.get_events("hash-test")
    assert events[0]["input_hash"] == database._sha16("hello world")
    assert events[0]["output_hash"] == database._sha16("goodbye world")


@pytest.mark.asyncio
async def test_append_event_token_usage_stored_as_json():
    await database.init_db()
    tu = {"total_tokens": 42, "prompt_tokens": 20, "completion_tokens": 22}
    await database.append_event(intent_id="tu-test", seq=1, action="call", token_usage=tu)
    events = await database.get_events("tu-test")
    loaded = json.loads(events[0]["token_usage"])
    assert loaded == tu


@pytest.mark.asyncio
async def test_append_event_no_detail():
    await database.init_db()
    await database.append_event(intent_id="no-detail", seq=1, action="call")
    events = await database.get_events("no-detail")
    assert events[0]["detail"] is None


@pytest.mark.asyncio
async def test_append_event_all_params():
    await database.init_db()
    await database.append_event(
        intent_id="all-params",
        seq=99,
        action="test",
        model="deepseek-chat",
        detail={"nested": {"data": 1}},
        input_text="input",
        output_text="output",
        token_usage={"total_tokens": 10},
        cost_usd=0.00123,
    )
    events = await database.get_events("all-params")
    e = events[0]
    assert e["intent_id"] == "all-params"
    assert e["seq"] == 99
    assert e["action"] == "test"
    assert e["model"] == "deepseek-chat"
    assert json.loads(e["detail"]) == {"nested": {"data": 1}}
    assert e["input_hash"] == database._sha16("input")
    assert e["output_hash"] == database._sha16("output")
    assert json.loads(e["token_usage"]) == {"total_tokens": 10}
    assert e["cost_usd"] == pytest.approx(0.00123)


@pytest.mark.asyncio
async def test_get_run_info_returns_goal_and_model():
    await database.init_db()
    await database.append_event(
        intent_id="info-test", seq=1, action="call", model="gpt-4o",
            detail={"goal": "test goal"},
    )
    info = await database.get_run_info("info-test")
    assert info["goal"] == "test goal"
    assert info["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_get_run_info_no_events():
    await database.init_db()
    info = await database.get_run_info("nonexistent")
    assert info is None or info == {}


@pytest.mark.asyncio
async def test_get_run_info_no_goal_in_detail():
    await database.init_db()
    await database.append_event(intent_id="no-goal", seq=1, action="call", model="claude-3")
    info = await database.get_run_info("no-goal")
    assert info["goal"] == "call"
    assert info["model"] == "claude-3"


@pytest.mark.asyncio
async def test_get_run_info_with_detail_dict():
    await database.init_db()
    await database.append_event(intent_id="detail-dict-test", seq=1, action="call", model="gpt-4o", detail={"key": "value"})
    info = await database.get_run_info("detail-dict-test")
    assert info["intent_id"] == "detail-dict-test"


@pytest.mark.asyncio
async def test_get_run_info_json_decode_error():
    """detail with invalid JSON triggers except branch (db.py lines 135-136)."""
    await database.init_db()
    await database.append_event(intent_id="bad-json", seq=1, action="call", model="gpt-4o", detail="initial")
    async with aiosqlite.connect(database.DB_PATH) as db:
        await db.execute("UPDATE event_log SET detail = ? WHERE intent_id = ?", ("{bad json}", "bad-json"))
        await db.commit()
    info = await database.get_run_info("bad-json")
    assert info["goal"] == "call"
