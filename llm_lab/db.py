import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import aiosqlite

DB_PATH = os.getenv("LLM_LAB_DB_PATH", "llm_lab.db")
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS event_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id    TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    timestamp    TEXT NOT NULL,
    action       TEXT NOT NULL,
    model        TEXT,
    detail       TEXT,
    input_hash   TEXT,
    output_hash  TEXT,
    token_usage  TEXT,
    cost_usd     REAL,
    verdict      TEXT
);
CREATE INDEX IF NOT EXISTS idx_event_log_intent ON event_log(intent_id);
CREATE TABLE IF NOT EXISTS tasks (
    task_id    TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    payload    TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_updated ON tasks(updated_at);
"""


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.executescript(_CREATE_SQL)
        # Migrate existing databases that predate the `verdict` column.
        cursor = await db.execute("PRAGMA table_info(event_log)")
        cols = {row["name"] for row in await cursor.fetchall()}
        if "verdict" not in cols:
            await db.execute("ALTER TABLE event_log ADD COLUMN verdict TEXT")
        await db.commit()


async def append_event(
    intent_id: str,
    seq: int,
    action: str,
    model: str | None = None,
    detail: str | dict[str, Any] | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    token_usage: dict[str, Any] | None = None,
    cost_usd: float | None = None,
    verdict: str | None = None,
) -> None:
    ts = datetime.now(tz=timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO event_log
               (intent_id, seq, timestamp, action, model, detail, input_hash, output_hash, token_usage, cost_usd, verdict)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intent_id,
                seq,
                ts,
                action,
                model,
                json.dumps(detail, ensure_ascii=False) if detail else None,
                _sha16(input_text) if input_text else None,
                _sha16(output_text) if output_text else None,
                json.dumps(token_usage, ensure_ascii=False) if token_usage else None,
                cost_usd,
                verdict,
            ),
        )
        await db.commit()


async def get_events(intent_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM event_log WHERE intent_id = ? ORDER BY seq", (intent_id,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_all_events() -> list[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, intent_id, seq, timestamp, action, model, detail, cost_usd "
            "FROM event_log ORDER BY id DESC LIMIT 10000"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def list_intents(limit: int = 20) -> list[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT intent_id, action, model, MIN(timestamp) as timestamp "
            "FROM event_log WHERE seq = 1 GROUP BY intent_id "
            "ORDER BY MAX(id) DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_run_summary(intent_id: str) -> dict[str, Any]:
    events = await get_events(intent_id)
    total_tokens = 0
    total_cost = 0.0
    for e in events:
        if e["token_usage"]:
            try:
                tu = json.loads(e["token_usage"])
                total_tokens += tu.get("total_tokens", 0)
            except (json.JSONDecodeError, TypeError):
                pass
        if e["cost_usd"]:
            total_cost += e["cost_usd"]
    return {
        "intent_id": intent_id,
        "events": len(events),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
    }


async def get_run_info(intent_id: str) -> dict[str, Any]:
    events = await get_events(intent_id)
    if not events:
        return {}
    first = events[0]
    detail_raw = first.get("detail") or "{}"
    try:
        detail = json.loads(detail_raw) if isinstance(detail_raw, str) else {}
        if not isinstance(detail, dict):
            detail = {}
    except (json.JSONDecodeError, TypeError):
        detail = {}
    return {
        "intent_id": intent_id,
        "goal": detail.get("goal", first.get("action", "")),
        "model": first.get("model", ""),
    }


async def save_task(task_id: str, payload: dict[str, Any]) -> None:
    ts = datetime.now(tz=timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO tasks (task_id, status, payload, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                   status=excluded.status,
                   payload=excluded.payload,
                   updated_at=excluded.updated_at""",
            (task_id, payload.get("status"), json.dumps(payload, ensure_ascii=False), ts),
        )
        await db.commit()


async def get_task(task_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT payload FROM tasks WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload"])
    except (json.JSONDecodeError, TypeError):
        return None


async def delete_task(task_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        await db.commit()
