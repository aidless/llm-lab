"""SQLite-backed event log with tamper-evident hash chain (ADR-0006).

Schema:

* ``event_log`` — every run's audit row. Tamper-evidence is implemented as
  a per-row hash chain: each row stores ``prev_hash`` (the previous row's
  ``row_hash``) and ``row_hash`` (``sha256(prev_hash || canonical_json(row))``).
  Walking the chain in ``id`` order detects any insertion, deletion, or
  modification of historical rows.
* ``tasks`` — async-task bookkeeping (not chained — it is a hot-path
  scratch table, not an audit surface).

Backward compatibility:

* Rows that pre-date the chain columns (``prev_hash`` IS NULL,
  ``row_hash`` IS NULL) are treated as **legacy genesis rows**. Verification
  starts a fresh chain from them. This means an existing audit log
  written before M3 is accepted by ``verify_log``; only new tampering is
  detected.
"""

import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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

# Columns that participate in the hash. Order matters: it must match the
# canonical-json field set so two implementations can't disagree.
_CHAIN_COLUMNS = (
    "id",
    "intent_id",
    "seq",
    "timestamp",
    "action",
    "model",
    "detail",
    "input_hash",
    "output_hash",
    "token_usage",
    "cost_usd",
    "verdict",
)
_CHAIN_COLUMNS_CSV = ", ".join(_CHAIN_COLUMNS)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sha64(text: str) -> str:
    """Full 64-char SHA-256 hex. Used for chain hashes (not truncated)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(row: dict[str, Any]) -> str:
    """Deterministic serialisation used as the hash input.

    Rules:
    * Keys sorted alphabetically.
    * No whitespace separators.
    * ``ensure_ascii=False`` so non-ASCII intent_ids hash the same on every
      platform (otherwise the chain would break across locales).
    * Numeric values are serialised via Python's default ``json.dumps``
      representation, which is platform-stable for finite floats.
    """
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _row_for_hash(row: dict[str, Any]) -> dict[str, Any]:
    """Project a row down to the hash-relevant columns.

    Excludes ``prev_hash`` (it IS the link) and ``row_hash`` (it IS the
    computed digest). Includes everything else, including nulls �?null
    must hash the same as null, not as missing.
    """
    out: dict[str, Any] = {}
    for col in _CHAIN_COLUMNS:
        v = row.get(col)
        out[col] = v
    return out


def compute_row_hash(prev_hash: str, row: dict[str, Any]) -> str:
    """Compute the chain hash for ``row`` given the previous row's hash.

    ``prev_hash`` may be the empty string (the very first row) or a 64-char
    hex string. ``row`` is the projected hash-relevant dict; ``compute_row_hash``
    does *not* read ``prev_hash`` / ``row_hash`` from the row itself.
    """
    return _sha64(prev_hash + _canonical_json(_row_for_hash(row)))


@asynccontextmanager
async def _connect() -> AsyncIterator[aiosqlite.Connection]:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    """Create tables and run idempotent migrations.

    Adds two columns (``prev_hash``, ``row_hash``) on existing databases.
    Both default to NULL �?``verify_log`` treats NULL as "legacy genesis".
    """
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        await db.executescript(_CREATE_SQL)
        cursor = await db.execute("PRAGMA table_info(event_log)")
        cols = {row["name"] for row in await cursor.fetchall()}
        if "verdict" not in cols:
            await db.execute("ALTER TABLE event_log ADD COLUMN verdict TEXT")
        if "prev_hash" not in cols:
            await db.execute("ALTER TABLE event_log ADD COLUMN prev_hash TEXT")
        if "row_hash" not in cols:
            await db.execute("ALTER TABLE event_log ADD COLUMN row_hash TEXT")
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
    """Append one event row, computing and storing its chain hash.

    Reads the last row's ``row_hash`` (within the same transaction) and
    uses it as ``prev_hash``. If no prior row exists, ``prev_hash = ""``.

    Concurrency: wraps the read-modify-write in a ``BEGIN IMMEDIATE``
    transaction so concurrent ``append_event`` calls (across processes
    or threads) serialize on the SQLite reserved lock. Without this,
    two writers reading the same prev_hash and both writing rows with
    that prev_hash would produce a chain whose second row's stored
    ``prev_hash`` no longer matches the actual previous row's
    ``row_hash`` �?i.e., a chain break ``verify_log`` would falsely flag.
    """
    ts = datetime.now(tz=timezone.utc).isoformat()
    detail_str = json.dumps(detail, ensure_ascii=False) if detail else None
    input_h = _sha16(input_text) if input_text else None
    output_h = _sha16(output_text) if output_text else None
    tu_str = json.dumps(token_usage, ensure_ascii=False) if token_usage else None

    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        # Acquire the reserved lock up front. aiosqlite exposes the
        # underlying connection via ``db._conn`` but we go through the
        # public ``execute`` so the same code path works on the connection
        # wrapper. ``busy_timeout=5000`` (set in ``_connect``) makes
        # contending writers wait up to 5 s for the lock.
        await db.execute("BEGIN IMMEDIATE")

        try:
            # Look up the previous row's hash (the link). Inside the
            # IMMEDIATE transaction this read is guaranteed to see the
            # committed state of any prior writer.
            prev_cursor = await db.execute(
                "SELECT row_hash FROM event_log ORDER BY id DESC LIMIT 1"
            )
            prev_row = await prev_cursor.fetchone()
            prev_hash = (
                prev_row["row_hash"] if prev_row and prev_row["row_hash"] else ""
            )

            # Insert with a placeholder row_hash so we can compute the real
            # one after we know the autoincrement id.
            cursor = await db.execute(
                """INSERT INTO event_log
                   (intent_id, seq, timestamp, action, model, detail, input_hash, output_hash,
                    token_usage, cost_usd, verdict, prev_hash, row_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    intent_id,
                    seq,
                    ts,
                    action,
                    model,
                    detail_str,
                    input_h,
                    output_h,
                    tu_str,
                    cost_usd,
                    verdict,
                    prev_hash,
                    "",
                ),
            )
            new_id = cursor.lastrowid
            if new_id is None:
                # SQLite's INTEGER PRIMARY KEY AUTOINCREMENT always assigns
                # a rowid; if it didn't, the INSERT itself failed.
                raise RuntimeError("event_log INSERT did not return a row id")

            # Compute the row hash using the row's full column set. We read
            # the row back so the hash matches what ``verify_log`` will
            # recompute (single source of truth: column �?hash).
            sel_cursor = await db.execute(
                f"SELECT {_CHAIN_COLUMNS_CSV} FROM event_log WHERE id = ?",  # noqa: B608, S608
                (new_id,),
            )
            row = await sel_cursor.fetchone()
            if row is None:
                raise RuntimeError(f"event_log INSERT vanished: id={new_id}")
            row_dict = dict(row)

            row_hash = compute_row_hash(prev_hash, row_dict)
            await db.execute(
                "UPDATE event_log SET row_hash = ? WHERE id = ?",
                (row_hash, new_id),
            )
            await db.commit()
        except Exception:
            # Roll back so we don't leave the connection in an open
            # transaction (which would deadlock subsequent appends).
            await db.rollback()
            raise


async def get_events(intent_id: str) -> list[dict[str, Any]]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM event_log WHERE intent_id = ? ORDER BY seq", (intent_id,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_all_events() -> list[dict[str, Any]]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, intent_id, seq, timestamp, action, model, detail, cost_usd "
            "FROM event_log ORDER BY id DESC LIMIT 10000"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def list_intents(limit: int = 20) -> list[dict[str, Any]]:
    async with _connect() as db:
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
    async with _connect() as db:
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
    async with _connect() as db:
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
    async with _connect() as db:
        await db.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Chain verification (ADR-0006)
# ---------------------------------------------------------------------------


async def verify_log(*, limit: int | None = None) -> dict[str, Any]:
    """Walk the chain in ``id`` order and report the first inconsistency.

    Returns a dict with:
        * ``ok`` �?True if every row's stored ``row_hash`` matches the
          recomputed one, AND every row's ``prev_hash`` matches the
          previous row's stored ``row_hash``.
        * ``rows_checked`` �?number of rows walked.
        * ``first_break`` �?``None`` or ``{"id": int, "expected": str,
          "found": str, "kind": "row_hash" | "prev_hash"}``.
        * ``legacy_genesis_count`` �?number of rows with NULL
          ``prev_hash`` / ``row_hash`` (pre-M3 rows).
    """
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        # Fetch everything we need (id + the chain columns + prev_hash + row_hash)
        # in one go. ``id`` orders rows globally; intent_id+seq do not, because
        # multiple intents can interleave.
        cols_csv = ", ".join(_CHAIN_COLUMNS)
        # Column list is a module constant, not user input �?safe to interpolate.
        sql = (
            f"SELECT id, prev_hash, row_hash, {cols_csv} FROM event_log "  # noqa: B608, S608
            "ORDER BY id ASC"
        )
        if limit:
            # ``limit`` is a Python int, not user input; the int() guard makes
            # that explicit and protects against future callers passing a str.
            sql += f" LIMIT {int(limit)}"
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()

    prev_row_hash = ""
    rows_checked = 0
    legacy_genesis = 0
    for row in rows:
        rows_checked += 1
        d = dict(row)
        stored_row_hash = d.pop("row_hash")
        stored_prev_hash = d.pop("prev_hash")

        if stored_row_hash is None and stored_prev_hash is None:
            # Legacy row (pre-M3). Treat as the start of a sub-chain.
            legacy_genesis += 1
            prev_row_hash = ""  # next chained row treats it as the start
            continue

        if stored_prev_hash != prev_row_hash:
            return {
                "ok": False,
                "rows_checked": rows_checked,
                "first_break": {
                    "id": d["id"],
                    "kind": "prev_hash",
                    "expected": prev_row_hash,
                    "found": stored_prev_hash,
                },
                "legacy_genesis_count": legacy_genesis,
            }

        expected = compute_row_hash(stored_prev_hash, d)
        if stored_row_hash != expected:
            return {
                "ok": False,
                "rows_checked": rows_checked,
                "first_break": {
                    "id": d["id"],
                    "kind": "row_hash",
                    "expected": expected,
                    "found": stored_row_hash,
                },
                "legacy_genesis_count": legacy_genesis,
            }

        prev_row_hash = stored_row_hash

    return {
        "ok": True,
        "rows_checked": rows_checked,
        "first_break": None,
        "legacy_genesis_count": legacy_genesis,
    }