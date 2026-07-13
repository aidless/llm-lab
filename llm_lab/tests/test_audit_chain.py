"""Tests for the tamper-evident hash chain in ``llm_lab.db`` (ADR-0006).

Covers:

* Backward-compatibility: pre-M3 rows (no ``prev_hash`` / ``row_hash``)
  are accepted as legacy genesis rows.
* Round-trip: appending rows produces a chain that ``verify_log`` reports
  intact.
* Tamper detection: a modified row is caught at the right place.
* Insertion / deletion detection: chain breaks at the next row.
* CLI surface: ``llm-lab verify`` exits 0 on a clean log and 1 on a broken
  one, with the right message.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from llm_lab import cli as cli_mod
from llm_lab import db as database


@pytest.fixture(autouse=True)
def _clean_db(monkeypatch, tmp_path):
    db_file = tmp_path / "chain_test.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    yield db_file


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_db_adds_chain_columns_on_existing_db():
    """Migration is idempotent and creates the chain columns."""
    await database.init_db()
    with sqlite3.connect(database.DB_PATH) as raw_db:
        cols = {row[1] for row in raw_db.execute("PRAGMA table_info(event_log)").fetchall()}
    assert "prev_hash" in cols
    assert "row_hash" in cols


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_verifies_clean_log():
    """Appending N rows produces a chain that ``verify_log`` reports intact."""
    await database.init_db()
    for i in range(5):
        await database.append_event(
            intent_id="chain-1",
            seq=i + 1,
            action="call",
            model="gpt-4o",
            output_text=f"output {i}",
        )
    report = await database.verify_log()
    assert report["ok"] is True
    assert report["rows_checked"] == 5
    assert report["first_break"] is None
    assert report["legacy_genesis_count"] == 0


@pytest.mark.asyncio
async def test_chain_links_each_row_to_previous():
    """Each row's ``prev_hash`` equals the previous row's ``row_hash``."""
    await database.init_db()
    for i in range(3):
        await database.append_event(
            intent_id="chain-2", seq=i + 1, action="call", model="m"
        )
    rows = await database.get_events("chain-2")
    assert len(rows) == 3
    # Each row's stored prev_hash equals the previous row's stored row_hash.
    assert rows[0]["prev_hash"] == ""  # first row �?empty
    assert rows[0]["row_hash"] != ""  # but has its own hash
    for prev, cur in zip(rows, rows[1:], strict=False):
        assert cur["prev_hash"] == prev["row_hash"]


@pytest.mark.asyncio
async def test_chain_digest_is_deterministic():
    """The same row content produces the same hash on re-computation."""
    await database.init_db()
    await database.append_event(
        intent_id="det-1", seq=1, action="call", model="m", output_text="hello"
    )
    rows = await database.get_events("det-1")
    row = rows[0]
    project = {k: row.get(k) for k in database._CHAIN_COLUMNS}
    expected = database.compute_row_hash(row["prev_hash"], project)
    assert expected == row["row_hash"]


# ---------------------------------------------------------------------------
# Tamper detection �?modify an existing row's content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_detects_modification_of_existing_row():
    await database.init_db()
    for i in range(4):
        await database.append_event(
            intent_id="tamper-1",
            seq=i + 1,
            action="call",
            model="gpt-4o",
            output_text=f"original {i}",
        )
    # Mutate the third row's verdict directly in the DB (bypass the API).
    with sqlite3.connect(database.DB_PATH) as raw_db:
        raw_db.execute(
            "UPDATE event_log SET verdict = ? WHERE seq = 3 AND intent_id = ?",
            ("fail-faked", "tamper-1"),
        )
        raw_db.commit()
    report = await database.verify_log()
    assert report["ok"] is False
    assert report["first_break"] is not None
    assert report["first_break"]["kind"] == "row_hash"
    assert report["first_break"]["id"] > 0  # some row id


@pytest.mark.asyncio
async def test_chain_detects_deletion():
    """Deleting a middle row breaks the chain at the next one."""
    await database.init_db()
    for i in range(4):
        await database.append_event(
            intent_id="del-1", seq=i + 1, action="call", model="m"
        )
    with sqlite3.connect(database.DB_PATH) as raw_db:
        # Delete the second row �?chain breaks at the third row's prev_hash check.
        raw_db.execute(
            "DELETE FROM event_log WHERE intent_id = ? AND seq = 2", ("del-1",)
        )
        raw_db.commit()
    report = await database.verify_log()
    assert report["ok"] is False
    # The break is the third row (its prev_hash now points to row 1, not row 2).
    # It might also be the last row (its row_hash is recomputed from wrong prev_hash).
    assert report["first_break"] is not None


# ---------------------------------------------------------------------------
# Backward compatibility �?pre-M3 rows are legacy genesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_rows_accepted_as_genesis():
    """Rows with NULL prev_hash / row_hash are accepted as chain starts."""
    await database.init_db()
    # Bypass append_event to write a "legacy" row directly.
    with sqlite3.connect(database.DB_PATH) as raw_db:
        raw_db.execute(
            """INSERT INTO event_log
               (intent_id, seq, timestamp, action, model, detail,
                input_hash, output_hash, token_usage, cost_usd, verdict,
                prev_hash, row_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
            (
                "legacy-1", 1, "2026-01-01T00:00:00+00:00", "call", "m",
                None, None, None, None, None, None,
            ),
        )
        raw_db.commit()
    # And add a chained row on top.
    await database.append_event(
        intent_id="legacy-1", seq=2, action="call", model="m"
    )
    report = await database.verify_log()
    assert report["ok"] is True
    assert report["legacy_genesis_count"] == 1
    assert report["rows_checked"] == 2


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_verify_exits_zero_on_clean_log():
    """Autouse fixture provides an isolated DB path."""
    asyncio.run(database.init_db())
    for i in range(3):
        asyncio.run(
            database.append_event(
                intent_id="cli-clean", seq=i + 1, action="call", model="m"
            )
        )

    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["verify"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "3 rows verified" in result.output


def test_cli_verify_exits_one_on_broken_chain(monkeypatch, tmp_path):
    db_file = tmp_path / "cli_broken.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    asyncio.run(database.init_db())
    for i in range(3):
        asyncio.run(
            database.append_event(
                intent_id="cli-broken", seq=i + 1, action="call", model="m"
            )
        )
    # Tamper with the middle row.
    with sqlite3.connect(str(db_file)) as raw_db:
        raw_db.execute(
            "UPDATE event_log SET verdict = ? WHERE seq = 2 AND intent_id = ?",
            ("forged", "cli-broken"),
        )
        raw_db.commit()

    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["verify"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "expected" in result.output and "found" in result.output


def test_cli_verify_json_output():
    """Autouse fixture provides an isolated DB path."""
    asyncio.run(database.init_db())
    asyncio.run(
        database.append_event(
            intent_id="cli-json", seq=1, action="call", model="m"
        )
    )

    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["verify", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["rows_checked"] == 1


# ---------------------------------------------------------------------------
# Idempotency of init_db (running it twice does not duplicate columns)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_db_is_idempotent():
    await database.init_db()
    await database.init_db()  # should not raise / duplicate
    with sqlite3.connect(database.DB_PATH) as raw_db:
        cols = [row[1] for row in raw_db.execute("PRAGMA table_info(event_log)").fetchall()]
    # Exactly one prev_hash, one row_hash, one verdict.
    assert cols.count("prev_hash") == 1
    assert cols.count("row_hash") == 1
    assert cols.count("verdict") == 1


# ---------------------------------------------------------------------------
# Concurrency — multi-writer appends must not break the chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_appends_produce_valid_chain():
    """4 concurrent writers, each appending 25 rows → chain verifies intact.

    This is the regression test for S2 in REVIEW-M3.md. Without
    ``BEGIN IMMEDIATE``, two writers can read the same ``prev_hash``
    before either commits, producing rows whose ``prev_hash`` no longer
    matches the previous row's actual ``row_hash`` — ``verify_log``
    would flag a (false-positive) tamper.
    """
    import asyncio

    await database.init_db()

    async def writer(writer_id: int, n: int) -> None:
        for i in range(n):
            await database.append_event(
                intent_id=f"conc-{writer_id}",
                seq=i + 1,
                action="call",
                model="m",
                output_text=f"writer {writer_id} step {i}",
            )

    await asyncio.gather(*(writer(w, 25) for w in range(4)))
    report = await database.verify_log()
    assert report["ok"] is True, (
        f"concurrent appends broke the chain; "
        f"first_break={report['first_break']}, "
        f"rows_checked={report['rows_checked']}"
    )
    # 4 writers × 25 rows = 100 rows.
    assert report["rows_checked"] == 100