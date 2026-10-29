"""SQLite-backed job state shared by API and worker processes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_owner() -> str:
    return f"local:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class JobStore:
    """Small durable queue with idempotency, leases and terminal-state wins."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
        yield connection

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    state TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    lease_owner TEXT,
                    lease_expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS jobs_state_created
                    ON jobs(state, created_at);
                """
            )
            # Keep a previously-created smoke database usable after this upgrade.
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for name, declaration in (
                ("lease_owner", "TEXT"),
                ("lease_expires_at", "REAL"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def create(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if idempotency_key:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing, True
        job_id = uuid.uuid4().hex
        timestamp = _now()
        try:
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, idempotency_key, state, request_json,
                        created_at, updated_at
                    ) VALUES (?, ?, 'queued', ?, ?, ?)
                    """,
                    (
                        job_id,
                        idempotency_key,
                        json.dumps(request, ensure_ascii=False, sort_keys=True),
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError:
            if idempotency_key:
                existing = self.get_by_idempotency_key(idempotency_key)
                if existing is not None:
                    return existing, True
            raise
        return self.get(job_id), False

    def get(self, job_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode(row)

    def get_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (key,)).fetchone()
        return self._decode(row) if row is not None else None

    def claim(
        self,
        *,
        owner: str | None = None,
        lease_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> dict[str, Any] | None:
        """Atomically claim one queued job or recover only an expired lease."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        claim_owner = owner or _default_owner()
        lease_deadline = time.time() + lease_seconds
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_locked(connection, max_attempts=max_attempts)
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state = 'queued' AND attempts < ?
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (max_attempts,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            timestamp = _now()
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state = 'running', attempts = attempts + 1,
                    error = NULL, started_at = ?, updated_at = ?,
                    lease_owner = ?, lease_expires_at = ?
                WHERE job_id = ? AND state = 'queued' AND attempts < ?
                """,
                (
                    timestamp,
                    timestamp,
                    claim_owner,
                    lease_deadline,
                    row["job_id"],
                    max_attempts,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get(str(row["job_id"]))

    def renew_lease(self, job_id: str, owner: str, lease_seconds: float) -> bool:
        """Extend a lease only while this worker still owns the running job."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running' AND lease_owner = ?
                """,
                (time.time() + lease_seconds, _now(), job_id, owner),
            )
        return cursor.rowcount == 1

    def complete(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        owner: str | None = None,
    ) -> str:
        """Write a result only if the caller still owns a running job."""
        timestamp = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            where = "job_id = ? AND state = 'running'"
            values: list[Any] = [
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                timestamp,
                timestamp,
                job_id,
            ]
            if owner is not None:
                where += " AND lease_owner = ?"
                values.append(owner)
            connection.execute(
                f"""
                UPDATE jobs
                SET state = 'completed', result_json = ?, error = NULL,
                    finished_at = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE {where}
                """,
                values,
            )
            row = connection.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return str(row["state"])

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        owner: str | None = None,
        max_attempts: int | None = None,
    ) -> str:
        """Fail or requeue a worker error, bounded by max_attempts."""
        timestamp = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, attempts, lease_owner FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["state"] not in {"queued", "running"}:
                connection.commit()
                return str(row["state"])
            if owner is not None and row["lease_owner"] != owner:
                connection.commit()
                return str(row["state"])
            retry = max_attempts is not None and int(row["attempts"]) < max_attempts
            if retry:
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'queued', error = ?, updated_at = ?,
                        started_at = NULL, finished_at = NULL,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE job_id = ? AND state IN ('queued', 'running')
                    """,
                    (error[:2000], timestamp, job_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'failed', error = ?, finished_at = ?,
                        updated_at = ?, lease_owner = NULL, lease_expires_at = NULL
                    WHERE job_id = ? AND state IN ('queued', 'running')
                    """,
                    (error[:2000], timestamp, timestamp, job_id),
                )
            row = connection.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return str(row["state"])

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel queued/running work atomically; completed work is unchanged."""
        timestamp = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1, state = 'cancelled',
                    finished_at = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE job_id = ? AND state IN ('queued', 'running')
                """,
                (timestamp, timestamp, job_id),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self._decode(row)

    def recover_expired(self, *, max_attempts: int = 2, stale_seconds: float = 0.0) -> int:
        """Recover only leases older than ``stale_seconds``; never live claims."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = self._recover_expired_locked(
                connection,
                max_attempts=max_attempts,
                stale_seconds=stale_seconds,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return count

    def recover_orphaned(self, *, max_attempts: int = 2) -> int:
        """Backward-compatible name for lease-expiry recovery."""
        return self.recover_expired(max_attempts=max_attempts)

    @staticmethod
    def _recover_expired_locked(
        connection: sqlite3.Connection,
        *,
        max_attempts: int,
        stale_seconds: float = 0.0,
    ) -> int:
        cutoff = time.time() - stale_seconds
        timestamp = _now()
        rows = connection.execute(
            """
            SELECT job_id, attempts FROM jobs
            WHERE state = 'running'
              AND (
                lease_owner IS NULL OR lease_expires_at IS NULL
                OR lease_expires_at <= ?
              )
            """,
            (cutoff,),
        ).fetchall()
        for row in rows:
            if int(row["attempts"]) < max_attempts:
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'queued', error = 'worker lease expired; requeued',
                        updated_at = ?, started_at = NULL,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE job_id = ? AND state = 'running'
                    """,
                    (timestamp, row["job_id"]),
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'failed',
                        error = 'worker lease expired after maximum attempts',
                        finished_at = ?, updated_at = ?,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE job_id = ? AND state = 'running'
                    """,
                    (timestamp, timestamp, row["job_id"]),
                )
        return len(rows)

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["request"] = json.loads(payload.pop("request_json"))
        raw_result = payload.pop("result_json")
        payload["result"] = json.loads(raw_result) if raw_result else None
        payload["cancel_requested"] = bool(payload["cancel_requested"])
        return payload
