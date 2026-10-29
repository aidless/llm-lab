"""Persistent worker process for queued evaluation jobs."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import uuid
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, Field

from .app import EvalTask, _deterministic_predict, _openai_predict
from .jobstore import JobStore
from .settings import Settings


class WorkerTaskRequest(BaseModel):
    tasks: list[EvalTask] = Field(min_length=1, max_length=500)
    concurrency: int = Field(default=4, ge=1, le=32)


async def _execute(
    request: dict[str, Any],
    settings: Settings,
    store: JobStore,
    job_id: str,
) -> dict[str, Any]:
    parsed = WorkerTaskRequest.model_validate(request)
    settings.validate_request(len(parsed.tasks), parsed.concurrency)
    predictor = _openai_predict if settings.mode == "openai-compatible" else _deterministic_predict
    semaphore = asyncio.Semaphore(parsed.concurrency)

    def cancelled_result(task: EvalTask) -> dict[str, Any]:
        return {
            "id": task.id,
            "predicted": None,
            "gold": task.gold,
            "correct": None,
            "error": "cancelled",
            "latency_ms": 0.0,
        }

    def is_cancelled() -> bool:
        job = store.get(job_id)
        return job["state"] != "running" or job["cancel_requested"]

    async def run_one(task: EvalTask) -> dict[str, Any]:
        await asyncio.sleep(0)
        if is_cancelled():
            return cancelled_result(task)
        async with semaphore:
            try:
                if settings.task_delay_seconds:
                    await asyncio.sleep(settings.task_delay_seconds)
                predicted = await predictor(task, settings)
                if is_cancelled():
                    return cancelled_result(task)
                return {
                    "id": task.id,
                    "predicted": str(predicted),
                    "gold": task.gold,
                    "correct": str(predicted).strip() == task.gold.strip(),
                    "error": None,
                    "latency_ms": 0.0,
                }
            except Exception as exc:
                return {
                    "id": task.id,
                    "predicted": None,
                    "gold": task.gold,
                    "correct": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": 0.0,
                }

    results = await asyncio.gather(*(run_one(task) for task in parsed.tasks))
    errors = sum(item["error"] is not None for item in results)
    correct = sum(item["correct"] is True for item in results)
    cancelled = sum(item["error"] == "cancelled" for item in results)
    return {
        "schema_version": "kube-eval.worker-result.v1",
        "job_id": job_id,
        "summary": {
            "tasks": len(results),
            "completed": len(results) - errors,
            "errors": errors,
            "correct": correct,
            "accuracy": correct / len(results) if results else None,
            "cancelled": cancelled,
        },
        "results": results,
    }


async def _heartbeat(
    store: JobStore,
    job_id: str,
    owner: str,
    lease_seconds: float,
) -> None:
    """Renew a lease while this worker is actively executing the job."""
    interval = max(0.25, lease_seconds / 3.0)
    while True:
        await asyncio.sleep(interval)
        if not store.renew_lease(job_id, owner, lease_seconds):
            print(f"[worker] lease lost job={job_id}", file=sys.stderr, flush=True)
            return


async def run_worker(settings: Settings, *, stop_event: asyncio.Event | None = None) -> None:
    store = JobStore(settings.database_path)
    owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    recovered = store.recover_expired(max_attempts=settings.max_attempts)
    if recovered:
        print(f"[worker] recovered {recovered} expired lease(s)", file=sys.stderr, flush=True)
    stop = stop_event or asyncio.Event()
    try:
        while not stop.is_set():
            job = store.claim(
                owner=owner,
                lease_seconds=settings.lease_seconds,
                max_attempts=settings.max_attempts,
            )
            if job is None:
                await asyncio.sleep(0.25)
                continue
            job_id = str(job["job_id"])
            heartbeat = asyncio.create_task(
                _heartbeat(store, job_id, owner, settings.lease_seconds)
            )
            try:
                result = await _execute(job["request"], settings, store, job_id)
                state = store.complete(job_id, result, owner=owner)
                print(f"[worker] job={job_id} state={state}", file=sys.stderr, flush=True)
            except Exception as exc:
                state = store.fail(
                    job_id,
                    f"{type(exc).__name__}: {exc}",
                    owner=owner,
                    max_attempts=settings.max_attempts,
                )
                print(f"[worker] job={job_id} state={state} error={exc}", file=sys.stderr, flush=True)
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
    finally:
        store.close()


def entry() -> None:
    from .settings import Settings

    settings = Settings.from_env()
    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    for signal_name in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signal_name, stop.set)
    try:
        loop.run_until_complete(run_worker(settings, stop_event=stop))
    finally:
        loop.close()


if __name__ == "__main__":
    entry()
