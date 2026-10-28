"""FastAPI evaluation service designed for local Kubernetes acceptance tests."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .settings import Settings, SettingsError


class EvalTask(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=20_000)
    gold: str = Field(min_length=1, max_length=2_000)


class EvalRequest(BaseModel):
    tasks: list[EvalTask] = Field(min_length=1, max_length=500)
    concurrency: int = Field(default=4, ge=1, le=32)


@dataclass
class Metrics:
    request_count: int = 0
    task_count: int = 0
    error_count: int = 0
    total_latency_seconds: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(
        self,
        *,
        task_count: int,
        error_count: int,
        latency_seconds: float,
    ) -> None:
        async with self.lock:
            self.request_count += 1
            self.task_count += task_count
            self.error_count += error_count
            self.total_latency_seconds += latency_seconds

    def render(self) -> str:
        average = self.total_latency_seconds / self.request_count if self.request_count else 0.0
        return "\n".join(
            [
                "# HELP eval_service_requests_total Evaluation requests.",
                "# TYPE eval_service_requests_total counter",
                f"eval_service_requests_total {self.request_count}",
                "# HELP eval_service_tasks_total Evaluation tasks.",
                "# TYPE eval_service_tasks_total counter",
                f"eval_service_tasks_total {self.task_count}",
                "# HELP eval_service_errors_total Task-level evaluation errors.",
                "# TYPE eval_service_errors_total counter",
                f"eval_service_errors_total {self.error_count}",
                "# HELP eval_service_request_latency_seconds Average request latency.",
                "# TYPE eval_service_request_latency_seconds gauge",
                f"eval_service_request_latency_seconds {average:.6f}",
                "",
            ]
        )


def _settings() -> Settings:
    try:
        return Settings.from_env()
    except (SettingsError, ValueError) as exc:
        raise RuntimeError(f"invalid service configuration: {exc}") from exc


async def _deterministic_predict(task: EvalTask, _settings: Settings) -> str:
    await asyncio.sleep(0)
    return task.gold


async def _openai_predict(task: EvalTask, settings: Settings) -> str:
    payload = {
        "model": settings.upstream_model,
        "messages": [
            {
                "role": "system",
                "content": "Answer the task directly and concisely. Do not add labels.",
            },
            {"role": "user", "content": task.prompt},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        response = await client.post(
            f"{settings.upstream_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=payload,
        )
    if response.status_code != 200:
        raise RuntimeError(f"upstream HTTP {response.status_code}: {response.text[:500]}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid upstream response: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("upstream returned empty content")
    return content.strip()


def create_app(
    settings: Settings | None = None,
    predictor: Callable[[EvalTask, Settings], Any] | None = None,
) -> FastAPI:
    runtime_settings = settings or _settings()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        app.state.settings = runtime_settings
        app.state.metrics = metrics
        yield

    app = FastAPI(
        title="Kube Eval Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def predict(task: EvalTask) -> str:
        selected = predictor
        if selected is None:
            selected = _openai_predict if runtime_settings.mode == "openai-compatible" else _deterministic_predict
        result = selected(task, runtime_settings)
        if asyncio.iscoroutine(result):
            return str(await result)
        return str(result)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        if runtime_settings.mode == "deterministic":
            return {"status": "ready", "mode": runtime_settings.mode}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    f"{runtime_settings.upstream_base_url}/models",
                    headers={"Authorization": f"Bearer {runtime_settings.api_key}"},
                )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"upstream unavailable: {exc}") from exc
        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail=f"upstream readiness failed: HTTP {response.status_code}",
            )
        return {
            "status": "ready",
            "mode": runtime_settings.mode,
            "upstream_status": response.status_code,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/evaluate")
    async def evaluate(payload: EvalRequest, request: Request) -> dict[str, Any]:
        try:
            runtime_settings.validate_request(
                len(payload.tasks),
                payload.concurrency,
            )
        except SettingsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(payload.concurrency)

        async def run_one(task: EvalTask) -> dict[str, Any]:
            task_started = time.perf_counter()
            async with semaphore:
                try:
                    predicted = await predict(task)
                    return {
                        "id": task.id,
                        "predicted": predicted,
                        "gold": task.gold,
                        "correct": predicted.strip() == task.gold.strip(),
                        "error": None,
                        "latency_ms": (time.perf_counter() - task_started) * 1000.0,
                    }
                except Exception as exc:  # boundary: report one task, not kill the batch
                    return {
                        "id": task.id,
                        "predicted": None,
                        "gold": task.gold,
                        "correct": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": (time.perf_counter() - task_started) * 1000.0,
                    }

        results = await asyncio.gather(*(run_one(task) for task in payload.tasks))
        elapsed = time.perf_counter() - started
        errors = sum(item["error"] is not None for item in results)
        correct = sum(item["correct"] is True for item in results)
        completed = len(results) - errors
        await metrics.record(
            task_count=len(results),
            error_count=errors,
            latency_seconds=elapsed,
        )
        response = {
            "schema_version": "kube-eval.response.v1",
            "request_id": request_id,
            "mode": runtime_settings.mode,
            "summary": {
                "tasks": len(results),
                "completed": completed,
                "errors": errors,
                "correct": correct,
                "accuracy": correct / len(results) if results else None,
                "latency_ms": elapsed * 1000.0,
            },
            "results": results,
            "limitations": [
                "Deterministic mode is a service plumbing fixture, not model-quality evidence.",
                "This local Kubernetes service is not a production evaluation platform.",
                "The request cannot override the configured upstream URL.",
            ],
        }
        return response

    return app


app = create_app()


def entry() -> None:
    import uvicorn

    uvicorn.run(
        "kube_eval_service.app:app",
        host=os.getenv("EVAL_HOST", "0.0.0.0"),
        port=int(os.getenv("EVAL_PORT", "8080")),
        log_level=os.getenv("EVAL_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    entry()
