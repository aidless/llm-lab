"""FastAPI evaluation service designed for local Kubernetes acceptance tests."""

from __future__ import annotations

import asyncio
import hmac
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .jobstore import JobStore
from .settings import Settings, SettingsError


class EvalTask(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=20_000)
    gold: str = Field(min_length=1, max_length=2_000)


class EvalRequest(BaseModel):
    tasks: list[EvalTask] = Field(min_length=1, max_length=500)
    concurrency: int = Field(default=4, ge=1, le=32)


class JobSubmitRequest(EvalRequest):
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


@dataclass
class Metrics:
    request_count: int = 0
    task_count: int = 0
    error_count: int = 0
    total_latency_seconds: float = 0.0
    job_requests: int = 0
    job_cancellations: int = 0
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
                "# HELP eval_service_jobs_total Durable jobs submitted.",
                "# TYPE eval_service_jobs_total counter",
                f"eval_service_jobs_total {self.job_requests}",
                "# HELP eval_service_job_cancellations_total Job cancellation requests.",
                "# TYPE eval_service_job_cancellations_total counter",
                f"eval_service_job_cancellations_total {self.job_cancellations}",
                "",
            ]
        )

    async def record_job(self, *, cancelled: bool = False) -> None:
        async with self.lock:
            self.job_requests += 1
            self.job_cancellations += int(cancelled)


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
    job_store = JobStore(runtime_settings.database_path)

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

    async def require_api_token(
        authorization: str | None = Header(default=None),
        x_eval_token: str | None = Header(default=None),
    ) -> None:
        expected = runtime_settings.api_token
        if not expected:
            return
        supplied = x_eval_token or ""
        if not supplied and authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                supplied = value
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid evaluation API token")

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

    @app.post("/v1/evaluate", dependencies=[Depends(require_api_token)])
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

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_api_token)])
    async def submit_job(payload: JobSubmitRequest) -> dict[str, Any]:
        try:
            runtime_settings.validate_request(len(payload.tasks), payload.concurrency)
        except SettingsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request = {
            "tasks": [task.model_dump() for task in payload.tasks],
            "concurrency": payload.concurrency,
        }
        job, existing = job_store.create(
            request,
            idempotency_key=payload.idempotency_key,
        )
        if not existing:
            await metrics.record_job()
        return {
            "schema_version": "kube-eval.job.v1",
            "job_id": job["job_id"],
            "state": job["state"],
            "existing": existing,
            "status_url": f"/v1/jobs/{job['job_id']}",
            "cancel_url": f"/v1/jobs/{job['job_id']}/cancel",
        }

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_token)])
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return {"schema_version": "kube-eval.job.v1", **job_store.get(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_api_token)])
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            before = job_store.get(job_id)
            job = job_store.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if not before["cancel_requested"]:
            await metrics.record_job(cancelled=True)
        return {"schema_version": "kube-eval.job.v1", **job}

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
