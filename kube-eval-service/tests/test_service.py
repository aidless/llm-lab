"""API tests for the Kubernetes evaluation service."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from kube_eval_service.app import EvalTask, create_app
from kube_eval_service.jobstore import JobStore
from kube_eval_service.settings import Settings


def _settings(tmp_path: Any = None, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "mode": "deterministic",
        "upstream_base_url": "",
        "upstream_model": "",
        "api_key": "local",
        "timeout_seconds": 5.0,
        "max_tasks": 20,
        "max_concurrency": 8,
        "role": "api",
        "database_path": str(Path(tmp_path or "/tmp") / "jobs.sqlite3"),
        "api_token": "",
        "max_attempts": 2,
        "lease_seconds": 30.0,
        "task_delay_seconds": 0.0,
    }
    values.update(overrides)
    return Settings(**values)


def test_health_readiness_metrics_and_evaluation(tmp_path: Any) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json()["status"] == "ready"
        response = client.post(
            "/v1/evaluate",
            json={
                "concurrency": 2,
                "tasks": [
                    {"id": "a", "prompt": "2+2", "gold": "4"},
                    {"id": "b", "prompt": "3+3", "gold": "6"},
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["tasks"] == 2
        assert payload["summary"]["correct"] == 2
        assert payload["summary"]["errors"] == 0
        metrics = client.get("/metrics").text
        assert "eval_service_requests_total 1" in metrics
        assert "eval_service_tasks_total 2" in metrics


def test_request_limits_reject_oversized_or_excessive_requests(tmp_path: Any) -> None:
    with TestClient(create_app(_settings(tmp_path, max_tasks=1))) as client:
        response = client.post(
            "/v1/evaluate",
            json={
                "tasks": [
                    {"id": "a", "prompt": "a", "gold": "a"},
                    {"id": "b", "prompt": "b", "gold": "b"},
                ]
            },
        )
        assert response.status_code == 422


def test_task_failure_is_isolated(tmp_path: Any) -> None:
    def predictor(task: EvalTask, _settings: Settings) -> str:
        if task.id == "bad":
            raise RuntimeError("synthetic failure")
        return task.gold

    with TestClient(create_app(_settings(tmp_path), predictor=predictor)) as client:
        payload = client.post(
            "/v1/evaluate",
            json={
                "tasks": [
                    {"id": "ok", "prompt": "a", "gold": "a"},
                    {"id": "bad", "prompt": "b", "gold": "b"},
                ]
            },
        ).json()
    assert payload["summary"]["completed"] == 1
    assert payload["summary"]["errors"] == 1
    failed = next(item for item in payload["results"] if item["id"] == "bad")
    assert failed["correct"] is None
    assert "synthetic failure" in failed["error"]


def test_openai_mode_requires_fixed_upstream(tmp_path: Any) -> None:
    missing = Settings(
        mode="openai-compatible",
        upstream_base_url="",
        upstream_model="",
        api_key="local",
        timeout_seconds=5.0,
        max_tasks=10,
        max_concurrency=4,
        role="api",
        database_path=str(tmp_path / "jobs.sqlite3"),
        api_token="",
        max_attempts=2,
        lease_seconds=30.0,
        task_delay_seconds=0.0,
    )
    with TestClient(create_app(missing)) as client:
        response = client.get("/readyz")
    assert response.status_code == 503


def test_durable_job_submission_is_idempotent_and_persistent(tmp_path: Any) -> None:
    settings = _settings(tmp_path)
    payload = {
        "idempotency_key": "request-42",
        "tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}],
    }
    with TestClient(create_app(settings)) as client:
        first = client.post("/v1/jobs", json=payload)
        second = client.post("/v1/jobs", json=payload)
        assert first.status_code == 202
        assert first.json()["existing"] is False
        assert second.json()["existing"] is True
        job_id = first.json()["job_id"]
        status = client.get(f"/v1/jobs/{job_id}")
        assert status.json()["state"] == "queued"

    reopened = JobStore(settings.database_path)
    assert reopened.get(job_id)["request"]["tasks"][0]["id"] == "a"


def test_job_cancel_and_expired_lease_recovery(tmp_path: Any) -> None:
    settings = _settings(tmp_path)
    store = JobStore(settings.database_path)
    queued, _ = store.create({"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}], "concurrency": 1})
    cancelled = store.cancel(queued["job_id"])
    assert cancelled["state"] == "cancelled"

    interrupted, _ = store.create({"tasks": [{"id": "b", "prompt": "3+3", "gold": "6"}], "concurrency": 1})
    claimed = store.claim(owner="worker-a", lease_seconds=0.01, max_attempts=2)
    assert claimed is not None
    assert claimed["job_id"] == interrupted["job_id"]
    time.sleep(0.03)
    assert store.recover_expired(max_attempts=2) == 1
    assert store.get(interrupted["job_id"])["state"] == "queued"


def test_live_lease_is_not_recovered_by_another_worker(tmp_path: Any) -> None:
    settings = _settings(tmp_path)
    first_worker = JobStore(settings.database_path)
    second_worker = JobStore(settings.database_path)
    first, _ = first_worker.create(
        {"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}], "concurrency": 1}
    )
    second, _ = first_worker.create(
        {"tasks": [{"id": "b", "prompt": "3+3", "gold": "6"}], "concurrency": 1}
    )

    claimed = first_worker.claim(owner="worker-a", lease_seconds=60, max_attempts=2)
    assert claimed is not None
    claimed_id = claimed["job_id"]
    remaining_id = second["job_id"] if claimed_id == first["job_id"] else first["job_id"]
    assert second_worker.recover_expired(max_attempts=2) == 0

    other = second_worker.claim(owner="worker-b", lease_seconds=60, max_attempts=2)
    assert other is not None and other["job_id"] == remaining_id
    assert second_worker.get(claimed_id)["lease_owner"] == "worker-a"


def test_max_attempts_bounds_lease_recovery_and_worker_failures(tmp_path: Any) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job, _ = store.create(
        {"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}], "concurrency": 1}
    )
    claimed = store.claim(owner="worker-a", lease_seconds=0.01, max_attempts=2)
    assert claimed is not None
    time.sleep(0.03)
    assert store.recover_expired(max_attempts=2) == 1
    assert store.get(job["job_id"])["state"] == "queued"

    claimed = store.claim(owner="worker-b", lease_seconds=0.01, max_attempts=2)
    assert claimed is not None and claimed["attempts"] == 2
    time.sleep(0.03)
    assert store.recover_expired(max_attempts=2) == 1
    assert store.get(job["job_id"])["state"] == "failed"
    assert store.claim(owner="worker-c", lease_seconds=60, max_attempts=2) is None


def test_cancel_wins_against_in_flight_completion(tmp_path: Any) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job, _ = store.create(
        {"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}], "concurrency": 1}
    )
    claimed = store.claim(owner="worker-a", lease_seconds=60, max_attempts=2)
    assert claimed is not None

    cancelled = store.cancel(job["job_id"])
    assert cancelled["state"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert store.complete(job["job_id"], {"summary": {"correct": 1}}, owner="worker-a") == "cancelled"
    assert store.get(job["job_id"])["result"] is None


def test_completed_job_is_not_overwritten_by_late_cancel(tmp_path: Any) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job, _ = store.create(
        {"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}], "concurrency": 1}
    )
    assert store.claim(owner="worker-a", lease_seconds=60, max_attempts=2) is not None
    assert store.complete(job["job_id"], {"summary": {"correct": 1}}, owner="worker-a") == "completed"

    late = store.cancel(job["job_id"])
    assert late["state"] == "completed"
    assert late["cancel_requested"] is False
    assert late["result"] == {"summary": {"correct": 1}}


def test_worker_failure_retries_only_until_max_attempts(tmp_path: Any) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job, _ = store.create(
        {"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}], "concurrency": 1}
    )
    assert store.claim(owner="worker-a", lease_seconds=60, max_attempts=2) is not None
    assert store.fail(job["job_id"], "synthetic", owner="worker-a", max_attempts=2) == "queued"
    assert store.claim(owner="worker-b", lease_seconds=60, max_attempts=2) is not None
    assert store.fail(job["job_id"], "synthetic", owner="worker-b", max_attempts=2) == "failed"


def test_api_token_is_enforced(tmp_path: Any) -> None:
    settings = _settings(tmp_path, api_token="secret-token")
    with TestClient(create_app(settings)) as client:
        evaluate_denied = client.post(
            "/v1/evaluate",
            json={"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}]},
        )
        denied = client.post(
            "/v1/jobs",
            json={"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}]},
        )
        allowed = client.post(
            "/v1/jobs",
            headers={"authorization": "Bearer secret-token"},
            json={"tasks": [{"id": "a", "prompt": "2+2", "gold": "4"}]},
        )
    assert evaluate_denied.status_code == 401
    assert denied.status_code == 401
    assert allowed.status_code == 202
