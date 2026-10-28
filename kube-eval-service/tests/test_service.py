"""API tests for the Kubernetes evaluation service."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from kube_eval_service.app import EvalTask, create_app
from kube_eval_service.settings import Settings


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "mode": "deterministic",
        "upstream_base_url": "",
        "upstream_model": "",
        "api_key": "local",
        "timeout_seconds": 5.0,
        "max_tasks": 20,
        "max_concurrency": 8,
    }
    values.update(overrides)
    return Settings(**values)


def test_health_readiness_metrics_and_evaluation() -> None:
    with TestClient(create_app(_settings())) as client:
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


def test_request_limits_reject_oversized_or_excessive_requests() -> None:
    with TestClient(create_app(_settings(max_tasks=1))) as client:
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


def test_task_failure_is_isolated() -> None:
    def predictor(task: EvalTask, _settings: Settings) -> str:
        if task.id == "bad":
            raise RuntimeError("synthetic failure")
        return task.gold

    with TestClient(create_app(_settings(), predictor=predictor)) as client:
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


def test_openai_mode_requires_fixed_upstream() -> None:
    missing = Settings(
        mode="openai-compatible",
        upstream_base_url="",
        upstream_model="",
        api_key="local",
        timeout_seconds=5.0,
        max_tasks=10,
        max_concurrency=4,
    )
    with TestClient(create_app(missing)) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
