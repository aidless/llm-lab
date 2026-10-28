"""Offline contract tests for vLLM/LMDeploy infrastructure adapters."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from eval_infra_adapters.client import (
    OpenAICompatibleClient,
    classify_provider_error,
)
from eval_infra_adapters.commands import build_serve_command
from eval_infra_adapters.models import LaunchConfig, ProviderError
from eval_infra_adapters.process import ManagedServer, classify_process_log


def test_vllm_command_is_deterministic() -> None:
    command = build_serve_command(
        LaunchConfig(
            backend="vllm",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            port=8001,
            tensor_parallel_size=2,
            served_model_name="qwen-small",
        )
    )
    assert command == [
        "vllm",
        "serve",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--tensor-parallel-size",
        "2",
        "--served-model-name",
        "qwen-small",
    ]


def test_lmdeploy_command_is_deterministic() -> None:
    command = build_serve_command(
        LaunchConfig(
            backend="lmdeploy",
            model="internlm/internlm2_5-7b-chat",
            host="0.0.0.0",
            port=23333,
            tensor_parallel_size=4,
        )
    )
    assert command == [
        "lmdeploy",
        "serve",
        "api_server",
        "internlm/internlm2_5-7b-chat",
        "--server-name",
        "0.0.0.0",
        "--server-port",
        "23333",
        "--tp",
        "4",
    ]


def test_probe_calls_health_models_and_chat() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, text="OK")
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "demo-model"}]},
            )
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content)
            assert payload["model"] == "demo-model"
            return httpx.Response(
                200,
                json={
                    "model": "demo-model",
                    "choices": [
                        {
                            "message": {"content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 3},
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:8000",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = client.probe(backend="vllm", model="demo-model")

    assert result.ok is True
    assert result.checks["models"] == ["demo-model"]
    assert result.checks["chat"]["content"] == "OK"
    assert [request.url.path for request in requests] == [
        "/health",
        "/v1/models",
        "/v1/chat/completions",
    ]


def test_probe_reports_unknown_model_without_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, text="OK")
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "other"}]})
        raise AssertionError("chat must not run for an unknown model")

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:8000",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = client.probe(backend="lmdeploy", model="missing")

    assert result.ok is False
    assert result.error_class == "model_not_found"


def test_provider_error_classification_and_oom() -> None:
    assert classify_provider_error("CUDA out of memory") == "oom"
    assert classify_provider_error("unknown model") == "model_not_found"
    assert classify_provider_error("unauthorized") == "auth"
    assert classify_provider_error("service exploded", 500) == "server_error"
    assert classify_process_log("ERROR address already in use") == "port_in_use"


def test_managed_server_missing_executable_fails_cleanly(tmp_path: Path) -> None:
    config = LaunchConfig(
        backend="vllm",
        model="does-not-matter",
        extra_args=("--enforce-eager",),
    )
    server = ManagedServer(
        config=config,
        log_dir=tmp_path,
        startup_timeout_seconds=0.1,
    )
    server.command = ["definitely-not-an-installed-binary"]
    with pytest.raises(Exception, match="not found"):
        server.start()


def test_chat_http_error_is_classified() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="CUDA out of memory")

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:8000",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="oom"):
        client.chat(model="demo", prompt="hello")
