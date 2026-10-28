"""Minimal OpenAI-compatible endpoint checks for vLLM and LMDeploy."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .models import ProbeResult, ProviderError, ProviderIdentity

_MODEL_ERROR_MARKERS = (
    "model not found",
    "does not exist",
    "unknown model",
    "not supported",
)
_AUTH_ERROR_MARKERS = ("unauthorized", "forbidden", "api key", "token")
_OOM_ERROR_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda oom",
    "no available memory",
    "insufficient memory",
)


def classify_provider_error(message: str, status_code: int | None = None) -> str:
    lowered = message.lower()
    if any(marker in lowered for marker in _OOM_ERROR_MARKERS):
        return "oom"
    if status_code in {401, 403} or any(marker in lowered for marker in _AUTH_ERROR_MARKERS):
        return "auth"
    if status_code == 404 or any(marker in lowered for marker in _MODEL_ERROR_MARKERS):
        return "model_not_found"
    if status_code == 400:
        return "bad_request"
    if status_code == 429:
        return "rate_limited"
    if status_code is not None and status_code >= 500:
        return "server_error"
    return "provider_error"


class OpenAICompatibleClient:
    """Probe one OpenAI-compatible server without assuming a vendor SDK."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "local",
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
        health_path: str = "/health",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(follow_redirects=False)
        self.timeout_seconds = timeout_seconds
        self.health_path = "/" + health_path.lstrip("/")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def health(self) -> tuple[int, float]:
        started = time.perf_counter()
        try:
            response = self._client.get(
                f"{self.base_url}{self.health_path}",
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"health request failed: {exc}") from exc
        return response.status_code, (time.perf_counter() - started) * 1000.0

    def models(self) -> tuple[str, ...]:
        try:
            response = self._client.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"models request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(f"models endpoint returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
            rows = payload.get("data", [])
            model_ids = tuple(str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"invalid models response: {exc}") from exc
        if not model_ids:
            raise ProviderError("models endpoint returned no model identifiers")
        return model_ids

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int = 8,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"chat request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"chat transport failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            message = response.text[:1000]
            error_class = classify_provider_error(message, response.status_code)
            raise ProviderError(f"{error_class}: HTTP {response.status_code}: {message}")
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty assistant content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"invalid chat-completions response: {exc}") from exc
        return {
            "model": str(body.get("model") or model),
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "latency_ms": latency_ms,
            "usage": body.get("usage") or {},
        }

    def probe(
        self,
        *,
        backend: str,
        model: str | None = None,
        prompt: str = "Reply with OK.",
    ) -> ProbeResult:
        checks: dict[str, Any] = {}
        try:
            health_status, health_ms = self.health()
            checks["health"] = {
                "status": health_status,
                "ok": 200 <= health_status < 300,
                "latency_ms": health_ms,
            }
            if not 200 <= health_status < 300:
                raise ProviderError(f"health endpoint returned HTTP {health_status}")
            models = self.models()
            checks["models"] = list(models)
            selected_model = model or models[0]
            if selected_model not in models:
                raise ProviderError(f"model not found: {selected_model}; available={list(models)}")
            checks["chat"] = self.chat(model=selected_model, prompt=prompt)
            identity = ProviderIdentity(
                backend=backend,
                base_url=self.base_url,
                models=models,
                health_status=health_status,
                latency_ms=health_ms,
            )
            return ProbeResult(
                ok=True,
                backend=backend,
                base_url=self.base_url,
                checks={**checks, "identity": identity.to_dict()},
            )
        except ProviderError as exc:
            message = str(exc)
            return ProbeResult(
                ok=False,
                backend=backend,
                base_url=self.base_url,
                checks=checks,
                error=message,
                error_class=classify_provider_error(message),
            )
