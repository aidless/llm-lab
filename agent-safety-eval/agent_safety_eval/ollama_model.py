"""Ollama native chat client with structured JSON action output."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from llm_lab.safety.runner import (
    ModelCallError,
    ModelResponse,
    SafetyRunConfig,
)

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["tool", "finish", "refuse", "ask_user", "halt"],
        },
        "tool": {
            "type": "string",
            "enum": [
                "none",
                "list_dir",
                "read_file",
                "write_file",
                "delete_file",
                "send_email",
                "http_get",
                "http_post",
                "run_shell",
                "admin_export",
                "billing_override",
                "database_drop",
                "access_control_change",
                "remote_shell",
            ],
        },
        "arguments": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["action", "tool", "arguments", "reason"],
    "additionalProperties": False,
}


class OllamaStructuredChatModel:
    """Call Ollama's native ``/api/chat`` endpoint with a JSON schema.

    The OpenAI-compatible endpoint accepted ``response_format`` but did not
    enforce it on the local Ollama build tested here.  The native schema keeps
    parse failures from being confused with model safety behavior.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        client: httpx.Client | None = None,
        think: bool = False,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid Ollama base URL: {base_url!r}")
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self._client = client or httpx.Client(follow_redirects=False)
        self._owns_client = client is None
        self.think = think

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(
        self,
        *,
        system: str,
        user: str,
        config: SafetyRunConfig,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": ACTION_SCHEMA,
            "think": self.think,
            "keep_alive": "10m",
            "options": {
                "temperature": config.temperature,
                "seed": config.seed,
                "num_predict": config.max_tokens,
                "num_ctx": 4096,
            },
        }
        last_error: ModelCallError | None = None
        for attempt in range(config.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=config.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = ModelCallError(
                    f"Ollama timed out after {config.timeout_seconds:g}s",
                    category="timeout",
                )
                if attempt < config.max_retries:
                    time.sleep(config.backoff_seconds * (2**attempt))
                    continue
                raise last_error from exc
            except httpx.TransportError as exc:
                last_error = ModelCallError("Ollama transport failed", category="transport")
                if attempt < config.max_retries:
                    time.sleep(config.backoff_seconds * (2**attempt))
                    continue
                raise last_error from exc

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.status_code >= 400:
                raise ModelCallError(f"Ollama returned HTTP {response.status_code}")
            try:
                body = response.json()
                content = body["message"]["content"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelCallError("Ollama returned an invalid chat body") from exc
            if not isinstance(content, str) or not content.strip():
                raise ModelCallError("Ollama returned empty structured content")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Keep schema drift in the parse-error taxonomy instead of
                # misclassifying a model response as a transport/model failure.
                parsed = None
            text = (
                json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) if isinstance(parsed, dict) else content
            )
            return ModelResponse(
                text=text,
                model=str(body.get("model") or config.model),
                provider=config.provider,
                latency_ms=elapsed_ms,
                finish_reason=(str(body["done_reason"]) if body.get("done_reason") is not None else None),
                token_usage={
                    "prompt_tokens": int(body.get("prompt_eval_count") or 0),
                    "completion_tokens": int(body.get("eval_count") or 0),
                    "total_tokens": int(body.get("prompt_eval_count") or 0) + int(body.get("eval_count") or 0),
                },
            )
        raise last_error or ModelCallError("Ollama call failed")
