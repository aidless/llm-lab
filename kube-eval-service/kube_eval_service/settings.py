"""Environment-only service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class SettingsError(ValueError):
    """Raised when the service is configured with unsafe or invalid values."""


@dataclass(frozen=True)
class Settings:
    mode: str
    upstream_base_url: str
    upstream_model: str
    api_key: str
    timeout_seconds: float
    max_tasks: int
    max_concurrency: int

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.getenv("EVAL_MODE", "deterministic").strip().lower()
        if mode not in {"deterministic", "openai-compatible"}:
            raise SettingsError("EVAL_MODE must be deterministic or openai-compatible")
        base_url = os.getenv("EVAL_UPSTREAM_BASE_URL", "").rstrip("/")
        model = os.getenv("EVAL_MODEL", "").strip()
        if mode == "openai-compatible":
            if not base_url or not model:
                raise SettingsError("openai-compatible mode requires EVAL_UPSTREAM_BASE_URL and EVAL_MODEL")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise SettingsError("EVAL_UPSTREAM_BASE_URL must be an http(s) URL")
        return cls(
            mode=mode,
            upstream_base_url=base_url,
            upstream_model=model,
            api_key=os.getenv("EVAL_API_KEY", "local"),
            timeout_seconds=float(os.getenv("EVAL_TIMEOUT_SECONDS", "30")),
            max_tasks=int(os.getenv("EVAL_MAX_TASKS", "500")),
            max_concurrency=int(os.getenv("EVAL_MAX_CONCURRENCY", "32")),
        )

    def validate_request(self, task_count: int, concurrency: int) -> None:
        if task_count < 1:
            raise SettingsError("at least one task is required")
        if task_count > self.max_tasks:
            raise SettingsError(f"task count exceeds limit {self.max_tasks}")
        if concurrency < 1 or concurrency > self.max_concurrency:
            raise SettingsError(f"concurrency must be between 1 and {self.max_concurrency}")
