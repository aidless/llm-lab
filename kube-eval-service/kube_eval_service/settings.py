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
    role: str = "api"
    database_path: str = "jobs.sqlite3"
    api_token: str = ""
    max_attempts: int = 2
    lease_seconds: float = 30.0
    task_delay_seconds: float = 0.0

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.getenv("EVAL_MODE", "deterministic").strip().lower()
        if mode not in {"deterministic", "openai-compatible"}:
            raise SettingsError("EVAL_MODE must be deterministic or openai-compatible")
        role = os.getenv("EVAL_ROLE", "api").strip().lower()
        if role not in {"api", "worker"}:
            raise SettingsError("EVAL_ROLE must be api or worker")
        base_url = os.getenv("EVAL_UPSTREAM_BASE_URL", "").rstrip("/")
        model = os.getenv("EVAL_MODEL", "").strip()
        if mode == "openai-compatible":
            if not base_url or not model:
                raise SettingsError("openai-compatible mode requires EVAL_UPSTREAM_BASE_URL and EVAL_MODEL")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise SettingsError("EVAL_UPSTREAM_BASE_URL must be an http(s) URL")
        max_attempts = int(os.getenv("EVAL_MAX_ATTEMPTS", "2"))
        lease_seconds = float(os.getenv("EVAL_LEASE_SECONDS", "30"))
        task_delay_seconds = float(os.getenv("EVAL_TASK_DELAY_SECONDS", "0"))
        if max_attempts < 1:
            raise SettingsError("EVAL_MAX_ATTEMPTS must be at least 1")
        if lease_seconds <= 0:
            raise SettingsError("EVAL_LEASE_SECONDS must be positive")
        if task_delay_seconds < 0:
            raise SettingsError("EVAL_TASK_DELAY_SECONDS cannot be negative")
        return cls(
            mode=mode,
            upstream_base_url=base_url,
            upstream_model=model,
            api_key=os.getenv("EVAL_API_KEY", "local"),
            timeout_seconds=float(os.getenv("EVAL_TIMEOUT_SECONDS", "30")),
            max_tasks=int(os.getenv("EVAL_MAX_TASKS", "500")),
            max_concurrency=int(os.getenv("EVAL_MAX_CONCURRENCY", "32")),
            role=role,
            database_path=os.getenv("EVAL_DATABASE_PATH", "jobs.sqlite3").strip(),
            api_token=os.getenv("EVAL_API_TOKEN", "").strip(),
            max_attempts=max_attempts,
            lease_seconds=lease_seconds,
            task_delay_seconds=task_delay_seconds,
        )

    def validate_request(self, task_count: int, concurrency: int) -> None:
        if task_count < 1:
            raise SettingsError("at least one task is required")
        if task_count > self.max_tasks:
            raise SettingsError(f"task count exceeds limit {self.max_tasks}")
        if concurrency < 1 or concurrency > self.max_concurrency:
            raise SettingsError(f"concurrency must be between 1 and {self.max_concurrency}")
