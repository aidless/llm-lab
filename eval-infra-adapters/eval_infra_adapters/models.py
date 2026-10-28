"""Small data models shared by the CLI, process manager and HTTP client."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BackendKind = Literal["vllm", "lmdeploy"]


class InfraError(RuntimeError):
    """Base error for launch, health and provider contract failures."""


class LaunchError(InfraError):
    """The serving process could not start or exited during startup."""


class ProviderError(InfraError):
    """The OpenAI-compatible HTTP endpoint failed its contract."""


@dataclass(frozen=True)
class LaunchConfig:
    backend: BackendKind
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    served_model_name: str | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in {"vllm", "lmdeploy"}:
            raise ValueError("backend must be vllm or lmdeploy")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.host.strip():
            raise ValueError("host must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.tensor_parallel_size <= 128:
            raise ValueError("tensor_parallel_size must be between 1 and 128")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderIdentity:
    backend: str
    base_url: str
    models: tuple[str, ...]
    health_status: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["models"] = list(self.models)
        return payload


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    backend: str
    base_url: str
    checks: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
