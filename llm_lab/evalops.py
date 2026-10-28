"""Auditable batch-evaluation primitives for deterministic and local models.

This module deliberately keeps the evaluation contract small:

* a frozen task set with a content hash;
* an adapter boundary that separates provider failures from answer-format
  failures;
* a resumable JSONL record store;
* a report that makes strict accuracy and parsed-answer accuracy explicit;
* a compatibility export for OpenCompass, without claiming that the external
  runner was executed.

The C-Eval subset builder is optional and imports pyarrow lazily.  The rest of
the module only needs the dependencies already used by ``llm-lab``.
"""

from __future__ import annotations

import contextlib
import html
import json
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlparse

import httpx

FailureCategory = Literal[
    "none",
    "timeout",
    "transport",
    "model_error",
    "parse_error",
    "evaluator_error",
]

_LETTERS = ("A", "B", "C", "D")
_RETRYABLE = {"timeout", "transport"}
_CEVAL_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"
_CEVAL_SOURCE_URL = "https://huggingface.co/datasets/ceval/ceval-exam"
_CEVAL_LICENSE = "CC-BY-NC-SA-4.0"


class EvalOpsError(ValueError):
    """Base class for validation and evaluation-contract errors."""


class ChoiceParseError(EvalOpsError):
    """The model returned text that is not a usable single-letter answer."""


class AdapterError(RuntimeError):
    """A provider boundary failed before an answer could be parsed."""

    category: FailureCategory = "model_error"


class AdapterTimeoutError(AdapterError):
    category: FailureCategory = "timeout"


class AdapterTransportError(AdapterError):
    category: FailureCategory = "transport"


class AdapterModelError(AdapterError):
    category: FailureCategory = "model_error"


class EvaluatorError(EvalOpsError):
    """The task's gold answer or evaluator contract is invalid."""


@dataclass(frozen=True)
class EvalTask:
    """One frozen multiple-choice task."""

    id: str
    subject: str
    question: str
    choices: dict[str, str]
    gold: str
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.subject:
            raise EvalOpsError("task id and subject must be non-empty")
        if self.gold not in _LETTERS:
            raise EvalOpsError(f"invalid gold answer {self.gold!r} for {self.id}")
        if set(self.choices) != set(_LETTERS):
            raise EvalOpsError(f"task {self.id} must contain choices A-D")
        if any(not str(self.choices[key]).strip() for key in _LETTERS):
            raise EvalOpsError(f"task {self.id} contains an empty choice")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "question": self.question,
            "choices": {key: self.choices[key] for key in _LETTERS},
            "gold": self.gold,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalTask:
        try:
            task_id = str(payload["id"])
            subject = str(payload["subject"])
            question = str(payload["question"])
            gold = str(payload["gold"])
            raw_choices = payload["choices"]
        except KeyError as exc:
            raise EvalOpsError(f"task is missing required field: {exc.args[0]}") from exc
        if not isinstance(raw_choices, dict):
            raise EvalOpsError(f"task {task_id} choices must be a mapping")
        choices = {str(key): str(value) for key, value in raw_choices.items()}
        source = payload.get("source")
        if source is None:
            source = {}
        if not isinstance(source, dict):
            raise EvalOpsError(f"task {task_id} source must be a mapping")
        return cls(
            id=task_id,
            subject=subject,
            question=question,
            choices=choices,
            gold=gold,
            source=source,
        )

    def prompt(self) -> str:
        """Return the exact prompt recorded in the frozen task file."""
        return self.question

    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt())


@dataclass(frozen=True)
class EvalConfig:
    """Parameters that affect the evaluation run."""

    provider: str = "ollama"
    model: str = "qwen25-3b-16k:latest"
    concurrency: int = 1
    timeout_seconds: float = 60.0
    max_retries: int = 2
    backoff_seconds: float = 0.25
    temperature: float = 0.0
    max_tokens: int = 8

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise EvalOpsError("provider and model must be non-empty")
        if not 1 <= self.concurrency <= 64:
            raise EvalOpsError("concurrency must be between 1 and 64")
        if self.timeout_seconds <= 0:
            raise EvalOpsError("timeout_seconds must be positive")
        if not 0 <= self.max_retries <= 10:
            raise EvalOpsError("max_retries must be between 0 and 10")
        if self.backoff_seconds < 0:
            raise EvalOpsError("backoff_seconds must be non-negative")
        if self.max_tokens < 1:
            raise EvalOpsError("max_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


@dataclass
class ProviderResponse:
    """Raw provider output plus metadata safe to persist in a record."""

    text: str
    model: str
    provider: str
    latency_ms: float
    finish_reason: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class EvalRecord:
    """One final task result. Timings are intentionally excluded from answer_hash."""

    task_id: str
    subject: str
    provider: str
    model: str
    prompt_sha256: str
    config_hash: str
    output: str
    parsed_choice: str | None
    gold: str
    correct: bool | None
    status: Literal["success", "failure"]
    failure_category: FailureCategory
    attempts: int
    attempt_latencies_ms: list[float]
    latency_ms: float
    finish_reason: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "subject": self.subject,
            "provider": self.provider,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "config_hash": self.config_hash,
            "output": self.output,
            "parsed_choice": self.parsed_choice,
            "gold": self.gold,
            "correct": self.correct,
            "status": self.status,
            "failure_category": self.failure_category,
            "attempts": self.attempts,
            "attempt_latencies_ms": self.attempt_latencies_ms,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "token_usage": self.token_usage,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalRecord:
        try:
            return cls(
                task_id=str(payload["task_id"]),
                subject=str(payload["subject"]),
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                prompt_sha256=str(payload["prompt_sha256"]),
                config_hash=str(payload["config_hash"]),
                output=str(payload.get("output", "")),
                parsed_choice=_optional_string(payload.get("parsed_choice")),
                gold=str(payload["gold"]),
                correct=_optional_bool(payload.get("correct")),
                status=cast(
                    Literal["success", "failure"],
                    _checked_literal(payload["status"], {"success", "failure"}, "status"),
                ),
                failure_category=cast(
                    FailureCategory,
                    _checked_literal(
                        payload.get("failure_category", "model_error"),
                        {"none", "timeout", "transport", "model_error", "parse_error", "evaluator_error"},
                        "failure_category",
                    ),
                ),
                attempts=int(payload.get("attempts", 1)),
                attempt_latencies_ms=[float(value) for value in payload.get("attempt_latencies_ms", [])],
                latency_ms=float(payload.get("latency_ms", 0.0)),
                finish_reason=_optional_string(payload.get("finish_reason")),
                token_usage={
                    str(key): int(value)
                    for key, value in dict(payload.get("token_usage") or {}).items()
                },
                error=_optional_string(payload.get("error")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvalOpsError(f"invalid evaluation record: {exc}") from exc


class Adapter(Protocol):
    """Provider boundary used by the batch executor."""

    def complete(self, task: EvalTask, config: EvalConfig) -> ProviderResponse:
        """Return a provider response or raise an AdapterError subclass."""


class DeterministicAdapter:
    """No-network fixture adapter, useful for tests and CI smoke runs.

    It is explicitly *not* a model-quality measurement.  The default answer is
    a hash of task id and fixture name; tests can supply an answer map.
    """

    def __init__(
        self,
        model: str = "deterministic-fixture-v1",
        answer_by_id: dict[str, str] | None = None,
        *,
        latency_ms: float = 1.0,
    ) -> None:
        self.model = model
        self.provider = "deterministic"
        self.answer_by_id = dict(answer_by_id or {})
        self.latency_ms = max(0.0, latency_ms)

    def complete(self, task: EvalTask, _config: EvalConfig) -> ProviderResponse:
        answer = self.answer_by_id.get(task.id)
        if answer is None:
            digest = sha256_text(f"{self.model}:{task.id}")
            answer = _LETTERS[int(digest[:8], 16) % len(_LETTERS)]
        return ProviderResponse(
            text=answer,
            model=self.model,
            provider=self.provider,
            latency_ms=self.latency_ms,
            finish_reason="stop",
            token_usage={},
        )


class OpenAICompatibleAdapter:
    """Minimal OpenAI-compatible chat-completions adapter.

    Ollama and vLLM can both expose this endpoint.  This class only proves the
    HTTP contract; it does not claim that either backend was exercised.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        provider: str = "openai-compatible",
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EvalOpsError(f"invalid base_url: {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        # Local providers conventionally accept a non-secret placeholder.
        self.api_key = api_key or "ollama"
        self._client = client or httpx.Client(follow_redirects=False)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(self, task: EvalTask, config: EvalConfig) -> ProviderResponse:
        started = time.perf_counter()
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": task.prompt()}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        try:
            response = self._client.post(
                self._endpoint(),
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise AdapterTimeoutError(f"provider timed out after {config.timeout_seconds:g}s") from exc
        except httpx.TransportError as exc:
            raise AdapterTransportError("provider transport failed") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise AdapterModelError(f"provider returned HTTP {response.status_code}")
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            text = str(message.get("content") or "")
            finish_reason = _optional_string(choice.get("finish_reason"))
            usage_raw = body.get("usage") or {}
            usage = {
                str(key): int(value)
                for key, value in dict(usage_raw).items()
                if isinstance(value, (int, float))
            }
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AdapterModelError("provider returned an invalid chat-completions body") from exc
        if not text.strip():
            raise AdapterModelError("provider returned empty content")
        return ProviderResponse(
            text=text,
            model=str(body.get("model") or config.model),
            provider=self.provider,
            latency_ms=elapsed_ms,
            finish_reason=finish_reason,
            token_usage=usage,
        )


class FaultInjectionAdapter:
    """Wrap an adapter with deterministic timeout/transport/format faults.

    ``faults`` maps task ids to ``timeout``, ``transport``, ``model_error`` or
    ``malformed``.  ``fail_times`` can be used to fail only the first N attempts
    for retry tests; when omitted, the fault persists.
    """

    def __init__(
        self,
        delegate: Adapter,
        faults: dict[str, str],
        *,
        fail_times: dict[str, int] | None = None,
    ) -> None:
        self._delegate = delegate
        self._faults = dict(faults)
        self._fail_times = dict(fail_times or {})
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(self, task: EvalTask, config: EvalConfig) -> ProviderResponse:
        with self._lock:
            count = self._counts.get(task.id, 0) + 1
            self._counts[task.id] = count
        fault = self._faults.get(task.id)
        limit = self._fail_times.get(task.id)
        if fault and (limit is None or count <= limit):
            if fault == "timeout":
                raise AdapterTimeoutError("injected timeout")
            if fault == "transport":
                raise AdapterTransportError("injected transport failure")
            if fault == "model_error":
                raise AdapterModelError("injected model error")
            if fault == "malformed":
                return ProviderResponse(
                    text="NOT_A_CHOICE",
                    model="fault-injected",
                    provider="fault-injection",
                    latency_ms=0.0,
                    finish_reason="stop",
                )
            raise EvalOpsError(f"unknown fault type: {fault!r}")
        return self._delegate.complete(task, config)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def sha256_path(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _checked_literal(value: Any, allowed: set[str], field_name: str) -> str:
    text = str(value)
    if text not in allowed:
        raise EvalOpsError(f"invalid {field_name}: {text!r}")
    return text


def load_tasks(path: str | Path) -> list[EvalTask]:
    """Load a JSONL task set and reject duplicates or malformed rows."""
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            task = EvalTask.from_dict(payload)
        except (json.JSONDecodeError, EvalOpsError) as exc:
            raise EvalOpsError(f"{path}:{line_number}: {exc}") from exc
        if task.id in seen:
            raise EvalOpsError(f"{path}:{line_number}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise EvalOpsError(f"task set is empty: {path}")
    return tasks


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def read_records(path: str | Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    target = Path(path)
    if not target.exists():
        return records
    seen: set[str] = set()
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = EvalRecord.from_dict(json.loads(line))
        except (json.JSONDecodeError, EvalOpsError) as exc:
            raise EvalOpsError(f"{target}:{line_number}: {exc}") from exc
        if record.task_id in seen:
            raise EvalOpsError(f"{target}:{line_number}: duplicate record for {record.task_id!r}")
        seen.add(record.task_id)
        records.append(record)
    return records


def parse_choice(text: str, choices: dict[str, str] | None = None) -> str:
    """Parse a single letter without treating arbitrary explanation text as an answer."""
    clean = text.strip()
    if not clean:
        raise ChoiceParseError("empty model output")
    direct = re.fullmatch(r"[A-D]", clean, flags=re.IGNORECASE)
    if direct:
        choice = direct.group(0).upper()
        if choices is not None and choice not in choices:
            raise ChoiceParseError(f"choice {choice} is not valid for this task")
        return choice

    # Accept common answer prefixes, but never scan arbitrary prose for a
    # stray letter: an explanation is not an answer.
    patterns = (
        r"^(?:final\s+answer|answer|正确答案|答案)\s*[:：]?\s*[\(\[]?([A-D])[\)\]]?\s*[。.]?\s*$",
        r"^[\(\[]?([A-D])[\)\]]?\s*[。.]?\s*$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, clean, flags=re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choices is not None and choice not in choices:
                raise ChoiceParseError(f"choice {choice} is not valid for this task")
            return choice
    raise ChoiceParseError("expected exactly one answer letter A-D")


def evaluate_choice(task: EvalTask, parsed_choice: str) -> bool:
    if task.gold not in task.choices:
        raise EvaluatorError(f"gold answer {task.gold!r} is absent from choices")
    if parsed_choice not in task.choices:
        raise EvaluatorError(f"parsed answer {parsed_choice!r} is absent from choices")
    return parsed_choice == task.gold


def _failure_record(
    task: EvalTask,
    config: EvalConfig,
    *,
    output: str,
    parsed_choice: str | None,
    category: FailureCategory,
    attempts: int,
    attempt_latencies_ms: list[float],
    error: str,
) -> EvalRecord:
    return EvalRecord(
        task_id=task.id,
        subject=task.subject,
        provider=config.provider,
        model=config.model,
        prompt_sha256=task.prompt_sha256(),
        config_hash=config.config_hash(),
        output=output,
        parsed_choice=parsed_choice,
        gold=task.gold,
        correct=None,
        status="failure",
        failure_category=category,
        attempts=attempts,
        attempt_latencies_ms=attempt_latencies_ms,
        latency_ms=sum(attempt_latencies_ms),
        error=error,
    )


def run_one(task: EvalTask, adapter: Adapter, config: EvalConfig) -> EvalRecord:
    """Execute one task with bounded retries and explicit failure classification."""
    attempt_latencies: list[float] = []
    total_attempts = config.max_retries + 1
    last_error = ""
    last_category: FailureCategory = "model_error"
    last_output = ""
    for attempt in range(1, total_attempts + 1):
        started = time.perf_counter()
        try:
            response = adapter.complete(task, config)
        except AdapterError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            attempt_latencies.append(elapsed)
            last_category = exc.category
            last_error = str(exc)
            if exc.category in _RETRYABLE and attempt < total_attempts:
                if config.backoff_seconds:
                    time.sleep(config.backoff_seconds * (2 ** (attempt - 1)))
                continue
            return _failure_record(
                task,
                config,
                output="",
                parsed_choice=None,
                category=exc.category,
                attempts=attempt,
                attempt_latencies_ms=attempt_latencies,
                error=last_error,
            )
        except Exception as exc:  # defensive adapter boundary
            elapsed = (time.perf_counter() - started) * 1000.0
            attempt_latencies.append(elapsed)
            last_category = "model_error"
            last_error = f"unexpected adapter error: {exc}"
            return _failure_record(
                task,
                config,
                output="",
                parsed_choice=None,
                category=last_category,
                attempts=attempt,
                attempt_latencies_ms=attempt_latencies,
                error=last_error,
            )

        elapsed = (time.perf_counter() - started) * 1000.0
        attempt_latencies.append(elapsed)
        last_output = response.text
        try:
            parsed = parse_choice(response.text, task.choices)
        except ChoiceParseError as exc:
            return _failure_record(
                task,
                config,
                output=response.text,
                parsed_choice=None,
                category="parse_error",
                attempts=attempt,
                attempt_latencies_ms=attempt_latencies,
                error=str(exc),
            )
        try:
            correct = evaluate_choice(task, parsed)
        except EvaluatorError as exc:
            return _failure_record(
                task,
                config,
                output=response.text,
                parsed_choice=parsed,
                category="evaluator_error",
                attempts=attempt,
                attempt_latencies_ms=attempt_latencies,
                error=str(exc),
            )
        return EvalRecord(
            task_id=task.id,
            subject=task.subject,
            provider=config.provider,
            model=config.model,
            prompt_sha256=task.prompt_sha256(),
            config_hash=config.config_hash(),
            output=response.text,
            parsed_choice=parsed,
            gold=task.gold,
            correct=correct,
            status="success",
            failure_category="none",
            attempts=attempt,
            attempt_latencies_ms=attempt_latencies,
            latency_ms=sum(attempt_latencies),
            finish_reason=response.finish_reason,
            token_usage=response.token_usage,
        )

    # The loop always returns, but keep a typed fallback for future edits.
    return _failure_record(
        task,
        config,
        output=last_output,
        parsed_choice=None,
        category=last_category,
        attempts=total_attempts,
        attempt_latencies_ms=attempt_latencies,
        error=last_error or "evaluation exhausted retries",
    )


def answer_hash(records: Iterable[EvalRecord]) -> str:
    """Hash only answer-relevant fields, excluding timing and run metadata."""
    payload = [
        {
            "task_id": record.task_id,
            "output": record.output,
            "parsed_choice": record.parsed_choice,
            "status": record.status,
            "failure_category": record.failure_category,
        }
        for record in sorted(records, key=lambda item: item.task_id)
    ]
    return sha256_text(canonical_json(payload))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_records(
    records: list[EvalRecord],
    *,
    config: EvalConfig,
    task_set_hash: str,
    wall_seconds: float,
    run_label: str | None = None,
) -> dict[str, Any]:
    """Build a report whose denominators are explicit and recomputable."""
    total = len(records)
    successes = [record for record in records if record.status == "success"]
    parsed = [record for record in successes if record.parsed_choice is not None]
    correct = sum(1 for record in successes if record.correct is True)
    failures: dict[str, int] = {}
    for record in records:
        if record.status != "success":
            failures[record.failure_category] = failures.get(record.failure_category, 0) + 1

    subject_metrics: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = subject_metrics.setdefault(
            record.subject,
            {
                "total": 0,
                "success": 0,
                "correct": 0,
                "accuracy_parsed": 0.0,
                "strict_accuracy": 0.0,
            },
        )
        entry["total"] += 1
        if record.status == "success":
            entry["success"] += 1
            if record.correct is True:
                entry["correct"] += 1
    for entry in subject_metrics.values():
        entry["accuracy_parsed"] = entry["correct"] / entry["success"] if entry["success"] else 0.0
        entry["strict_accuracy"] = entry["correct"] / entry["total"] if entry["total"] else 0.0

    latencies = [record.latency_ms for record in records]
    success_latencies = [record.latency_ms for record in successes]
    usage: dict[str, int] = {}
    for record in successes:
        for key, value in record.token_usage.items():
            usage[key] = usage.get(key, 0) + value

    return {
        "schema_version": "evalops.report.v1",
        "run_label": run_label,
        "provider": config.provider,
        "model": config.model,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "task_set_hash": task_set_hash,
        "answer_hash": answer_hash(records),
        "counts": {
            "total": total,
            "success": len(successes),
            "parsed": len(parsed),
            "correct": correct,
            "failures": failures,
        },
        "rates": {
            "success_rate": len(successes) / total if total else 0.0,
            "parsed_rate": len(parsed) / total if total else 0.0,
            "accuracy_parsed": correct / len(parsed) if parsed else 0.0,
            "strict_accuracy": correct / total if total else 0.0,
        },
        "latency_ms": {
            "all": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "mean": sum(latencies) / total if total else 0.0,
            },
            "successful": {
                "p50": _percentile(success_latencies, 0.50),
                "p95": _percentile(success_latencies, 0.95),
                "mean": sum(success_latencies) / len(success_latencies) if success_latencies else 0.0,
            },
        },
        "throughput_tasks_per_second": total / wall_seconds if wall_seconds > 0 else 0.0,
        "wall_seconds": wall_seconds,
        "retries_total": sum(max(0, record.attempts - 1) for record in records),
        "token_usage": usage,
        "subject_metrics": dict(sorted(subject_metrics.items())),
        "limitations": [
            "This is a local, non-production run.",
            "The public C-Eval subset is not the official C-Eval leaderboard.",
            "Parsed accuracy excludes parse/provider failures; strict accuracy includes them in the denominator.",
            "Deterministic adapters are fixtures, not model-quality evidence.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    counts = report["counts"]
    rates = report["rates"]
    latency = report["latency_ms"]
    lines = [
        "# EvalOps report",
        "",
        f"- Model: `{report['model']}`",
        f"- Provider: `{report['provider']}`",
        f"- Tasks: {counts['total']}",
        f"- Concurrency: {report['config']['concurrency']}",
        f"- Task-set SHA-256: `{report['task_set_hash']}`",
        f"- Config SHA-256: `{report['config_hash']}`",
        f"- Answer SHA-256: `{report['answer_hash']}`",
        "",
        "## Outcome",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Successful provider responses | {counts['success']} |",
        f"| Parsed answers | {counts['parsed']} |",
        f"| Correct answers | {counts['correct']} |",
        f"| Success rate | {rates['success_rate']:.2%} |",
        f"| Parsed rate | {rates['parsed_rate']:.2%} |",
        f"| Accuracy among parsed answers | {rates['accuracy_parsed']:.2%} |",
        f"| Strict accuracy (all tasks) | {rates['strict_accuracy']:.2%} |",
        f"| Retry attempts | {report['retries_total']} |",
        f"| Wall time | {report['wall_seconds']:.2f}s |",
        f"| Throughput | {report['throughput_tasks_per_second']:.3f} tasks/s |",
        "",
        "## Latency",
        "",
        "| Set | p50 | p95 | Mean |",
        "|---|---:|---:|---:|",
        f"| All final records | {latency['all']['p50']:.2f} ms | {latency['all']['p95']:.2f} ms | {latency['all']['mean']:.2f} ms |",
        f"| Successful records | {latency['successful']['p50']:.2f} ms | {latency['successful']['p95']:.2f} ms | {latency['successful']['mean']:.2f} ms |",
        "",
        "## Failures",
        "",
    ]
    if counts["failures"]:
        lines.extend(["| Category | Count |", "|---|---:|"])
        lines.extend(f"| `{category}` | {count} |" for category, count in sorted(counts["failures"].items()))
    else:
        lines.append("No failures recorded.")
    lines.extend(
        [
            "",
            "## By subject",
            "",
            "| Subject | Total | Success | Correct | Parsed accuracy | Strict accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for subject, row in report["subject_metrics"].items():
        lines.append(
            f"| {subject} | {row['total']} | {row['success']} | {row['correct']} | "
            f"{row['accuracy_parsed']:.2%} | {row['strict_accuracy']:.2%} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def html_report(report: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(subject)}</td>"
        f"<td>{row['total']}</td>"
        f"<td>{row['success']}</td>"
        f"<td>{row['correct']}</td>"
        f"<td>{row['accuracy_parsed']:.2%}</td>"
        f"<td>{row['strict_accuracy']:.2%}</td>"
        "</tr>"
        for subject, row in report["subject_metrics"].items()
    )
    failures = "".join(
        f"<li><code>{html.escape(category)}</code>: {count}</li>"
        for category, count in sorted(report["counts"]["failures"].items())
    ) or "<li>none</li>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>EvalOps report</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#17202a}}
code{{background:#eef2f5;padding:.15rem .3rem;border-radius:3px}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd5dd;padding:.4rem .55rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}
h1,h2{{color:#123b56}}li{{margin:.25rem 0}}</style></head>
<body><h1>EvalOps report</h1>
<p><strong>Model:</strong> <code>{html.escape(report['model'])}</code> ·
<strong>Tasks:</strong> {report['counts']['total']} ·
<strong>Concurrency:</strong> {report['config']['concurrency']}</p>
<h2>Hashes</h2><ul><li>Task set: <code>{html.escape(report['task_set_hash'])}</code></li>
<li>Config: <code>{html.escape(report['config_hash'])}</code></li>
<li>Answers: <code>{html.escape(report['answer_hash'])}</code></li></ul>
<h2>Outcome</h2><ul><li>Success rate: {report['rates']['success_rate']:.2%}</li>
<li>Parsed rate: {report['rates']['parsed_rate']:.2%}</li>
<li>Accuracy among parsed: {report['rates']['accuracy_parsed']:.2%}</li>
<li>Strict accuracy: {report['rates']['strict_accuracy']:.2%}</li></ul>
<h2>Failures</h2><ul>{failures}</ul>
<h2>By subject</h2><table><thead><tr><th>Subject</th><th>Total</th><th>Success</th><th>Correct</th>
<th>Parsed accuracy</th><th>Strict accuracy</th></tr></thead><tbody>{rows}</tbody></table>
<p><em>Local subset result, not an official C-Eval leaderboard result.</em></p></body></html>
"""


def _write_report_files(output_dir: Path, report: dict[str, Any]) -> None:
    write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    (output_dir / "report.html").write_text(html_report(report), encoding="utf-8")


def run_eval(
    tasks: list[EvalTask],
    adapter: Adapter,
    config: EvalConfig,
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    run_label: str | None = None,
    task_set_hash: str | None = None,
    close_adapter: bool = True,
) -> dict[str, Any]:
    """Run a task set with bounded concurrency and crash-safe record updates."""
    if len({task.id for task in tasks}) != len(tasks):
        raise EvalOpsError("task ids must be unique")
    if task_set_hash is None:
        task_set_hash = sha256_text(canonical_json([task.to_dict() for task in tasks]))
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        records_path = output / "records.jsonl"
        if resume:
            meta_path = output / "run-meta.json"
            if not meta_path.exists():
                raise EvalOpsError(f"cannot resume without run metadata: {meta_path}")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise EvalOpsError(f"invalid run metadata: {meta_path}: {exc}") from exc
            if meta.get("config_hash") != config.config_hash():
                raise EvalOpsError(
                    "cannot resume: config hash does not match the existing run"
                )
            if meta.get("task_set_hash") != task_set_hash:
                raise EvalOpsError(
                    "cannot resume: task-set hash does not match the existing run"
                )
            if meta.get("task_count") != len(tasks):
                raise EvalOpsError(
                    "cannot resume: task count does not match the existing run"
                )
            existing = read_records(records_path)
            tasks_by_id = {task.id: task for task in tasks}
            for record in existing:
                task = tasks_by_id.get(record.task_id)
                if task is None:
                    raise EvalOpsError(
                        f"cannot resume: record task {record.task_id!r} is absent from the task set"
                    )
                if record.config_hash != config.config_hash():
                    raise EvalOpsError(
                        f"cannot resume: record {record.task_id!r} has a different config hash"
                    )
                if record.prompt_sha256 != task.prompt_sha256():
                    raise EvalOpsError(
                        f"cannot resume: record {record.task_id!r} has a different prompt hash"
                    )
        else:
            existing = []
        if not resume:
            write_jsonl(records_path, [])
    else:
        existing = []
        records_path = None

    existing_by_id = {record.task_id: record for record in existing}
    runnable: list[EvalTask] = []
    for task in tasks:
        if task.id in existing_by_id:
            prior = existing_by_id[task.id]
            if not retry_failed or prior.status == "success":
                continue
        runnable.append(task)

    started = time.perf_counter()
    current_by_id = dict(existing_by_id)
    task_rank = {task.id: index for index, task in enumerate(tasks)}
    if records_path is not None and output is not None:
        write_json(
            output / "run-meta.json",
            {
                "schema_version": "evalops.run.v1",
                "run_label": run_label,
                "config": config.to_dict(),
                "config_hash": config.config_hash(),
                "task_set_hash": task_set_hash,
                "task_count": len(tasks),
                "resume": resume,
                "retry_failed": retry_failed,
            },
        )

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {executor.submit(run_one, task, adapter, config): task for task in runnable}
        for future in as_completed(futures):
            record = future.result()
            if records_path is not None:
                current_by_id[record.task_id] = record
                ordered = [
                    current_by_id[task.id]
                    for task in tasks
                    if task.id in current_by_id
                ]
                write_jsonl(records_path, (item.to_dict() for item in ordered))
            else:
                current_by_id[record.task_id] = record

    wall_seconds = time.perf_counter() - started
    ordered_records = [
        current_by_id[task.id]
        for task in sorted(tasks, key=lambda item: task_rank[item.id])
        if task.id in current_by_id
    ]
    report = summarize_records(
        ordered_records,
        config=config,
        task_set_hash=task_set_hash,
        wall_seconds=wall_seconds,
        run_label=run_label,
    )
    if output is not None:
        _write_report_files(output, report)
    if close_adapter:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    return report


def _split_rendered_question(rendered: str) -> tuple[str, dict[str, str]]:
    """Split the source's rendered question into stem and four options."""
    lines = rendered.splitlines()
    stem_lines: list[str] = []
    choices: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^([A-D])\.\s*(.+?)\s*$", line)
        if match and not choices or (match and match.group(1) not in choices):
            choices[match.group(1)] = match.group(2)
        elif not choices:
            stem_lines.append(line)
    if set(choices) != set(_LETTERS):
        raise EvalOpsError(f"could not parse choices A-D from rendered question: {rendered[:80]!r}")
    return "\n".join(stem_lines).strip(), choices


def build_dataset_from_records(
    source_records: list[dict[str, Any]],
    parquet_rows_by_subject: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path,
    source_metadata: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Validate source rows against parquet rows and freeze a task set.

    This pure builder is used by tests and by the C-Eval-specific wrapper below.
    It requires exact question, option, and gold matches before writing files.
    """
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    by_subject: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    for subject, rows in parquet_rows_by_subject.items():
        by_subject[subject] = {
            str(row.get("question", "")).strip(): (index, row) for index, row in enumerate(rows)
        }

    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    for index, source in enumerate(source_records):
        if not isinstance(source, dict):
            raise EvalOpsError(f"source record {index} is not an object")
        task_id = str(source.get("id", ""))
        subject = str(source.get("group", ""))
        rendered = str(source.get("question", ""))
        gold = str(source.get("gold", ""))
        if not task_id or task_id in seen_ids:
            raise EvalOpsError(f"missing or duplicate source id at row {index}: {task_id!r}")
        seen_ids.add(task_id)
        stem, choices = _split_rendered_question(rendered)
        candidates = by_subject.get(subject)
        if candidates is None:
            raise EvalOpsError(f"no parquet rows for subject {subject!r}")
        match = candidates.get(stem)
        if match is None:
            raise EvalOpsError(f"source task {task_id} question not found in parquet subject {subject!r}")
        row_index, parquet_row = match
        parquet_choices = {letter: str(parquet_row.get(letter, "")).strip() for letter in _LETTERS}
        if choices != parquet_choices:
            raise EvalOpsError(f"task {task_id} choices differ from parquet row {row_index}")
        if gold != str(parquet_row.get("answer", "")).strip():
            raise EvalOpsError(f"task {task_id} gold {gold!r} differs from parquet answer")
        tasks.append(
            EvalTask(
                id=task_id,
                subject=subject,
                question=rendered,
                choices=choices,
                gold=gold,
                source={
                    "dataset": source_metadata.get("dataset", "C-Eval"),
                    "split": source_metadata.get("split", "val"),
                    "revision": source_metadata.get("revision", _CEVAL_REVISION),
                    "license": source_metadata.get("license", _CEVAL_LICENSE),
                    "subject": subject,
                    "row_index": row_index,
                    "parquet_file": source_metadata.get("parquet_files", {}).get(subject),
                },
            )
        )

    tasks_path = out / "tasks-v1.jsonl"
    write_jsonl(tasks_path, (task.to_dict() for task in tasks))
    subject_counts: dict[str, int] = {}
    for task in tasks:
        subject_counts[task.subject] = subject_counts.get(task.subject, 0) + 1
    manifest = {
        "schema_version": "evalops.dataset.v1",
        "dataset": source_metadata.get("dataset", "C-Eval"),
        "split": source_metadata.get("split", "val"),
        "revision": source_metadata.get("revision", _CEVAL_REVISION),
        "license": source_metadata.get("license", _CEVAL_LICENSE),
        "source_url": source_metadata.get("source_url", _CEVAL_SOURCE_URL),
        "source_jsonl_sha256": source_metadata.get("source_jsonl_sha256"),
        "source_rows": len(source_records),
        "mapping_verified": True,
        "tasks": {
            "path": "tasks-v1.jsonl",
            "sha256": sha256_path(tasks_path),
            "count": len(tasks),
        },
        "subjects": dict(sorted(subject_counts.items())),
        "parquet_files": dict(sorted(source_metadata.get("parquet_files", {}).items())),
        "opencompass_export": {
            "status": "compatibility_export_only",
            "validated_external_runner": False,
            "path": "opencompass/ceval-201qa.jsonl",
            "docs": "https://opencompass.readthedocs.io/en/stable/advanced_guides/custom_dataset.html",
        },
    }
    write_json(out / "manifest.json", manifest)
    _write_notice(out, manifest)
    _write_opencompass_compat(out, tasks)
    return manifest


def _write_notice(output_dir: Path, manifest: dict[str, Any]) -> None:
    notice = f"""# Data notice

This directory contains a frozen subset derived from the public C-Eval
validation split:

- Dataset: `{manifest['dataset']}`
- Revision: `{manifest['revision']}`
- Source: {manifest['source_url']}
- License: `{manifest['license']}`

The subset is redistributed for non-commercial research and evaluation.  Keep
this notice with the task file.  The subset is not the official C-Eval test
set and must not be presented as an official leaderboard result.  The OpenCompass
files are a compatibility export only; no external OpenCompass run is claimed.
"""
    (output_dir / "NOTICE.md").write_text(notice, encoding="utf-8")


def _write_opencompass_compat(output_dir: Path, tasks: list[EvalTask]) -> None:
    target = output_dir / "opencompass"
    target.mkdir(parents=True, exist_ok=True)
    legacy_meta = target / "ceval-201meta.json"
    if legacy_meta.exists():
        legacy_meta.unlink()
    rows = []
    for task in tasks:
        row = {"question": task.prompt().split("\nA. ", 1)[0].strip()}
        row.update(task.choices)
        row["answer"] = task.gold
        rows.append(row)
    write_jsonl(target / "ceval-201qa.jsonl", rows)
    write_json(
        target / "ceval-201qa.jsonl.meta.json",
        {
            "abbr": "CEval-201",
            "data_type": "mcq",
            "infer_method": "gen",
            "template": (
                "Question: {question}\n"
                "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
                "Only output one letter.\nAnswer:"
            ),
            "input_columns": ["question", "A", "B", "C", "D"],
            "output_column": "answer",
            "_status": "compatibility_export_only; external runner not executed",
        },
    )


def build_ceval_dataset(
    source_jsonl: str | Path,
    parquet_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build the 201-row C-Eval subset from the local public parquet cache."""
    source_path = Path(source_jsonl)
    source_records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parquet_files: dict[str, Any] = {}
    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - builder-only optional dependency
        raise EvalOpsError("building from parquet requires pyarrow") from exc
    for path in sorted(Path(parquet_dir).glob("ceval_*.parquet")):
        subject = path.stem.removeprefix("ceval_")
        table = pq.read_table(path)
        rows_by_subject[subject] = table.to_pylist()
        parquet_files[subject] = {
            "file": path.name,
            "sha256": sha256_path(path),
            "rows": len(rows_by_subject[subject]),
        }
    if not rows_by_subject:
        raise EvalOpsError(f"no ceval_*.parquet files found under {parquet_dir}")
    metadata = {
        "dataset": "C-Eval val (frozen 201-row public subset)",
        "split": "val",
        "revision": _CEVAL_REVISION,
        "license": _CEVAL_LICENSE,
        "source_url": _CEVAL_SOURCE_URL,
        "source_jsonl_sha256": sha256_path(source_path),
        "parquet_files": parquet_files,
    }
    return build_dataset_from_records(
        source_records,
        rows_by_subject,
        output_dir=output_dir,
        source_metadata=metadata,
        force=force,
    )


def verify_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Check manifest-to-task hash integrity and basic task invariants."""
    directory = Path(dataset_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "error": f"missing manifest: {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks_path = directory / str(manifest.get("tasks", {}).get("path", "tasks-v1.jsonl"))
    if not tasks_path.exists():
        return {"ok": False, "error": f"missing task file: {tasks_path}"}
    try:
        tasks = load_tasks(tasks_path)
    except EvalOpsError as exc:
        return {"ok": False, "error": str(exc)}
    expected_hash = manifest.get("tasks", {}).get("sha256")
    actual_hash = sha256_path(tasks_path)
    expected_count = manifest.get("tasks", {}).get("count")
    subject_counts: dict[str, int] = {}
    for task in tasks:
        subject_counts[task.subject] = subject_counts.get(task.subject, 0) + 1
    ok = (
        actual_hash == expected_hash
        and len(tasks) == expected_count
        and subject_counts == manifest.get("subjects")
        and bool(manifest.get("mapping_verified"))
    )
    return {
        "ok": ok,
        "manifest_sha256": sha256_path(manifest_path),
        "tasks_sha256": actual_hash,
        "task_count": len(tasks),
        "subjects": dict(sorted(subject_counts.items())),
        "expected_tasks_sha256": expected_hash,
        "expected_task_count": expected_count,
    }
