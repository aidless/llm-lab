"""Frozen task contracts for the local Agent Safety evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SAFETY_SCHEMA_VERSION = "agent-safety.task.v1"
JUDGE_SCHEMA_VERSION = "agent-safety.judge-case.v1"
CATEGORIES = (
    "prompt_injection",
    "unauthorized_tool",
    "dangerous_file_op",
    "data_exfiltration",
    "stop_failure",
    "tool_error_recovery",
)
TERMINALS = ("finish", "refuse", "ask_user", "halt", "policy_halt")
JUDGE_LABELS = ("safe", "unsafe")


class SafetyDataError(ValueError):
    """A frozen safety task or manifest violates the data contract."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DraftLabel:
    """Model-assisted draft labels that must be human-reviewed before acceptance."""

    expected_terminals: tuple[str, ...]
    must_halt: bool
    benign: bool
    required_tools: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.expected_terminals:
            raise SafetyDataError("draft label must contain at least one expected terminal")
        invalid = sorted(set(self.expected_terminals) - set(TERMINALS))
        if invalid:
            raise SafetyDataError(f"invalid expected terminal(s): {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_terminals": list(self.expected_terminals),
            "must_halt": self.must_halt,
            "benign": self.benign,
            "required_tools": list(self.required_tools),
            "rationale": self.rationale,
            "review_status": "draft",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DraftLabel:
        try:
            expected = tuple(str(item) for item in payload["expected_terminals"])
            required = tuple(str(item) for item in payload.get("required_tools", []))
            return cls(
                expected_terminals=expected,
                must_halt=bool(payload["must_halt"]),
                benign=bool(payload["benign"]),
                required_tools=required,
                rationale=str(payload.get("rationale", "")),
            )
        except KeyError as exc:
            raise SafetyDataError(f"draft label missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class SafetyTask:
    """One synthetic agent task executed entirely inside a mock environment."""

    id: str
    category: str
    title: str
    user_goal: str
    allowed_tools: tuple[str, ...]
    policy: dict[str, Any]
    files: dict[str, dict[str, Any]]
    urls: dict[str, str]
    mock_responses: dict[str, Any]
    tool_faults: tuple[dict[str, Any], ...]
    max_steps: int
    draft_label: DraftLabel
    judge_request: str
    judge_label: str
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.title or not self.user_goal:
            raise SafetyDataError("task id, title and user_goal must be non-empty")
        if self.category not in CATEGORIES:
            raise SafetyDataError(f"unknown category {self.category!r} in {self.id}")
        if not self.allowed_tools:
            raise SafetyDataError(f"task {self.id} must expose at least one tool")
        if self.max_steps < 1 or self.max_steps > 20:
            raise SafetyDataError(f"task {self.id} max_steps must be between 1 and 20")
        if self.judge_label not in JUDGE_LABELS:
            raise SafetyDataError(f"task {self.id} judge_label must be safe or unsafe")
        if any(not path.startswith("/") for path in self.files):
            raise SafetyDataError(f"task {self.id} file paths must be absolute")
        for url in self.urls:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise SafetyDataError(f"task {self.id} contains an invalid mock URL: {url!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SAFETY_SCHEMA_VERSION,
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "user_goal": self.user_goal,
            "allowed_tools": list(self.allowed_tools),
            "policy": self.policy,
            "files": self.files,
            "urls": self.urls,
            "mock_responses": self.mock_responses,
            "tool_faults": list(self.tool_faults),
            "max_steps": self.max_steps,
            "draft_label": self.draft_label.to_dict(),
            "judge_request": self.judge_request,
            "judge_label": self.judge_label,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SafetyTask:
        if payload.get("schema_version") != SAFETY_SCHEMA_VERSION:
            raise SafetyDataError(f"task {payload.get('id', '<unknown>')} has unsupported schema_version")
        try:
            raw_files = payload.get("files", {})
            raw_urls = payload.get("urls", {})
            raw_responses = payload.get("mock_responses", {})
            raw_faults = payload.get("tool_faults", [])
            raw_policy = payload["policy"]
            raw_label = payload["draft_label"]
        except KeyError as exc:
            raise SafetyDataError(f"task missing field: {exc.args[0]}") from exc
        if not isinstance(raw_files, dict) or not isinstance(raw_urls, dict):
            raise SafetyDataError(f"task {payload.get('id')} files/urls must be mappings")
        if not isinstance(raw_responses, dict) or not isinstance(raw_policy, dict):
            raise SafetyDataError(f"task {payload.get('id')} policy/mock_responses must be mappings")
        if not isinstance(raw_faults, list) or not isinstance(raw_label, dict):
            raise SafetyDataError(f"task {payload.get('id')} tool_faults/draft_label have invalid types")
        source = payload.get("source")
        if source is None:
            source = {}
        if not isinstance(source, dict):
            raise SafetyDataError(f"task {payload.get('id')} source must be a mapping")
        files = {
            str(path): dict(metadata) if isinstance(metadata, dict) else {"content": str(metadata)}
            for path, metadata in raw_files.items()
        }
        return cls(
            id=str(payload["id"]),
            category=str(payload["category"]),
            title=str(payload["title"]),
            user_goal=str(payload["user_goal"]),
            allowed_tools=tuple(str(tool) for tool in payload["allowed_tools"]),
            policy=dict(raw_policy),
            files=files,
            urls={str(url): str(content) for url, content in raw_urls.items()},
            mock_responses=dict(raw_responses),
            tool_faults=tuple(dict(fault) for fault in raw_faults),
            max_steps=int(payload["max_steps"]),
            draft_label=DraftLabel.from_dict(raw_label),
            judge_request=str(payload["judge_request"]),
            judge_label=str(payload["judge_label"]),
            source=dict(source),
        )


@dataclass(frozen=True)
class JudgeCase:
    """A request-level safety label used to calibrate an LLM judge."""

    id: str
    request: str
    draft_label: str
    rationale: str
    source_task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.request:
            raise SafetyDataError("judge case id and request must be non-empty")
        if self.draft_label not in JUDGE_LABELS:
            raise SafetyDataError(f"judge case {self.id} has invalid draft_label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JUDGE_SCHEMA_VERSION,
            "id": self.id,
            "request": self.request,
            "draft_label": self.draft_label,
            "rationale": self.rationale,
            "source_task_id": self.source_task_id,
            "review_status": "draft",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JudgeCase:
        if payload.get("schema_version") != JUDGE_SCHEMA_VERSION:
            raise SafetyDataError(f"judge case {payload.get('id', '<unknown>')} has unsupported schema_version")
        return cls(
            id=str(payload["id"]),
            request=str(payload["request"]),
            draft_label=str(payload["draft_label"]),
            rationale=str(payload.get("rationale", "")),
            source_task_id=(str(payload["source_task_id"]) if payload.get("source_task_id") is not None else None),
        )


def load_tasks(path: Path) -> list[SafetyTask]:
    tasks: list[SafetyTask] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SafetyDataError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise SafetyDataError(f"{path}:{line_number}: task must be a JSON object")
        tasks.append(SafetyTask.from_dict(payload))
    if not tasks:
        raise SafetyDataError(f"{path}: no tasks found")
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise SafetyDataError(f"{path}: duplicate task id")
    return tasks


def load_judge_cases(path: Path) -> list[JudgeCase]:
    cases: list[JudgeCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SafetyDataError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise SafetyDataError(f"{path}:{line_number}: judge case must be a JSON object")
        cases.append(JudgeCase.from_dict(payload))
    if not cases:
        raise SafetyDataError(f"{path}: no judge cases found")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise SafetyDataError(f"{path}: duplicate judge case id")
    return cases


def task_set_hash(tasks: list[SafetyTask] | tuple[SafetyTask, ...]) -> str:
    return sha256_text(canonical_json([task.to_dict() for task in tasks]))


def judge_case_set_hash(cases: list[JudgeCase] | tuple[JudgeCase, ...]) -> str:
    return sha256_text(canonical_json([case.to_dict() for case in cases]))


def verify_safety_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "error": f"missing manifest: {manifest_path}"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks_path = dataset_dir / str(manifest["tasks"]["path"])
        tasks = load_tasks(tasks_path)
        actual_hash = task_set_hash(tasks)
        expected_hash = str(manifest["tasks"]["sha256"])
        if actual_hash != expected_hash:
            return {
                "ok": False,
                "error": "task-set hash mismatch",
                "expected": expected_hash,
                "actual": actual_hash,
            }
        if int(manifest["tasks"]["count"]) != len(tasks):
            return {
                "ok": False,
                "error": "task count mismatch",
                "expected": int(manifest["tasks"]["count"]),
                "actual": len(tasks),
            }
        judge_path = dataset_dir / str(manifest["judge_cases"]["path"])
        cases = load_judge_cases(judge_path)
        actual_judge_hash = judge_case_set_hash(cases)
        expected_judge_hash = str(manifest["judge_cases"]["sha256"])
        if actual_judge_hash != expected_judge_hash:
            return {
                "ok": False,
                "error": "judge-case hash mismatch",
                "expected": expected_judge_hash,
                "actual": actual_judge_hash,
            }
        if int(manifest["judge_cases"]["count"]) != len(cases):
            return {
                "ok": False,
                "error": "judge-case count mismatch",
                "expected": int(manifest["judge_cases"]["count"]),
                "actual": len(cases),
            }
        category_counts = {category: 0 for category in CATEGORIES}
        for task in tasks:
            category_counts[task.category] += 1
        expected_counts = {str(key): int(value) for key, value in dict(manifest.get("category_counts", {})).items()}
        if expected_counts and category_counts != expected_counts:
            return {
                "ok": False,
                "error": "category counts mismatch",
                "expected": expected_counts,
                "actual": category_counts,
            }
        return {
            "ok": True,
            "schema_version": str(manifest.get("schema_version", "")),
            "task_count": len(tasks),
            "tasks_sha256": actual_hash,
            "judge_case_count": len(cases),
            "judge_cases_sha256": actual_judge_hash,
            "category_counts": category_counts,
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def write_frozen_dataset(
    output_dir: Path,
    tasks: list[SafetyTask],
    judge_cases: list[JudgeCase],
    *,
    dataset_version: str,
    source_note: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "tasks-v1.jsonl"
    judge_path = output_dir / "judge-cases-v1.jsonl"
    tasks_path.write_text(
        "".join(canonical_json(task.to_dict()) + "\n" for task in tasks),
        encoding="utf-8",
    )
    judge_path.write_text(
        "".join(canonical_json(case.to_dict()) + "\n" for case in judge_cases),
        encoding="utf-8",
    )
    category_counts = {category: 0 for category in CATEGORIES}
    for task in tasks:
        category_counts[task.category] += 1
    manifest = {
        "schema_version": "agent-safety.dataset.v1",
        "dataset_version": dataset_version,
        "created_at_utc": "2026-10-27T00:00:00Z",
        "source_note": source_note,
        "safety_label_source": "model-assisted draft; independent human review pending",
        "mock_only": True,
        "tasks": {
            "path": tasks_path.name,
            "count": len(tasks),
            "sha256": task_set_hash(tasks),
        },
        "judge_cases": {
            "path": judge_path.name,
            "count": len(judge_cases),
            "sha256": judge_case_set_hash(judge_cases),
        },
        "category_counts": category_counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
