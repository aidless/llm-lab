#!/usr/bin/env python3
"""Verify the Agent Safety evidence package without trusting report claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIVATE_PATH_RE = re.compile(r"(?:/home|/Users)/[^/\s\"']+")
sys.path.insert(0, str(ROOT))

from llm_lab.evalops import write_json  # noqa: E402
from llm_lab.safety.models import (  # noqa: E402
    load_tasks,
    task_set_hash,
    verify_safety_dataset,
)
from llm_lab.safety.runner import (  # noqa: E402
    SafetyRunConfig,
    Trajectory,
    aggregate_trajectories,
    rescore_with_label,
)

from agent_safety_eval.reporting import (  # noqa: E402
    build_combined_report,
    html_report,
    markdown_report,
)
from agent_safety_eval.review import (  # noqa: E402
    required_trajectory_review_ids,
    summarize_review_records,
    trajectory_review_id,
    trajectory_set_hash,
    validate_review_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(payload)
    return rows


def _resolve_review_file(
    evidence_dir: Path,
    *,
    reviews: str | None,
    review_dir: str | None,
) -> Path:
    """Resolve human reviews from an explicit path or the run's review packet."""
    if reviews:
        return Path(reviews).expanduser().resolve()
    if review_dir:
        return Path(review_dir).expanduser().resolve() / "reviews-human.jsonl"

    packet_path = evidence_dir / "review-packet.json"
    if packet_path.exists():
        packet = _read_json(packet_path)
        recorded_dir = packet.get("review_dir")
        if isinstance(recorded_dir, str) and recorded_dir.strip():
            candidate = Path(recorded_dir).expanduser()
            if not candidate.is_absolute():
                candidate = evidence_dir.expanduser().resolve() / candidate
            return candidate.resolve() / "reviews-human.jsonl"

    local_review = evidence_dir / "review"
    if (local_review / "review-queue.json").exists():
        return local_review / "reviews-human.jsonl"
    return ROOT / "review" / "reviews-human.jsonl"


def _reviews(
    path: Path,
    tasks: list[Any],
    trajectories: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(path) if path.exists() else []
    known_trajectory_ids = {trajectory_review_id(item) for item in trajectories}
    rows = validate_review_records(
        rows,
        expected_task_ids={task.id for task in tasks},
        known_trajectory_ids=known_trajectory_ids,
    )
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") != "task_label":
            continue
        if row.get("decision") not in {"accept", "revise"}:
            continue
        reviewed = row.get("reviewed_label")
        if not isinstance(reviewed, dict):
            continue
        task_id = str(row["task_id"])
        labels[task_id] = {
            "id": task_id,
            **reviewed,
            "reviewer": row.get("reviewer"),
            "reviewed_at_utc": row.get("reviewed_at_utc"),
        }
    summary = summarize_review_records(
        rows,
        expected_task_ids={task.id for task in tasks},
        required_trajectory_ids=required_trajectory_review_ids(trajectories),
        task_set_hash=task_set_hash(tasks),
        trajectory_set_hash=trajectory_set_hash(trajectories),
    )
    return labels, summary


def _compare_report(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("task_set_hash", "model", "policy_mode", "config_hash"):
        if actual.get(key) != expected.get(key):
            errors.append(f"report {key} mismatch")
    for key in ("counts", "metrics", "category_breakdown", "failure_counts", "tool_failure_counts"):
        if actual.get(key) != expected.get(key):
            errors.append(f"report {key} mismatch")
    if actual.get("trajectory_set_hash") != expected.get("trajectory_set_hash"):
        errors.append("report trajectory_set_hash mismatch")
    return errors


def verify(
    *,
    dataset_dir: Path,
    evidence_dir: Path,
    review_file: Path,
    allow_pending_human: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    dataset_result = verify_safety_dataset(dataset_dir)
    checks["dataset_verified"] = bool(dataset_result.get("ok"))
    if not dataset_result.get("ok"):
        errors.append(f"dataset verification failed: {dataset_result}")
    tasks = load_tasks(dataset_dir / "tasks-v1.jsonl")
    task_by_id = {task.id: task for task in tasks}
    checks["task_count"] = len(tasks)
    checks["category_counts"] = dict(Counter(task.category for task in tasks))
    if len(tasks) < 80:
        errors.append("fewer than 80 tasks")
    selection_path = evidence_dir / "run-selection.json"
    if not selection_path.exists():
        errors.append("missing run-selection.json")
    else:
        selection = _read_json(selection_path)
        if selection.get("task_ids") != [task.id for task in tasks]:
            errors.append("run selection is not the complete frozen task set")
        if selection.get("mode") != "full":
            errors.append(f"final evidence must use full selection, got {selection.get('mode')}")
        checks["selection_mode"] = selection.get("mode")

    all_trajectories = _read_jsonl(evidence_dir / "trajectories.jsonl")
    checks["raw_trajectory_count"] = len(all_trajectories)
    if len(all_trajectories) != len(tasks) * 4:
        errors.append(f"expected {len(tasks) * 4} raw trajectories, got {len(all_trajectories)}")
    models = sorted({str(item.get("model")) for item in all_trajectories})
    policies = sorted({str(item.get("policy_mode")) for item in all_trajectories})
    checks["models"] = models
    checks["policies"] = policies
    if len(models) < 2:
        errors.append("fewer than two models")
    if policies != ["off", "on"]:
        errors.append(f"expected policy off/on, got {policies}")
    trajectory_keys = Counter(
        (
            str(item.get("model")),
            str(item.get("policy_mode")),
            str(item.get("task_id")),
        )
        for item in all_trajectories
    )
    duplicates = sorted(key for key, count in trajectory_keys.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate trajectories: {duplicates[:5]}")
    expected_trajectory_keys = {(model, policy, task.id) for model in models for policy in policies for task in tasks}
    if set(trajectory_keys) != expected_trajectory_keys:
        errors.append("trajectory model/policy/task matrix is incomplete")

    required_trajectory_fields = {
        "task_id",
        "events",
        "tool_calls",
        "terminal",
        "terminal_reason",
        "environment",
    }
    for index, item in enumerate(all_trajectories, 1):
        missing = sorted(required_trajectory_fields - set(item))
        if missing:
            errors.append(f"trajectory {index} missing fields: {missing}")
            continue
        environment = item.get("environment")
        if not isinstance(environment, dict) or not environment.get("no_real_filesystem_or_network_access"):
            errors.append(f"trajectory {index} lacks the mock-only environment marker")
        for call in item.get("tool_calls", []):
            if not isinstance(call, dict) or "arguments" not in call or "policy" not in call:
                errors.append(f"trajectory {index} has an incomplete tool-call record")
                break

    try:
        review_labels, review_summary = _reviews(
            review_file,
            tasks,
            all_trajectories,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid human review records: {exc}")
        review_labels = {}
        review_summary = summarize_review_records(
            [],
            expected_task_ids={task.id for task in tasks},
            required_trajectory_ids=required_trajectory_review_ids(all_trajectories),
            task_set_hash=task_set_hash(tasks),
            trajectory_set_hash=trajectory_set_hash(all_trajectories),
        )
    checks["human_review"] = review_summary
    queue_path = review_file.with_name("review-queue.json")
    if not queue_path.exists():
        errors.append(f"missing review queue: {queue_path}")
    else:
        queue = _read_json(queue_path)
        if queue.get("task_set_hash") != task_set_hash(tasks):
            errors.append("review queue task-set hash mismatch")
        if queue.get("trajectory_set_hash") != trajectory_set_hash(all_trajectories):
            errors.append("review queue trajectory-set hash mismatch")
        expected_required = sorted(required_trajectory_review_ids(all_trajectories))
        if queue.get("required_trajectory_ids") != expected_required:
            errors.append("review queue required-trajectory set mismatch")
    if not review_summary["human_gate_complete"]:
        checks["human_gate_note"] = "technical evidence can pass while the complete human-review gate is pending"
        if not allow_pending_human:
            errors.append("human review does not cover all task labels and required unsafe trajectories")

    run_reports: list[dict[str, Any]] = []
    rescored_trajectories: list[dict[str, Any]] = []
    for run_dir in sorted(evidence_dir.glob("runs/*")):
        report_path = run_dir / "report.json"
        meta_path = run_dir / "run-meta.json"
        trajectory_path = run_dir / "trajectories.jsonl"
        if not (report_path.exists() and meta_path.exists() and trajectory_path.exists()):
            errors.append(f"incomplete run directory: {run_dir}")
            continue
        meta = _read_json(meta_path)
        config = SafetyRunConfig(**meta["config"])
        raw = [Trajectory.from_dict(item) for item in _read_jsonl(trajectory_path)]
        scored: list[Trajectory] = []
        for trajectory in raw:
            label = review_labels.get(trajectory.task_id)
            if label is not None:
                trajectory = rescore_with_label(
                    trajectory,
                    label=label,
                    label_source="human-review",
                )
            scored.append(trajectory)
        matching = [task_by_id[trajectory.task_id] for trajectory in scored]
        expected = aggregate_trajectories(
            scored,
            tasks=matching,
            config=config,
            wall_seconds=float(meta.get("wall_seconds") or 0.0),
        )
        actual = _read_json(report_path)
        mismatch = _compare_report(actual, expected)
        if mismatch:
            errors.extend(f"{run_dir.name}: {item}" for item in mismatch)
        # Drop wall-clock-only timestamps from the deterministic comparison;
        # all safety and metric fields are still checked above and below.
        deterministic_actual = {key: value for key, value in actual.items() if key != "captured_at_utc"}
        run_reports.append(deterministic_actual)
        rescored_trajectories.extend(item.to_dict() for item in scored)

    combined_path = evidence_dir / "report.json"
    combined = _read_json(combined_path)
    if combined.get("schema_version") != "agent-safety.suite-report.v1":
        errors.append("combined report has an unsupported schema")
    if combined.get("dataset", {}).get("task_set_hash") != task_set_hash(tasks):
        errors.append("combined report task hash mismatch")
    selected_tasks = tasks
    expected_combined = build_combined_report(
        runs=[{key: value for key, value in run.items() if key != "captured_at_utc"} for run in run_reports],
        task_count=len(selected_tasks),
        task_hash=task_set_hash(selected_tasks),
        category_counts=dict(Counter(task.category for task in selected_tasks)),
        review_summary=review_summary,
    )
    if selection_path.exists():
        expected_combined["dataset"]["selection"] = _read_json(selection_path)
    comparable_keys = set(combined) | set(expected_combined)
    mismatched_keys = [
        key
        for key in sorted(comparable_keys)
        if key != "runs" and combined.get(key) != expected_combined.get(key)
    ]
    if mismatched_keys:
        errors.append(f"combined report does not match deterministic recomputation: {mismatched_keys}")
    for name, expected_text in (
        ("report.md", markdown_report(expected_combined)),
        ("report.html", html_report(expected_combined)),
    ):
        path = evidence_dir / name
        checks[f"{name}_bytes"] = path.stat().st_size if path.exists() else 0
        actual_text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not path.exists() or actual_text != expected_text:
            errors.append(f"{name} does not match deterministic recomputation")

    rescored_path = evidence_dir / "trajectories-rescored.jsonl"
    if rescored_path.exists() and _read_jsonl(rescored_path) != rescored_trajectories:
        errors.append("trajectories-rescored.jsonl does not match the reviewed rescore")

    for path in evidence_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md", ".html", ".log", ".txt"}:
            continue
        if PRIVATE_PATH_RE.search(path.read_text(encoding="utf-8", errors="replace")):
            errors.append(f"private workstation path leaked into {path.relative_to(evidence_dir)}")

    checksums_path = evidence_dir / "SHA256SUMS"
    checksum_count = 0
    listed_targets: set[str] = set()
    if not checksums_path.exists():
        errors.append("missing SHA256SUMS")
    else:
        for line_number, line in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"SHA256SUMS:{line_number} malformed")
                continue
            expected_hash, relative = parts
            target = evidence_dir / relative.strip()
            listed_targets.add(relative.strip())
            if not target.exists():
                errors.append(f"SHA256SUMS target missing: {relative}")
                continue
            if _sha256(target) != expected_hash:
                errors.append(f"SHA256SUMS mismatch: {relative}")
                continue
            checksum_count += 1
        actual_targets = {
            path.relative_to(evidence_dir).as_posix()
            for path in evidence_dir.rglob("*")
            if path.is_file()
            and path.relative_to(evidence_dir).as_posix()
            not in {"SHA256SUMS", "verification-result.json"}
        }
        if listed_targets != actual_targets:
            errors.append("SHA256SUMS does not cover exactly the evidence files")
    checks["checksum_files_verified"] = checksum_count

    technical_ok = not errors
    acceptance_ok = technical_ok and review_summary["human_gate_complete"]
    result = {
        "schema_version": "agent-safety.verification.v1",
        "technical_ok": technical_ok,
        "acceptance_ok": acceptance_ok,
        "human_gate_pending": not review_summary["human_gate_complete"],
        "checks": checks,
        "errors": errors,
        "environment": _read_json(evidence_dir / "environment.json")
        if (evidence_dir / "environment.json").exists()
        else {},
        "boundary": "本地安全评测，不是生产安全系统。 Mock-only local evaluation with a separate human-review gate.",
    }
    write_json(evidence_dir / "verification-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--evidence", default=str(ROOT / "evidence"))
    parser.add_argument("--reviews", default=None)
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--allow-pending-human", action="store_true")
    args = parser.parse_args()
    review_file = _resolve_review_file(
        Path(args.evidence),
        reviews=args.reviews,
        review_dir=args.review_dir,
    )
    result = verify(
        dataset_dir=Path(args.dataset),
        evidence_dir=Path(args.evidence),
        review_file=review_file,
        allow_pending_human=args.allow_pending_human,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["technical_ok"] and result["human_gate_pending"]:
        return 0 if args.allow_pending_human else 1
    return 0 if result["acceptance_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
