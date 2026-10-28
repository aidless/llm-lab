"""Auditable human-review packet import and export."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from llm_lab.safety.models import (
    JudgeCase,
    SafetyDataError,
    SafetyTask,
    canonical_json,
    judge_case_set_hash,
    task_set_hash,
)

_TASK_FIELDS = (
    "record_type",
    "id",
    "draft_label",
    "expected_terminals",
    "must_halt",
    "benign",
    "required_tools",
    "rationale",
    "decision",
    "notes",
)


def export_review_packet(
    output_path: Path,
    *,
    tasks: list[SafetyTask],
    judge_cases: list[JudgeCase],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TASK_FIELDS)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "record_type": "task",
                    "id": task.id,
                    "draft_label": task.judge_label,
                    "expected_terminals": "|".join(task.draft_label.expected_terminals),
                    "must_halt": str(task.draft_label.must_halt).lower(),
                    "benign": str(task.draft_label.benign).lower(),
                    "required_tools": "|".join(task.draft_label.required_tools),
                    "rationale": task.draft_label.rationale,
                    "decision": "",
                    "notes": "",
                }
            )
        for case in judge_cases:
            writer.writerow(
                {
                    "record_type": "judge",
                    "id": case.id,
                    "draft_label": case.draft_label,
                    "expected_terminals": "",
                    "must_halt": "",
                    "benign": "",
                    "required_tools": "",
                    "rationale": case.rationale,
                    "decision": "",
                    "notes": "",
                }
            )


def _parse_yes_no(value: str, *, field: str, record_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise SafetyDataError(f"{record_id}: {field} must be true/false, yes/no, or 1/0")


def import_review_packet(
    input_path: Path,
    output_path: Path,
    *,
    tasks: list[SafetyTask],
    judge_cases: list[JudgeCase],
    reviewer: str,
    attest_human: bool,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise SafetyDataError("reviewer identity must be non-empty")
    if not attest_human:
        raise SafetyDataError("--attest-human is required for a human review")

    expected_ids = {f"task:{task.id}" for task in tasks} | {f"judge:{case.id}" for case in judge_cases}
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(_TASK_FIELDS):
            raise SafetyDataError("review CSV columns must exactly match the exported packet")
        for row in reader:
            record_type = str(row["record_type"]).strip()
            record_id = str(row["id"]).strip()
            key = f"{record_type}:{record_id}"
            if key not in expected_ids:
                raise SafetyDataError(f"unknown or wrong-type review row: {key}")
            if key in seen:
                raise SafetyDataError(f"duplicate review row: {key}")
            seen.add(key)
            decision = str(row["decision"]).strip().lower()
            if decision not in {"accept", "revise"}:
                raise SafetyDataError(f"{key}: decision must be accept or revise")
            if record_type == "task":
                expected_terminals = [
                    item.strip() for item in str(row["expected_terminals"]).split("|") if item.strip()
                ]
                if not expected_terminals:
                    raise SafetyDataError(f"{key}: expected_terminals cannot be empty")
                records.append(
                    {
                        "record_type": "task",
                        "id": record_id,
                        "decision": decision,
                        "expected_terminals": expected_terminals,
                        "must_halt": _parse_yes_no(row["must_halt"], field="must_halt", record_id=key),
                        "benign": _parse_yes_no(row["benign"], field="benign", record_id=key),
                        "required_tools": [
                            item.strip() for item in str(row["required_tools"]).split("|") if item.strip()
                        ],
                        "notes": str(row["notes"]),
                    }
                )
            else:
                label = str(row["draft_label"]).strip().lower()
                if label not in {"safe", "unsafe"}:
                    raise SafetyDataError(f"{key}: judge label must be safe or unsafe")
                records.append(
                    {
                        "record_type": "judge",
                        "id": record_id,
                        "decision": decision,
                        "label": label,
                        "notes": str(row["notes"]),
                    }
                )
    missing = sorted(expected_ids - seen)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise SafetyDataError(f"review is incomplete; missing {len(missing)} rows: {preview}{suffix}")

    payload = {
        "schema_version": "agent-safety.review.v1",
        "reviewed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reviewer": reviewer.strip(),
        "attestation": "I reviewed every record as a human and did not treat draft labels as truth.",
        "task_set_hash": task_set_hash(tasks),
        "judge_case_set_hash": judge_case_set_hash(judge_cases),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_review_packet(
    path: Path,
    *,
    tasks: list[SafetyTask],
    judge_cases: list[JudgeCase],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "agent-safety.review.v1":
        raise SafetyDataError("unsupported review schema_version")
    if str(payload.get("task_set_hash")) != task_set_hash(tasks):
        raise SafetyDataError("review task-set hash does not match the frozen dataset")
    if str(payload.get("judge_case_set_hash")) != judge_case_set_hash(judge_cases):
        raise SafetyDataError("review judge-case hash does not match the frozen dataset")
    records = payload.get("records")
    if not isinstance(records, list):
        raise SafetyDataError("review records must be a list")
    expected = {f"task:{task.id}" for task in tasks} | {f"judge:{case.id}" for case in judge_cases}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise SafetyDataError("review record must be an object")
        key = f"{record.get('record_type')}:{record.get('id')}"
        if key not in expected:
            raise SafetyDataError(f"unknown review record: {key}")
        if key in seen:
            raise SafetyDataError(f"duplicate review record: {key}")
        seen.add(key)
    missing = sorted(expected - seen)
    if missing:
        raise SafetyDataError(f"review packet is incomplete ({len(missing)} missing rows)")
    return payload


def task_review_map(review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in review["records"] if record.get("record_type") == "task"}


def judge_label_map(review: dict[str, Any]) -> dict[str, str]:
    return {
        str(record["id"]): str(record["label"]) for record in review["records"] if record.get("record_type") == "judge"
    }


def review_packet_hash(review: dict[str, Any]) -> str:
    from llm_lab.safety.models import sha256_text

    return sha256_text(canonical_json(review))
