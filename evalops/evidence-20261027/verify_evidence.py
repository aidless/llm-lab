#!/usr/bin/env python3
"""Independently recompute and verify the packaged EvalOps evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2] / "llm-lab"
sys.path.insert(0, str(REPO))

from llm_lab.evalops import (
    EvalRecord,
    EvalTask,
    canonical_json,
    load_tasks,
    read_records,
)

EXPECTED_TASKS_SHA256 = "33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657"
EXPECTED_RUNS = {
    "ollama-201-c1": {
        "success": 201,
        "correct": 108,
        "answer_hash": "6569fbbf44b50e363dc731324ef991124538a5f177f9bfb09a49da6ba82690fb",
        "concurrency": 1,
    },
    "ollama-201-c4": {
        "success": 201,
        "correct": 107,
        "answer_hash": "e2e7ec77ce8fe2229710556471e0874f115b2eb52e98de202e7f3ba36add12ef",
        "concurrency": 4,
    },
    "ollama-201-c8": {
        "success": 201,
        "correct": 108,
        "answer_hash": "e6693e933ba8e64873a45daa4505308f90d8132829466dcd4cf0f218cd4dd509",
        "concurrency": 8,
    },
}
EXPECTED_AGREEMENT = {
    "c1_vs_c4": {"agree": 196, "total": 201},
    "c4_vs_c8": {"agree": 200, "total": 201},
    "c1_vs_c8": {"agree": 195, "total": 201},
}
EXPECTED_DETERMINISTIC_HASH = "77ace68bf48a4c309cf9ba5897c1d0238420cd0133cec9db8bb8be6a000781ed"
EXPECTED_FAULT_CATEGORIES = {
    "model_error": 1,
    "parse_error": 1,
    "timeout": 1,
    "transport": 1,
}


class VerificationError(RuntimeError):
    """Evidence does not match the recorded contract."""


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def independent_answer_hash(records: list[EvalRecord]) -> str:
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
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
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


def task_hash(tasks: list[EvalTask]) -> str:
    raw = canonical_json([task.to_dict() for task in tasks])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[EvalRecord]]:
    meta = json.loads((path / "run-meta.json").read_text(encoding="utf-8"))
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    records = read_records(path / "records.jsonl")
    return meta, report, records


def run_task_file(name: str) -> Path:
    if name == "fault-injection":
        return ROOT / name / "tasks-v1.jsonl"
    if name == "ollama-smoke12-c4":
        return ROOT / name / "tasks-smoke-12.jsonl"
    return ROOT / "dataset" / "tasks-v1.jsonl"


def verify_run(
    name: str,
    dataset_tasks_by_id: dict[str, EvalTask],
) -> dict[str, Any]:
    path = ROOT / name
    meta, report, records = load_run(path)
    tasks = load_tasks(run_task_file(name))
    tasks_by_id = {task.id: task for task in tasks}

    require(meta["schema_version"] == "evalops.run.v1", f"{name}: bad run schema")
    require(report["schema_version"] == "evalops.report.v1", f"{name}: bad report schema")
    require(meta["task_count"] == len(tasks), f"{name}: task-count mismatch")
    require(len(records) == len(tasks), f"{name}: record-count mismatch")
    require(len(tasks_by_id) == len(tasks), f"{name}: duplicate task ids")
    require(report["counts"]["total"] == len(records), f"{name}: report total mismatch")
    require(meta["config_hash"] == report["config_hash"], f"{name}: config hash mismatch")
    require(meta["task_set_hash"] == report["task_set_hash"], f"{name}: task hash mismatch")

    if name in {"fault-injection", "ollama-smoke12-c4"}:
        expected_task_file = run_task_file(name)
        require(tasks == load_tasks(expected_task_file), f"{name}: task reload mismatch")
    if name == "fault-injection":
        require(task_hash(tasks) == meta["task_set_hash"], f"{name}: local task hash mismatch")
    elif name == "ollama-smoke12-c4":
        require(
            meta["task_set_hash"] == EXPECTED_TASKS_SHA256,
            f"{name}: parent dataset hash mismatch",
        )

    for record in records:
        task = tasks_by_id.get(record.task_id)
        require(task is not None, f"{name}: unknown record task {record.task_id}")
        require(
            record.prompt_sha256 == task.prompt_sha256(),
            f"{name}: prompt hash mismatch for {record.task_id}",
        )
        require(
            record.config_hash == meta["config_hash"],
            f"{name}: record config hash mismatch for {record.task_id}",
        )
        require(record.gold == task.gold, f"{name}: gold mismatch for {record.task_id}")
        if name in {"ollama-201-c1", "ollama-201-c4", "ollama-201-c8"}:
            require(
                record.prompt_sha256 == dataset_tasks_by_id[record.task_id].prompt_sha256(),
                f"{name}: dataset prompt mismatch for {record.task_id}",
            )

    successes = [record for record in records if record.status == "success"]
    parsed = [record for record in successes if record.parsed_choice is not None]
    correct = sum(record.correct is True for record in records)
    computed_counts = {
        "total": len(records),
        "success": len(successes),
        "parsed": len(parsed),
        "correct": correct,
    }
    for field, value in computed_counts.items():
        require(report["counts"][field] == value, f"{name}: {field} count mismatch")

    recomputed_hash = independent_answer_hash(records)
    require(recomputed_hash == report["answer_hash"], f"{name}: answer hash mismatch")
    expected_strict = correct / len(records) if records else 0.0
    require(
        abs(report["rates"]["strict_accuracy"] - expected_strict) < 1e-12,
        f"{name}: strict accuracy mismatch",
    )
    expected_parsed = correct / len(parsed) if parsed else 0.0
    require(
        abs(report["rates"]["accuracy_parsed"] - expected_parsed) < 1e-12,
        f"{name}: parsed accuracy mismatch",
    )

    latencies = [record.latency_ms for record in records]
    require(
        abs(report["latency_ms"]["all"]["p50"] - percentile(latencies, 0.5)) < 1e-9,
        f"{name}: p50 mismatch",
    )
    require(
        abs(report["latency_ms"]["all"]["p95"] - percentile(latencies, 0.95)) < 1e-9,
        f"{name}: p95 mismatch",
    )
    require(
        abs(
            report["throughput_tasks_per_second"]
            - (len(records) / report["wall_seconds"] if report["wall_seconds"] > 0 else 0.0)
        )
        < 1e-12,
        f"{name}: throughput mismatch",
    )

    markdown = (path / "report.md").read_text(encoding="utf-8")
    html = (path / "report.html").read_text(encoding="utf-8")
    require(report["answer_hash"] in markdown, f"{name}: markdown lacks answer hash")
    require(report["answer_hash"] in html, f"{name}: HTML lacks answer hash")

    return {
        "run": name,
        "provider": report["provider"],
        "model": report["model"],
        "concurrency": report["config"]["concurrency"],
        "tasks": len(records),
        "success": len(successes),
        "correct": correct,
        "strict_accuracy": expected_strict,
        "parsed_accuracy": expected_parsed,
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "wall_seconds": report["wall_seconds"],
        "throughput_tasks_per_second": report["throughput_tasks_per_second"],
        "answer_hash": recomputed_hash,
        "config_hash": meta["config_hash"],
        "task_set_hash": meta["task_set_hash"],
        "failure_counts": report["counts"]["failures"],
    }


def choices(records_path: Path) -> tuple[dict[str, str | None], dict[str, str]]:
    records = read_records(records_path)
    return (
        {record.task_id: record.parsed_choice for record in records},
        {record.task_id: record.status for record in records},
    )


def agreement(name_a: str, name_b: str) -> dict[str, Any]:
    answers_a, status_a = choices(ROOT / name_a / "records.jsonl")
    answers_b, status_b = choices(ROOT / name_b / "records.jsonl")
    require(set(answers_a) == set(answers_b), f"{name_a}/{name_b}: task ids differ")
    task_ids = sorted(answers_a)
    agree = [
        task_id
        for task_id in task_ids
        if answers_a[task_id] == answers_b[task_id]
        and status_a[task_id] == status_b[task_id]
    ]
    return {
        "pair": f"{name_a} vs {name_b}",
        "agree": len(agree),
        "total": len(task_ids),
        "rate": len(agree) / len(task_ids),
        "disagreement_task_ids": sorted(set(task_ids) - set(agree)),
    }


def verify_checksum_file() -> int:
    checksum_path = ROOT / "SHA256SUMS"
    if not checksum_path.exists():
        return 0
    checked = 0
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, separator, raw_path = line.partition("  ")
        require(bool(separator and digest and raw_path), f"SHA256SUMS:{line_number}: malformed")
        path = ROOT / raw_path
        require(path.is_file(), f"SHA256SUMS:{line_number}: missing {raw_path}")
        require(sha256_path(path) == digest, f"SHA256SUMS:{line_number}: digest mismatch")
        checked += 1
    return checked


def build_summary() -> dict[str, Any]:
    dataset_path = ROOT / "dataset" / "tasks-v1.jsonl"
    require(sha256_path(dataset_path) == EXPECTED_TASKS_SHA256, "dataset hash mismatch")
    dataset_tasks = load_tasks(dataset_path)
    require(len(dataset_tasks) == 201, "dataset count mismatch")
    dataset_by_id = {task.id: task for task in dataset_tasks}

    runs = {
        name: verify_run(name, dataset_by_id)
        for name in sorted(path.name for path in ROOT.iterdir() if (path / "report.json").is_file())
    }
    for name, expected in EXPECTED_RUNS.items():
        run = runs[name]
        require(run["success"] == expected["success"], f"{name}: success summary mismatch")
        require(run["correct"] == expected["correct"], f"{name}: correct summary mismatch")
        require(run["answer_hash"] == expected["answer_hash"], f"{name}: answer summary mismatch")
        require(run["concurrency"] == expected["concurrency"], f"{name}: concurrency mismatch")

    repeat = {
        key: agreement("ollama-201-c1", "ollama-201-c4")
        if key == "c1_vs_c4"
        else agreement("ollama-201-c4", "ollama-201-c8")
        if key == "c4_vs_c8"
        else agreement("ollama-201-c1", "ollama-201-c8")
        for key in EXPECTED_AGREEMENT
    }
    for key, expected in EXPECTED_AGREEMENT.items():
        require(repeat[key]["agree"] == expected["agree"], f"{key}: agreement mismatch")
        require(repeat[key]["total"] == expected["total"], f"{key}: agreement total mismatch")

    deterministic = [runs[f"deterministic-c{value}"] for value in (1, 4, 8)]
    for run in deterministic:
        require(run["answer_hash"] == EXPECTED_DETERMINISTIC_HASH, "fixture hash mismatch")
    require(len({run["answer_hash"] for run in deterministic}) == 1, "fixture hash drift")

    fault = runs["fault-injection"]
    require(fault["failure_counts"] == EXPECTED_FAULT_CATEGORIES, "fault classes mismatch")
    smoke = runs["ollama-smoke12-c4"]

    opencompass_meta = json.loads(
        (ROOT / "dataset" / "opencompass" / "ceval-201qa.jsonl.meta.json").read_text(
            encoding="utf-8"
        )
    )
    require(opencompass_meta["data_type"] == "mcq", "OpenCompass data_type mismatch")
    require(opencompass_meta["infer_method"] == "gen", "OpenCompass infer_method mismatch")
    require(
        opencompass_meta["_status"].startswith("compatibility_export_only"),
        "OpenCompass boundary missing",
    )

    return {
        "schema_version": "evalops.evidence-summary.v1",
        "captured_date": "2026-10-27",
        "dataset": {
            "name": "C-Eval validation subset",
            "tasks": 201,
            "subjects": 10,
            "revision": "617524a00b307ff6f9933702f724131fe12ca7ce",
            "license": "CC-BY-NC-SA-4.0",
            "tasks_sha256": EXPECTED_TASKS_SHA256,
            "official_leaderboard_result": False,
        },
        "ollama_201": runs["ollama-201-c1"],
        "ollama_concurrency_runs": [
            runs["ollama-201-c1"],
            runs["ollama-201-c4"],
            runs["ollama-201-c8"],
        ],
        "repeat_consistency": repeat,
        "ollama_smoke_12": smoke,
        "deterministic_fixture": {
            "answer_hash": EXPECTED_DETERMINISTIC_HASH,
            "runs": [run["run"] for run in deterministic],
            "strict_accuracy": deterministic[0]["strict_accuracy"],
            "model_quality_evidence": False,
        },
        "fault_injection": {
            "tasks": fault["tasks"],
            "failure_counts": fault["failure_counts"],
            "source": "deterministic adapter with synthetic faults",
        },
        "compatibility_exports": {
            "opencompass": "exported and contract-checked; external runner not executed",
            "vllm": "adapter boundary exists; vLLM not installed or executed",
        },
        "limitations": [
            "The 201-item public validation subset is not the official C-Eval benchmark or leaderboard.",
            "Repeated agreement is not bitwise determinism or independent statistical significance.",
            "Local CPU timings are not production SLOs and do not generalize to GPU serving.",
            "The deterministic fixture is a plumbing test, never model-quality evidence.",
            "No human labels, customer traffic, paid API, GPU, vLLM, or external OpenCompass runner was used.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    try:
        checksum_entries = verify_checksum_file()
        summary = build_summary()
        if args.write_summary:
            (ROOT / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        result = {
            "ok": True,
            "checksum_entries_verified": checksum_entries,
            "summary": summary,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (VerificationError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
