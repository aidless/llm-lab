"""Command line interface for the Agent Safety Runtime Monitor."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from llm_lab.evalops import write_json, write_jsonl
from llm_lab.safety.models import (
    SafetyTask,
    load_tasks,
    task_set_hash,
    verify_safety_dataset,
)
from llm_lab.safety.runner import (
    ChatModel,
    SafetyRunConfig,
    ScriptedChatModel,
    Trajectory,
    aggregate_trajectories,
    rescore_with_label,
    run_task,
)

from .dataset_builder import build_dataset
from .ollama_model import OllamaStructuredChatModel
from .output_guard import (
    OutputPathError,
    claim_output,
    preflight_empty_output,
    require_owned_output,
)
from .reporting import build_combined_report, write_reports
from .review import (
    required_trajectory_review_ids,
    summarize_review_records,
    task_labels_from_reviews,
    trajectory_set_hash,
    validate_export,
    write_review_packet,
)
from .review import (
    write_jsonl as write_review_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_EVIDENCE = ROOT / "evidence"
DEFAULT_REVIEW = ROOT / "review"
DEFAULT_MODELS = ["qwen3:0.6b", "llama3.2:1b"]


def _resolve_review_dir(
    evidence: Path,
    review_output: str | None,
) -> Path:
    """Keep the formal review packet beside the evidence it describes."""
    if review_output:
        return Path(review_output).expanduser()
    if evidence.expanduser().resolve() == DEFAULT_EVIDENCE.resolve():
        return DEFAULT_REVIEW
    return evidence.expanduser() / "review"


def _portable_path(evidence: Path, target: Path) -> str:
    base = evidence.expanduser().resolve()
    return Path(os.path.relpath(target.expanduser().resolve(), start=base)).as_posix()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "model"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(payload)
    return rows


def _load_task_reviews(path: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}, []
    reviews = _read_jsonl(path)
    return task_labels_from_reviews(reviews), reviews


def _review_summary(
    reviews: list[dict[str, Any]],
    tasks: list[SafetyTask],
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    return summarize_review_records(
        reviews,
        expected_task_ids={task.id for task in tasks},
        required_trajectory_ids=required_trajectory_review_ids(trajectories),
        task_set_hash=task_set_hash(tasks),
        trajectory_set_hash=trajectory_set_hash(trajectories),
    )


def select_stratified_tasks(tasks: list[SafetyTask], limit: int) -> list[SafetyTask]:
    """Select a balanced smoke sample without relying on file order."""
    categories = sorted({task.category for task in tasks})
    if limit < len(categories):
        raise ValueError(f"stratified smoke needs at least {len(categories)} tasks (one per category), got {limit}")
    grouped: dict[str, list[SafetyTask]] = {}
    for category in categories:
        grouped[category] = sorted(
            (task for task in tasks if task.category == category),
            key=lambda task: (
                not task.draft_label.benign,
                not task.draft_label.must_halt,
                task.id,
            ),
        )
    positions = {category: 0 for category in categories}
    selected: list[SafetyTask] = []
    while len(selected) < limit:
        progressed = False
        for category in categories:
            index = positions[category]
            if index >= len(grouped[category]):
                continue
            selected.append(grouped[category][index])
            positions[category] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected_ids = {task.id for task in selected}
    return [task for task in tasks if task.id in selected_ids]


def _environment(models: list[str], *, adapter: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "mock_only": True,
        "models": {},
        "ollama": {},
    }
    if adapter == "deterministic":
        payload["models"] = {model: {"fixture": True} for model in models}
        return payload
    try:
        with httpx.Client(timeout=10.0) as client:
            tags = client.get("http://127.0.0.1:11434/api/tags").json()
            version = client.get("http://127.0.0.1:11434/api/version").json()
        payload["ollama"]["version"] = version
        by_name = {str(item.get("name")): item for item in tags.get("models", []) if isinstance(item, dict)}
        for model in models:
            entry = by_name.get(model, {})
            payload["models"][model] = {
                "digest": entry.get("digest"),
                "size_bytes": entry.get("size"),
                "details": entry.get("details"),
                "capabilities": entry.get("capabilities"),
            }
    except Exception as exc:  # captured as evidence, never hidden
        payload["ollama"]["error"] = f"{type(exc).__name__}: {exc}"
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise FileNotFoundError("nvidia-smi")
        process = subprocess.run(  # noqa: S603 - fixed executable discovered locally
            [nvidia_smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload["nvidia_smi"] = {
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        payload["nvidia_smi"] = {"error": f"{type(exc).__name__}: {exc}"}
    return payload


def _ensure_output(output: Path, *, force: bool) -> None:
    try:
        claim_output(output, "evidence", replace=force)
    except OutputPathError as exc:
        raise SystemExit(str(exc)) from exc


def command_build(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        claim_output(output, "dataset", replace=args.force)
    except OutputPathError as exc:
        raise SystemExit(str(exc)) from exc
    manifest = build_dataset(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_verify_dataset(args: argparse.Namespace) -> int:
    result = verify_safety_dataset(Path(args.dataset))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


def command_run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    evidence = Path(args.output)
    review_dir = _resolve_review_dir(evidence, args.review_output)
    verification = verify_safety_dataset(dataset)
    if not verification.get("ok"):
        raise SystemExit(f"dataset verification failed: {verification}")
    invalid_models = [model for model in args.models if not model.strip() or any(character.isspace() for character in model)]
    if invalid_models:
        raise SystemExit(f"model identifiers must be non-empty and contain no whitespace: {invalid_models}")
    tasks = load_tasks(dataset / "tasks-v1.jsonl")
    if args.max_tasks and args.stratified_smoke:
        raise SystemExit("use either --max-tasks or --stratified-smoke, not both")
    selection_mode = "full"
    if args.stratified_smoke:
        try:
            tasks = select_stratified_tasks(tasks, args.stratified_smoke)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        selection_mode = f"stratified-{len(tasks)}"
    elif args.max_tasks:
        tasks = tasks[: args.max_tasks]
        selection_mode = f"head-{len(tasks)}"
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    if not set(policies).issubset({"off", "on"}):
        raise SystemExit("--policies must contain only off and/or on")
    reviews, review_rows = _load_task_reviews(Path(args.reviews) if args.reviews else None)
    try:
        preflight_empty_output(review_dir, "review")
    except OutputPathError as exc:
        raise SystemExit(str(exc)) from exc
    _ensure_output(evidence, force=args.force)
    write_json(
        evidence / "environment.json",
        _environment(args.models, adapter=args.adapter),
    )
    selection = {
        "mode": selection_mode,
        "task_count": len(tasks),
        "task_ids": [task.id for task in tasks],
        "category_counts": dict(Counter(task.category for task in tasks)),
        "benign_controls": sum(task.draft_label.benign for task in tasks),
        "must_halt": sum(task.draft_label.must_halt for task in tasks),
    }
    write_json(evidence / "run-selection.json", selection)

    runs: list[dict[str, Any]] = []
    trajectories_all: list[dict[str, Any]] = []
    for model in args.models:
        client: ChatModel
        if args.adapter == "deterministic":
            client = ScriptedChatModel(
                {},
                model=model,
                provider="deterministic",
            )
        else:
            client = OllamaStructuredChatModel(
                base_url=args.base_url,
                think=args.think,
            )
        try:
            for policy_mode in policies:
                config = SafetyRunConfig(
                    provider=args.adapter,
                    model=model,
                    base_url=args.base_url,
                    policy_mode=policy_mode,
                    timeout_seconds=args.timeout,
                    max_retries=args.retries,
                    backoff_seconds=0.5,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                )
                print(
                    f"[agent-safety] model={model} policy={policy_mode} tasks={len(tasks)}",
                    file=sys.stderr,
                    flush=True,
                )
                started = time.perf_counter()
                trajectories = []
                for index, task in enumerate(tasks, 1):
                    trajectory = run_task(
                        task,
                        client,
                        config,
                        review=reviews.get(task.id),
                    )
                    trajectories.append(trajectory)
                    if index % 8 == 0 or index == len(tasks):
                        print(
                            f"  {index}/{len(tasks)} trajectories",
                            file=sys.stderr,
                            flush=True,
                        )
                wall_seconds = time.perf_counter() - started
                aggregate = aggregate_trajectories(
                    trajectories,
                    tasks=tasks,
                    config=config,
                    wall_seconds=wall_seconds,
                )
                run_dir = evidence / "runs" / f"{_slug(model)}-policy-{policy_mode}"
                write_jsonl(
                    run_dir / "trajectories.jsonl",
                    [item.to_dict() for item in trajectories],
                )
                write_json(run_dir / "report.json", aggregate)
                run_payload = {
                    "model": model,
                    "policy_mode": policy_mode,
                    "config": config.to_dict(),
                    "config_hash": config.config_hash(),
                    "wall_seconds": wall_seconds,
                    "task_count": len(tasks),
                    "task_ids": selection["task_ids"],
                    "selection_mode": selection_mode,
                }
                write_json(run_dir / "run-meta.json", run_payload)
                runs.append(aggregate)
                trajectories_all.extend(item.to_dict() for item in trajectories)
        finally:
            if hasattr(client, "close"):
                client.close()

    write_jsonl(evidence / "trajectories.jsonl", trajectories_all)
    review_summary = _review_summary(review_rows, tasks, trajectories_all)
    combined = build_combined_report(
        runs=runs,
        task_count=len(tasks),
        task_hash=task_set_hash(tasks),
        category_counts=dict(Counter(task.category for task in tasks)),
        review_summary=review_summary,
    )
    combined["dataset"]["selection"] = selection
    write_reports(str(evidence), combined)
    review_result = write_review_packet(
        review_dir,
        tasks,
        trajectories_all,
        trajectory_limit=24,
    )
    review_result["review_dir"] = _portable_path(evidence, review_dir)
    review_result["review_html"] = _portable_path(evidence, review_dir / "review.html")
    review_result["review_queue"] = _portable_path(evidence, review_dir / "review-queue.json")
    write_json(evidence / "review-packet.json", review_result)
    print(
        json.dumps(
            {
                "ok": True,
                "runs": len(runs),
                "trajectories": len(trajectories_all),
                "report": str(evidence / "report.html"),
                "review_packet": str(review_dir / "review.html"),
                "human_review_complete": review_summary.get("human_gate_complete", False),
                "unsafe_review_coverage": review_summary.get("unsafe_review_coverage", 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_report(args: argparse.Namespace) -> int:
    evidence = Path(args.output)
    dataset = Path(args.dataset)
    try:
        require_owned_output(evidence, "evidence")
    except OutputPathError as exc:
        raise SystemExit(str(exc)) from exc
    tasks = load_tasks(dataset / "tasks-v1.jsonl")
    task_by_id = {task.id: task for task in tasks}
    review_path = Path(args.reviews) if args.reviews else None
    reviews, review_rows = _load_task_reviews(review_path)
    run_reports: list[dict[str, Any]] = []
    trajectories_all: list[dict[str, Any]] = []
    selected_task_ids: set[str] | None = None
    for run_dir in sorted(path.parent for path in (evidence / "runs").glob("*/report.json")):
        meta = json.loads((run_dir / "run-meta.json").read_text(encoding="utf-8"))
        config = SafetyRunConfig(**meta["config"])
        raw_trajectories = _read_jsonl(run_dir / "trajectories.jsonl")
        scored: list[Trajectory] = []
        for payload in raw_trajectories:
            trajectory = Trajectory.from_dict(payload)
            review = reviews.get(trajectory.task_id)
            if review is not None:
                trajectory = rescore_with_label(
                    trajectory,
                    label=review,
                    label_source="human-review",
                )
            scored.append(trajectory)
        matching_tasks = [task_by_id[item.task_id] for item in scored if item.task_id in task_by_id]
        run_task_ids = {item.task_id for item in scored}
        if selected_task_ids is None:
            selected_task_ids = run_task_ids
        elif selected_task_ids != run_task_ids:
            raise SystemExit(f"run task selection differs in {run_dir}")
        report = aggregate_trajectories(
            scored,
            tasks=matching_tasks,
            config=config,
            wall_seconds=float(meta.get("wall_seconds") or 0.0),
        )
        write_json(run_dir / "report.json", report)
        run_reports.append(report)
        trajectories_all.extend(item.to_dict() for item in scored)
    if not run_reports:
        raise SystemExit(f"no run reports found below {evidence / 'runs'}")
    selected_tasks = [task for task in tasks if selected_task_ids is not None and task.id in selected_task_ids]
    review_summary = _review_summary(review_rows, selected_tasks, trajectories_all)
    if review_path is not None:
        write_json(
            review_path.with_name("reviews-summary.json"),
            review_summary,
        )
    write_jsonl(evidence / "trajectories-rescored.jsonl", trajectories_all)
    combined = build_combined_report(
        runs=run_reports,
        task_count=len(selected_tasks),
        task_hash=task_set_hash(selected_tasks),
        category_counts=dict(Counter(task.category for task in selected_tasks)),
        review_summary=review_summary,
    )
    selection_path = evidence / "run-selection.json"
    if selection_path.exists():
        combined["dataset"]["selection"] = json.loads(selection_path.read_text(encoding="utf-8"))
    write_reports(str(evidence), combined)
    print(str(evidence / "report.json"))
    return 0


def command_prepare_review(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    tasks = load_tasks(dataset / "tasks-v1.jsonl")
    trajectories = _read_jsonl(Path(args.trajectories))
    result = write_review_packet(
        Path(args.output),
        tasks,
        trajectories,
        trajectory_limit=args.trajectory_limit,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_ingest_review(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    tasks = load_tasks(dataset / "tasks-v1.jsonl")
    trajectories = _read_jsonl(Path(args.trajectories))
    reviews, summary = validate_export(
        Path(args.export),
        expected_task_ids={task.id for task in tasks},
        required_trajectory_ids=required_trajectory_review_ids(trajectories),
        expected_task_set_hash=task_set_hash(tasks),
        expected_trajectory_set_hash=trajectory_set_hash(trajectories),
    )
    output = Path(args.output)
    try:
        claim_output(output.parent, "review", allow_existing_owned=True)
    except OutputPathError as exc:
        raise SystemExit(str(exc)) from exc
    write_review_jsonl(output, reviews)
    write_json(output.with_name("reviews-summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["human_gate_complete"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-safety-eval",
        description="Mock-only Agent Safety runtime evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Freeze the synthetic 96-task dataset")
    build.add_argument("--output", default=str(DEFAULT_DATASET))
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=command_build)

    verify_dataset = sub.add_parser("verify-dataset")
    verify_dataset.add_argument("--dataset", default=str(DEFAULT_DATASET))
    verify_dataset.set_defaults(func=command_verify_dataset)

    run = sub.add_parser("run", help="Run local Ollama models and write evidence")
    run.add_argument("--dataset", default=str(DEFAULT_DATASET))
    run.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    run.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    run.add_argument(
        "--adapter",
        choices=("ollama", "deterministic"),
        default="ollama",
    )
    run.add_argument("--policies", default="off,on")
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--reviews", default="")
    run.add_argument("--timeout", type=float, default=90.0)
    run.add_argument("--retries", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=160)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--think", action="store_true")
    run.add_argument("--max-tasks", type=int, default=0)
    run.add_argument("--stratified-smoke", type=int, default=0)
    run.add_argument(
        "--review-output",
        default=None,
        help=(
            "Human-review packet directory. Defaults to agent-safety-eval/review "
            "for the formal evidence directory, or <output>/review otherwise."
        ),
    )
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=command_run)

    report = sub.add_parser("report", help="Recompute JSON/Markdown/HTML from saved runs")
    report.add_argument("--dataset", default=str(DEFAULT_DATASET))
    report.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    report.add_argument("--reviews", default="")
    report.set_defaults(func=command_report)

    prepare = sub.add_parser("prepare-review", help="Create the static human-review packet")
    prepare.add_argument("--dataset", default=str(DEFAULT_DATASET))
    prepare.add_argument("--trajectories", default=str(DEFAULT_EVIDENCE / "trajectories.jsonl"))
    prepare.add_argument("--output", default=str(DEFAULT_REVIEW))
    prepare.add_argument("--trajectory-limit", type=int, default=24)
    prepare.set_defaults(func=command_prepare_review)

    ingest = sub.add_parser("ingest-review", help="Validate a human review export")
    ingest.add_argument("--export", required=True)
    ingest.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ingest.add_argument("--trajectories", default=str(DEFAULT_EVIDENCE / "trajectories.jsonl"))
    ingest.add_argument("--output", default=str(DEFAULT_REVIEW / "reviews-human.jsonl"))
    ingest.set_defaults(func=command_ingest_review)
    return parser


def entry() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    entry()
