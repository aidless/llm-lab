"""CLI for the contamination preflight."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
from llm_lab.evalops import write_json, write_jsonl

from .data import build_fixture, load_ceval_tasks, read_jsonl
from .embeddings import Embedder, save_vectors
from .reporting import build_report, write_reports
from .scanner import Match, best_match, is_suspect

ROOT = Path(__file__).resolve().parent.parent
LLM_LAB_ROOT = ROOT.parent
DEFAULT_TASKS = LLM_LAB_ROOT / "evalops" / "ceval-201" / "tasks-v1.jsonl"
DEFAULT_KNOWLEDGE = LLM_LAB_ROOT / "knowledge"
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_EVIDENCE = ROOT / "evidence"
DEFAULT_REVIEW = ROOT / "review"


def _task_text(task: dict[str, Any]) -> str:
    choices = task.get("choices") or {}
    return f"{task['question']}\n" + "\n".join(f"{key}. {choices[key]}" for key in sorted(choices))


def command_build(args: argparse.Namespace) -> int:
    manifest = build_fixture(
        tasks_path=Path(args.tasks),
        knowledge_root=Path(args.knowledge),
        output_dir=Path(args.output),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _human_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _tool_provenance(args: argparse.Namespace, *, task_count: int, document_count: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": args.embedding_model,
        "base_url": args.base_url,
        "ngram_n": args.ngram_n,
        "task_count": task_count,
        "document_count": document_count,
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            version = client.get(f"{args.base_url.rstrip('/')}/api/version")
            version.raise_for_status()
            tags = client.get(f"{args.base_url.rstrip('/')}/api/tags")
            tags.raise_for_status()
        payload["ollama_version"] = str(version.json().get("version", "unknown"))
        models = tags.json().get("models", [])
        selected = next(
            (
                item
                for item in models
                if isinstance(item, dict) and str(item.get("name")) == args.embedding_model
            ),
            {},
        )
        payload["embedding_model_digest"] = selected.get("digest")
        payload["embedding_model_details"] = selected.get("details")
    except (httpx.HTTPError, ValueError) as exc:
        payload["ollama_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def command_scan(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    documents = read_jsonl(dataset / manifest["corpus"]["path"])
    labels = {str(row["task_id"]): row for row in read_jsonl(dataset / manifest["labels"]["path"])}
    tasks = load_ceval_tasks(Path(args.tasks))
    texts = [_task_text(task) for task in tasks]
    embedder = Embedder(model=args.embedding_model, base_url=args.base_url)
    try:
        task_vectors = embedder.encode(texts)
        document_vectors = embedder.encode([str(document["text"]) for document in documents])
    finally:
        embedder.close()

    evidence = Path(args.output)
    evidence.mkdir(parents=True, exist_ok=True)
    save_vectors(evidence / "task-embeddings.json", task_vectors)
    save_vectors(evidence / "document-embeddings.json", document_vectors)
    write_json(
        evidence / "tool-versions.json",
        _tool_provenance(args, task_count=len(tasks), document_count=len(documents)),
    )

    rows: list[dict[str, Any]] = []
    for task, text, vector in zip(tasks, texts, task_vectors, strict=True):
        match = best_match(
            task_text=text,
            task_vector=vector,
            documents=documents,
            document_vectors=document_vectors,
            ngram_n=args.ngram_n,
        )
        gold = labels[str(task["id"])]
        rows.append(
            {
                "task_id": str(task["id"]),
                "subject": str(task.get("subject")),
                "split": gold["split"],
                "gold_contaminated": bool(gold["gold_contaminated"]),
                "seed_type": gold.get("seed_type"),
                "match": match.to_dict(),
            }
        )
    write_jsonl(evidence / "matches.jsonl", rows)
    report = build_report(
        rows=rows,
        corpus_manifest=manifest,
        human_reviews=_human_reviews(Path(args.human_reviews)),
    )
    write_reports(evidence, report)
    write_review_queue(
        rows,
        report["thresholds"],
        DEFAULT_REVIEW,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "metrics": report["metrics"],
                "thresholds": report["thresholds"],
                "report": str(evidence / "report.html"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "PASS" else 2


def write_review_queue(
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    output: Path,
) -> None:
    def suspect(row: dict[str, Any]) -> bool:
        return is_suspect(
            Match(**row["match"]),
            ngram_threshold=thresholds["ngram_threshold"],
            semantic_threshold=thresholds["semantic_threshold"],
        )

    suspects = [
        row
        for row in rows
        if suspect(row)
    ]
    controls = [row for row in rows if row not in suspects][:10]
    selected = (suspects + controls)[:40]
    items = []
    for row in selected:
        match = row["match"]
        items.append(
            {
                "review_id": f"contamination:{row['task_id']}",
                "task_id": row["task_id"],
                "subject": row["subject"],
                "suggested_contaminated": suspect(row),
                "match": match,
                "decision": "pending",
                "reviewer": "",
                "reviewed_at_utc": "",
                "contaminated": None,
                "notes": "",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "review-queue.json",
        {
            "schema_version": "benchmark-contamination.review-queue.v1",
            "items": items,
        },
    )


def command_report(args: argparse.Namespace) -> int:
    evidence = Path(args.output)
    rows = read_jsonl(evidence / "matches.jsonl")
    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    report = build_report(
        rows=rows,
        corpus_manifest=manifest,
        human_reviews=_human_reviews(Path(args.human_reviews)),
    )
    write_reports(evidence, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


def command_gate(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    decision = str(report["decision"])
    preflight = {
        "schema_version": "evalops.preflight.v1",
        "source": "benchmark-contamination-scan",
        "decision": decision,
        "reasons": report.get("gate_reasons", []),
        "metrics": report.get("metrics", {}),
        "task_manifest": report.get("manifest", {}).get("tasks", {}),
        "human_review": report.get("human_review", {}),
    }
    output = Path(args.output)
    write_json(output, preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision == "PASS" else 3 if decision == "BLOCK" else 2


def command_ingest_review(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.export).read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise TypeError("review export items must be a list")
    complete = []
    for item in items:
        if item.get("decision") == "pending":
            continue
        if not item.get("reviewer") or not item.get("reviewed_at_utc"):
            raise ValueError("each reviewed item requires reviewer and timestamp")
        if not isinstance(item.get("contaminated"), bool):
            raise ValueError("each reviewed item requires a boolean contamination label")
        complete.append(item)
    write_jsonl(Path(args.output), complete)
    print(json.dumps({"reviewed_records": len(complete)}, indent=2))
    return 0 if len(complete) >= 40 else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="contamination-scan",
        description="Benchmark contamination preflight",
    )
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-fixture")
    build.add_argument("--tasks", default=str(DEFAULT_TASKS))
    build.add_argument("--knowledge", default=str(DEFAULT_KNOWLEDGE))
    build.add_argument("--output", default=str(DEFAULT_DATASET))
    build.set_defaults(func=command_build)
    scan = sub.add_parser("scan")
    scan.add_argument("--tasks", default=str(DEFAULT_TASKS))
    scan.add_argument("--dataset", default=str(DEFAULT_DATASET))
    scan.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    scan.add_argument("--base-url", default="http://127.0.0.1:11434")
    scan.add_argument("--embedding-model", default="nomic-embed-text:latest")
    scan.add_argument("--ngram-n", type=int, default=13)
    scan.add_argument("--human-reviews", default=str(DEFAULT_REVIEW / "reviewed.jsonl"))
    scan.set_defaults(func=command_scan)
    report = sub.add_parser("report")
    report.add_argument("--dataset", default=str(DEFAULT_DATASET))
    report.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    report.add_argument("--human-reviews", default=str(DEFAULT_REVIEW / "reviewed.jsonl"))
    report.set_defaults(func=command_report)
    gate = sub.add_parser("gate")
    gate.add_argument("--report", default=str(DEFAULT_EVIDENCE / "report.json"))
    gate.add_argument("--output", default=str(DEFAULT_EVIDENCE / "evalops-preflight.json"))
    gate.set_defaults(func=command_gate)
    ingest = sub.add_parser("ingest-review")
    ingest.add_argument("--export", required=True)
    ingest.add_argument("--output", default=str(DEFAULT_REVIEW / "reviewed.jsonl"))
    ingest.set_defaults(func=command_ingest_review)
    return root


def entry() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    entry()
