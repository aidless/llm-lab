#!/usr/bin/env python3
"""Independently verify contamination metrics, gate state and SHA-256 coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT
EXCLUDED = {"SHA256SUMS", "verification-result.json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain an object")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_number} must contain an object")
        rows.append(payload)
    return rows


def _wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 1.0}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _suspect(match: dict[str, Any], ngram_threshold: float, semantic_threshold: float) -> bool:
    return bool(
        match["exact"]
        or float(match["ngram_score"]) >= ngram_threshold
        or float(match["semantic_score"]) >= semantic_threshold
    )


def _predict(
    rows: list[dict[str, Any]],
    split: str,
    ngram_threshold: float,
    semantic_threshold: float,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if split == "all" or row["split"] == split]
    return [
        {
            **row,
            "suspect": _suspect(row["match"], ngram_threshold, semantic_threshold),
        }
        for row in selected
    ]


def _score(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    positives = [row for row in rows if row["gold_contaminated"]]
    negatives = [row for row in rows if not row["gold_contaminated"]]
    tp = sum(row["suspect"] for row in positives)
    fp = sum(row["suspect"] for row in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": split,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _tune(rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] | None = None
    for ngram_index in range(8, 20):
        for semantic_index in range(10, 20):
            ngram = ngram_index / 20.0
            semantic = semantic_index / 20.0
            score = _score(_predict(rows, "dev", ngram, semantic), "dev")
            candidate = {
                "ngram_threshold": ngram,
                "semantic_threshold": semantic,
                "dev_f1": score["f1"],
                "dev_precision": score["precision"],
                "dev_recall": score["recall"],
            }
            rank = (
                candidate["dev_f1"],
                candidate["dev_precision"],
                -candidate["ngram_threshold"],
                -candidate["semantic_threshold"],
            )
            if best is None or rank > (
                best["dev_f1"],
                best["dev_precision"],
                -best["ngram_threshold"],
                -best["semantic_threshold"],
            ):
                best = candidate
    if best is None:
        raise ValueError("threshold tuning produced no result")
    return best


def _resolve_tasks_path(manifest: dict[str, Any]) -> Path:
    raw = Path(str(manifest["tasks"]["path"]))
    return raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _validate_human_reviews(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"reviewed_records": 0, "gate_40_reviews": False, "reviewer_ids": []}
    rows = _jsonl(path)
    seen: set[str] = set()
    reviewers: set[str] = set()
    for index, row in enumerate(rows, 1):
        task_id = str(row.get("task_id", ""))
        if not task_id or task_id in seen:
            raise ValueError(f"human review {index} has a missing or duplicate task id")
        if not isinstance(row.get("contaminated"), bool):
            raise ValueError(f"human review {task_id} lacks a boolean contamination label")
        reviewer = str(row.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError(f"human review {task_id} has no reviewer")
        reviewed_at = str(row.get("reviewed_at_utc", "")).strip()
        reviewed_dt = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        if reviewed_dt.tzinfo is None:
            raise ValueError(f"human review {task_id} timestamp lacks timezone")
        seen.add(task_id)
        reviewers.add(reviewer)
    return {
        "reviewed_records": len(seen),
        "gate_40_reviews": len(seen) >= 40,
        "reviewer_ids": sorted(reviewers),
    }


def _expected(
    rows: list[dict[str, Any]],
    *,
    human_review: dict[str, Any],
) -> dict[str, Any]:
    thresholds = _tune(rows)
    test_rows = _predict(rows, "test", thresholds["ngram_threshold"], thresholds["semantic_threshold"])
    dev_rows = _predict(rows, "dev", thresholds["ngram_threshold"], thresholds["semantic_threshold"])
    test_confusion = _score(test_rows, "test")
    dev_confusion = _score(dev_rows, "dev")
    positives = [row for row in test_rows if row["gold_contaminated"]]
    negatives = [row for row in test_rows if not row["gold_contaminated"]]
    false_positives = [row for row in negatives if row["suspect"]]
    false_negatives = [row for row in positives if not row["suspect"]]
    fp_interval = _wilson(len(false_positives), len(negatives))
    fn_interval = _wilson(len(false_negatives), len(positives))
    metrics = {
        "contamination_rate": test_confusion["tp"] / len(test_rows) if test_rows else None,
        "false_positive_rate": len(false_positives) / len(negatives) if negatives else None,
        "false_negative_rate": len(false_negatives) / len(positives) if positives else None,
        "precision": test_confusion["precision"],
        "recall": test_confusion["recall"],
        "f1": test_confusion["f1"],
        "false_positive_rate_wilson_95": fp_interval,
        "false_negative_rate_wilson_95": fn_interval,
    }
    gate_reasons: list[str] = []
    if not human_review["gate_40_reviews"]:
        gate_reasons.append("human_review_lt_40")
    if test_confusion["recall"] < 0.85:
        gate_reasons.append("test_recall_lt_0.85")
    if test_confusion["precision"] < 0.85:
        gate_reasons.append("test_precision_lt_0.85")
    if fn_interval["high"] > 0.30:
        gate_reasons.append("false_negative_wilson_upper_gt_0.30")
    if fp_interval["high"] > 0.12:
        gate_reasons.append("false_positive_wilson_upper_gt_0.12")
    decision = (
        "BLOCK"
        if test_confusion["recall"] < 0.70 or fn_interval["high"] > 0.35
        else "HOLD"
        if gate_reasons
        else "PASS"
    )
    return {
        "thresholds": thresholds,
        "test_confusion": test_confusion,
        "dev_confusion": dev_confusion,
        "metrics": metrics,
        "decision": decision,
        "gate_reasons": gate_reasons,
        "false_positive_samples": [row["task_id"] for row in false_positives[:20]],
        "false_negative_samples": [row["task_id"] for row in false_negatives[:20]],
        "human_review": human_review,
    }


def verify(
    *,
    dataset_dir: Path,
    evidence_dir: Path,
    human_reviews_path: Path,
    allow_pending_human: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = _json(dataset_dir / "manifest.json")
    report = _json(evidence_dir / "report.json")
    rows = _jsonl(evidence_dir / "matches.jsonl")
    labels = _jsonl(dataset_dir / str(manifest["labels"]["path"]))
    corpus = _jsonl(dataset_dir / str(manifest["corpus"]["path"]))
    tasks_path = _resolve_tasks_path(manifest)

    for name, path, metadata, loaded in (
        ("corpus", dataset_dir / str(manifest["corpus"]["path"]), manifest["corpus"], corpus),
        ("labels", dataset_dir / str(manifest["labels"]["path"]), manifest["labels"], labels),
    ):
        if int(metadata["count"]) != len(loaded):
            errors.append(f"{name} count does not match manifest")
        if _sha256(path) != str(metadata["sha256"]):
            errors.append(f"{name} SHA-256 does not match manifest")
    if not tasks_path.exists():
        errors.append(f"frozen task file is missing: {tasks_path}")
    elif _sha256(tasks_path) != str(manifest["tasks"]["sha256"]):
        errors.append("frozen task SHA-256 does not match manifest")
    if len(corpus) != int(manifest["corpus"]["count"]):
        errors.append("corpus row count mismatch")
    if len(labels) != 201:
        errors.append(f"expected 201 labels, got {len(labels)}")
    if len(rows) != 201:
        errors.append(f"expected 201 scan rows, got {len(rows)}")

    label_by_id = {str(row["task_id"]): row for row in labels}
    row_by_id = {str(row["task_id"]): row for row in rows}
    if set(label_by_id) != set(row_by_id):
        errors.append("scan task IDs do not exactly match the frozen labels")
    for task_id, row in row_by_id.items():
        gold = label_by_id.get(task_id)
        if gold is None:
            continue
        for key in ("split", "gold_contaminated", "seed_type"):
            if row.get(key) != gold.get(key):
                errors.append(f"{task_id}: scan row {key} differs from the frozen label")
                break

    try:
        human_review = _validate_human_reviews(human_reviews_path)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        human_review = {"reviewed_records": 0, "gate_40_reviews": False, "reviewer_ids": []}
        errors.append(f"human review invalid: {exc}")
    expected = _expected(rows, human_review=human_review)
    for key in (
        "thresholds",
        "test_confusion",
        "dev_confusion",
        "metrics",
        "decision",
        "gate_reasons",
        "false_positive_samples",
        "false_negative_samples",
    ):
        if report.get(key) != expected[key]:
            errors.append(f"report {key} does not match independent recomputation")
    if report.get("human_review") != human_review:
        errors.append("report human_review does not match the reviewed record set")
    if not human_review["gate_40_reviews"] and not allow_pending_human:
        errors.append("human review has fewer than 40 records")

    preflight_path = evidence_dir / "evalops-preflight.json"
    if not preflight_path.exists():
        errors.append("missing evalops-preflight.json")
    else:
        preflight = _json(preflight_path)
        if preflight.get("decision") != expected["decision"]:
            errors.append("EvalOps preflight decision does not match recomputation")
        if preflight.get("reasons") != expected["gate_reasons"]:
            errors.append("EvalOps preflight reasons do not match recomputation")

    checksum_path = evidence_dir / "SHA256SUMS"
    checksum_count = 0
    listed_targets: set[str] = set()
    if not checksum_path.exists():
        errors.append("missing evidence/SHA256SUMS")
    else:
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"SHA256SUMS:{line_number} malformed")
                continue
            expected_hash, relative = parts
            target = evidence_dir / relative.strip()
            listed_targets.add(relative.strip())
            if not target.is_file():
                errors.append(f"SHA256SUMS target missing: {relative}")
            elif _sha256(target) != expected_hash:
                errors.append(f"SHA256SUMS mismatch: {relative}")
            else:
                checksum_count += 1
        actual_targets = {
            path.relative_to(evidence_dir).as_posix()
            for path in evidence_dir.rglob("*")
            if path.is_file() and path.relative_to(evidence_dir).as_posix() not in EXCLUDED
        }
        if listed_targets != actual_targets:
            errors.append("SHA256SUMS does not cover exactly the evidence files")

    tools_path = evidence_dir / "tool-versions.json"
    if not tools_path.exists():
        errors.append("missing tool-versions.json")
    else:
        tools = _json(tools_path)
        if tools.get("embedding_model") != "nomic-embed-text:latest":
            errors.append("embedding model provenance mismatch")
        if not isinstance(tools.get("ollama_version"), str):
            errors.append("Ollama version provenance is missing")
        if tools.get("task_count") != 201 or tools.get("document_count") != 240:
            errors.append("embedding input counts do not match the frozen fixture")

    technical_ok = not errors
    acceptance_ok = technical_ok and human_review["gate_40_reviews"] and expected["decision"] == "PASS"
    result = {
        "schema_version": "benchmark-contamination.verification.v1",
        "technical_ok": technical_ok,
        "acceptance_ok": acceptance_ok,
        "human_gate_pending": not human_review["gate_40_reviews"],
        "checks": {
            "task_count": len(rows),
            "label_count": len(labels),
            "corpus_count": len(corpus),
            "decision": expected["decision"],
            "gate_reasons": expected["gate_reasons"],
            "metrics": expected["metrics"],
            "human_review": human_review,
            "checksum_files_verified": checksum_count,
        },
        "errors": errors,
        "boundary": "本地安全评测，不是生产安全系统。 Local seeded-fixture audit, not production contamination proof.",
    }
    (evidence_dir / "verification-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--evidence", default=str(ROOT / "evidence"))
    parser.add_argument("--human-reviews", default=str(ROOT / "review" / "reviewed.jsonl"))
    parser.add_argument("--allow-pending-human", action="store_true")
    args = parser.parse_args()
    result = verify(
        dataset_dir=Path(args.dataset),
        evidence_dir=Path(args.evidence),
        human_reviews_path=Path(args.human_reviews),
        allow_pending_human=args.allow_pending_human,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["technical_ok"] and result["human_gate_pending"]:
        return 0 if args.allow_pending_human else 1
    return 0 if result["acceptance_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
