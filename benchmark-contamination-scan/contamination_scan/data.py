"""Frozen fixture construction and dataset loading."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def load_ceval_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("task row must be an object")
            tasks.append(payload)
    if len(tasks) != 201:
        raise ValueError(f"expected 201 C-Eval tasks, got {len(tasks)}")
    return tasks


def portable_project_path(path: Path) -> str:
    """Store task provenance without a developer workstation path."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = Path("..") / resolved.relative_to(PROJECT_ROOT.parent)
    return relative.as_posix()


def _task_text(task: dict[str, Any]) -> str:
    choices = task.get("choices") or {}
    choice_text = "\n".join(f"{key}. {choices[key]}" for key in sorted(choices))
    return f"{task['question']}\n{choice_text}"


def _paragraphs(knowledge_root: Path, limit: int) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(knowledge_root.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if 120 <= len(chunk.strip()) <= 1200]
        for chunk in chunks:
            documents.append(
                {
                    "id": f"local-{len(documents):04d}",
                    "text": chunk,
                    "source": path.name,
                    "gold_contaminated": False,
                    "gold_task_id": None,
                    "seed_type": "unrelated_local_corpus",
                }
            )
            if len(documents) >= limit:
                return documents
    if len(documents) < limit:
        while len(documents) < limit:
            index = len(documents)
            documents.append(
                {
                    "id": f"synthetic-{index:04d}",
                    "text": (
                        f"Local engineering note {index}. A deterministic fixture about "
                        "deployment health checks, bounded retries, explicit ownership and "
                        "auditable evidence. This paragraph is intentionally unrelated to "
                        "the C-Eval validation questions."
                    ),
                    "source": "synthetic-fixture",
                    "gold_contaminated": False,
                    "gold_task_id": None,
                    "seed_type": "unrelated_synthetic",
                }
            )
    return documents


def build_fixture(
    *,
    tasks_path: Path,
    knowledge_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tasks = load_ceval_tasks(tasks_path)
    positive_indices = [index for index in range(len(tasks)) if index % 3 == 0][:60]
    positive_ids = {str(tasks[index]["id"]) for index in positive_indices}
    documents = _paragraphs(knowledge_root, limit=180)

    split_rows: list[dict[str, Any]] = []
    for position, index in enumerate(positive_indices):
        task = tasks[index]
        task_text = _task_text(task)
        seed_type = ("exact", "near", "paraphrase")[position % 3]
        if seed_type == "exact":
            text = task_text
        elif seed_type == "near":
            choices = task.get("choices") or {}
            text = f"Study note copied with minor formatting.\nQuestion: {task['question']}\n" + "\n".join(
                f"Option {key}: {choices[key]}" for key in sorted(choices)
            )
        else:
            choices = task.get("choices") or {}
            text = (
                f"Reviewer paraphrase of {task.get('subject', 'a benchmark item')}: "
                f"{task['question'][:160]} "
                "The candidate answers are summarized as "
                + "; ".join(str(choices[key])[:80] for key in sorted(choices))
                + "."
            )
        documents.append(
            {
                "id": f"seed-{task['id']}",
                "text": text,
                "source": "seeded-c-eval-copy",
                "gold_contaminated": True,
                "gold_task_id": str(task["id"]),
                "seed_type": seed_type,
            }
        )

        split = "dev" if position < 30 else "test"
        split_rows.append(
            {
                "task_id": str(task["id"]),
                "gold_contaminated": True,
                "split": split,
                "seed_type": seed_type,
            }
        )
    remaining = [task for task in tasks if str(task["id"]) not in positive_ids]
    for position, task in enumerate(remaining):
        split_rows.append(
            {
                "task_id": str(task["id"]),
                "gold_contaminated": False,
                "split": "dev" if position < 70 else "test",
                "seed_type": None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus-v1.jsonl"
    labels_path = output_dir / "labels-v1.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
    with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(split_rows, key=lambda item: item["task_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "benchmark-contamination.fixture.v1",
        "tasks": {
            "path": portable_project_path(tasks_path),
            "path_base": "benchmark-contamination-scan project root",
            "count": len(tasks),
            "sha256": sha256_text(tasks_path.read_text(encoding="utf-8")),
        },
        "corpus": {
            "path": corpus_path.name,
            "count": len(documents),
            "sha256": sha256_text(corpus_path.read_text(encoding="utf-8")),
            "seeded_positive_documents": 60,
            "unrelated_documents": len(documents) - 60,
        },
        "labels": {
            "path": labels_path.name,
            "count": len(split_rows),
            "sha256": sha256_text(labels_path.read_text(encoding="utf-8")),
            "dev_count": sum(row["split"] == "dev" for row in split_rows),
            "test_count": sum(row["split"] == "test" for row in split_rows),
        },
        "limitations": [
            "The local corpus is a seeded audit fixture, not a model's pretraining corpus.",
            "Gold labels are known only because this fixture intentionally injects copies.",
            "Do not describe fixture metrics as evidence about any specific pretrained model.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_number} is not an object")
        rows.append(payload)
    return rows
