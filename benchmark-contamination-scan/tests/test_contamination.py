from __future__ import annotations

import shutil
from pathlib import Path

from contamination_scan.data import build_fixture, normalize_text
from contamination_scan.scanner import char_ngrams, ngram_containment
from verify_evidence import verify


def test_normalization_handles_unicode_and_punctuation() -> None:
    assert normalize_text("  Ａ， B!\n") == normalize_text("A B")
    assert normalize_text("你好，世界！") == "你好世界"


def test_ngram_detects_copy_and_distinguishes_unrelated_text() -> None:
    source = "现有一棵无重复关键字的AVL树，对其进行中序遍历可得到一个降序序列。"
    related = "Study note: 现有一棵无重复关键字的AVL树，对其进行中序遍历可得到一个降序序列。"
    unrelated = "The service uses bounded retries and an in-memory health check."

    assert char_ngrams(source)
    assert ngram_containment(source, related) == 1.0
    assert ngram_containment(source, unrelated) < 0.1


def test_fixture_has_dev_test_and_60_positive_documents(tmp_path: Path) -> None:
    tasks = Path(__file__).resolve().parents[2] / "evalops" / "ceval-201" / "tasks-v1.jsonl"
    knowledge = Path(__file__).resolve().parents[2] / "knowledge"
    manifest = build_fixture(
        tasks_path=tasks,
        knowledge_root=knowledge,
        output_dir=tmp_path,
    )

    assert manifest["tasks"]["count"] == 201
    assert manifest["corpus"]["seeded_positive_documents"] == 60
    assert manifest["labels"]["count"] == 201
    assert manifest["labels"]["dev_count"] + manifest["labels"]["test_count"] == 201


def test_checked_in_evidence_is_independently_recomputed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "dataset"
    evidence = tmp_path / "evidence"
    shutil.copytree(root / "dataset", dataset)
    shutil.copytree(root / "evidence", evidence)

    result = verify(
        dataset_dir=dataset,
        evidence_dir=evidence,
        human_reviews_path=tmp_path / "missing-human-reviews.jsonl",
        allow_pending_human=True,
    )

    assert result["technical_ok"] is True
    assert result["human_gate_pending"] is True
    assert result["checks"]["checksum_files_verified"] == 8
