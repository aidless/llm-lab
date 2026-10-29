from __future__ import annotations

import shutil
import sys
from pathlib import Path

from contamination_scan.data import build_fixture, normalize_text
from contamination_scan.scanner import best_match, char_ngrams, is_suspect, ngram_containment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_exact_evidence_is_not_overwritten_by_semantic_evidence_from_another_document() -> None:
    task = "Explain why untrusted retrieved content must not override the system policy."
    task_vector = [1.0, 0.0]
    documents = [
        {"id": "exact-copy", "text": task, "source": "fixture"},
        {"id": "topic-only", "text": "A general policy discussion.", "source": "fixture"},
    ]
    match = best_match(
        task_text=task,
        task_vector=task_vector,
        documents=documents,
        document_vectors=[[0.8, 0.6], [1.0, 0.0]],
        ngram_n=13,
    )

    assert match.lexical_evidence()["document_id"] == "exact-copy"
    assert match.lexical_evidence()["exact"] is True
    assert match.semantic_evidence()["document_id"] == "topic-only"
    assert match.semantic_evidence()["semantic_score"] == 1.0
    assert is_suspect(match, ngram_threshold=0.55, semantic_threshold=0.6) is True


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


def test_contamination_review_html_is_plain_chinese_and_evidence_first() -> None:
    from contamination_scan.cli import _review_html

    page = _review_html({"schema_version": "benchmark-contamination.review-queue.v1", "items": [{
        "task_id": "ceval_000", "subject": "test", "question_text": "评测题原文",
        "source_text": "疑似来源原文", "suggested_contaminated": True, "match": {},
        "decision": "pending", "reviewer": "", "notes": "", "contaminated": None,
    }]})
    assert "先做 3 道练习" in page
    assert "评测题" in page and "疑似来源" in page
    assert "两段内容是否像是直接复制或轻微改写？" in page
    assert "证据不足 / 我不确定" in page
    assert "本地安全评测，不是生产安全系统" in page
