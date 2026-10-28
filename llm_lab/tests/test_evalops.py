"""Tests for the frozen-task batch EvalOps contract."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from llm_lab.evalops import (
    ChoiceParseError,
    DeterministicAdapter,
    EvalConfig,
    EvalOpsError,
    EvalTask,
    FaultInjectionAdapter,
    ProviderResponse,
    build_dataset_from_records,
    load_tasks,
    parse_choice,
    read_records,
    run_eval,
    run_one,
    verify_dataset,
)


def _task(task_id: str, *, gold: str = "A", subject: str = "math") -> EvalTask:
    return EvalTask(
        id=task_id,
        subject=subject,
        question=f"Question {task_id}?\nA. one\nB. two\nC. three\nD. four\n\n只输出一个字母。",
        choices={"A": "one", "B": "two", "C": "three", "D": "four"},
        gold=gold,
    )


def _source_row(task_id: str, *, gold: str = "A", subject: str = "math") -> dict:
    return {
        "id": task_id,
        "group": subject,
        "question": f"Question {task_id}?\nA. one\nB. two\nC. three\nD. four\n\n只输出一个字母。",
        "gold": gold,
    }


def _parquet_row(index: int, *, question: str = "Question t1?", answer: str = "A") -> dict:
    return {
        "id": index,
        "question": question,
        "A": "one",
        "B": "two",
        "C": "three",
        "D": "four",
        "answer": answer,
        "explanation": "",
    }


def test_parse_choice_is_strict_and_accepts_common_answer_prefixes() -> None:
    assert parse_choice("B") == "B"
    assert parse_choice("答案：C") == "C"
    assert parse_choice("Answer: D.") == "D"
    with pytest.raises(ChoiceParseError):
        parse_choice("B. because the first option is wrong")
    with pytest.raises(ChoiceParseError):
        parse_choice("")


def test_build_dataset_verifies_question_options_and_gold(tmp_path: Path) -> None:
    source = [_source_row("t1")]
    parquet_rows = [
        _parquet_row(index, question=f"Question filler-{index}?") for index in range(3)
    ]
    parquet_rows.append(_parquet_row(3, question="Question t1?"))
    manifest = build_dataset_from_records(
        source,
        {"math": parquet_rows},
        output_dir=tmp_path,
        source_metadata={
            "dataset": "fixture",
            "source_jsonl_sha256": "source-sha",
            "parquet_files": {"math": {"file": "ceval_math.parquet", "sha256": "parquet-sha"}},
        },
    )
    result = verify_dataset(tmp_path)

    assert result["ok"] is True
    assert result["task_count"] == 1
    assert manifest["mapping_verified"] is True
    assert (tmp_path / "tasks-v1.jsonl").exists()
    assert (tmp_path / "NOTICE.md").exists()
    assert (tmp_path / "opencompass" / "ceval-201qa.jsonl").exists()
    opencompass_meta = tmp_path / "opencompass" / "ceval-201qa.jsonl.meta.json"
    assert opencompass_meta.exists()
    assert not (tmp_path / "opencompass" / "ceval-201meta.json").exists()
    meta_payload = json.loads(opencompass_meta.read_text(encoding="utf-8"))
    assert meta_payload["data_type"] == "mcq"
    assert meta_payload["infer_method"] == "gen"
    assert load_tasks(tmp_path / "tasks-v1.jsonl")[0].source["row_index"] == 3


def test_build_dataset_rejects_option_mismatch(tmp_path: Path) -> None:
    bad = _parquet_row(1, question="Question t1?")
    bad["B"] = "different"
    with pytest.raises(ValueError, match="choices differ"):
        build_dataset_from_records(
            [_source_row("t1")],
            {"math": [bad]},
            output_dir=tmp_path,
            source_metadata={"source_jsonl_sha256": "x"},
        )


def test_duplicate_task_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(json.dumps(_task("same").to_dict()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(path)


def test_timeout_retry_then_success() -> None:
    task = _task("retry")
    config = EvalConfig(model="fixture", max_retries=1, backoff_seconds=0)
    adapter = FaultInjectionAdapter(
        DeterministicAdapter(),
        {"retry": "timeout"},
        fail_times={"retry": 1},
    )
    record = run_one(task, adapter, config)

    assert record.status == "success"
    assert record.attempts == 2
    assert record.failure_category == "none"
    assert len(record.attempt_latencies_ms) == 2


def test_three_fault_classes_are_classified() -> None:
    tasks = [_task("timeout"), _task("transport"), _task("malformed")]
    config = EvalConfig(model="fixture", max_retries=0)
    adapter = FaultInjectionAdapter(
        DeterministicAdapter(),
        {"timeout": "timeout", "transport": "transport", "malformed": "malformed"},
    )
    report = run_eval(tasks, adapter, config)

    assert report["counts"]["success"] == 0
    assert report["counts"]["failures"] == {
        "parse_error": 1,
        "timeout": 1,
        "transport": 1,
    }


def test_deterministic_runs_have_same_answer_hash_at_different_concurrency(tmp_path: Path) -> None:
    tasks = [_task(f"t{index}", gold="ABCD"[index % 4]) for index in range(12)]
    adapter = DeterministicAdapter(answer_by_id={task.id: task.gold for task in tasks})
    one = run_eval(
        tasks,
        adapter,
        EvalConfig(model="fixture", concurrency=1, max_retries=0),
        output_dir=tmp_path / "one",
    )
    four = run_eval(
        tasks,
        DeterministicAdapter(answer_by_id={task.id: task.gold for task in tasks}),
        EvalConfig(model="fixture", concurrency=4, max_retries=0),
        output_dir=tmp_path / "four",
    )

    assert one["answer_hash"] == four["answer_hash"]
    assert one["task_set_hash"] == four["task_set_hash"]
    assert one["rates"]["strict_accuracy"] == 1.0


def test_resume_skips_existing_records(tmp_path: Path) -> None:
    tasks = [_task("a"), _task("b"), _task("c")]
    adapter = DeterministicAdapter(answer_by_id={task.id: task.gold for task in tasks})
    config = EvalConfig(model="fixture", concurrency=1, max_retries=0)
    first = run_eval(tasks, adapter, config, output_dir=tmp_path)
    assert first["counts"]["success"] == 3

    called: list[str] = []

    class CountingAdapter:
        def complete(self, task: EvalTask, _config: EvalConfig) -> ProviderResponse:
            called.append(task.id)
            return ProviderResponse("A", "counting", "test", 0.0)

    resumed = run_eval(
        tasks,
        CountingAdapter(),
        EvalConfig(model="fixture", concurrency=1, max_retries=0),
        output_dir=tmp_path,
        resume=True,
    )
    assert resumed["counts"]["success"] == 3
    assert called == []


def test_retry_failed_resume_replaces_failures_without_duplicate_rows(tmp_path: Path) -> None:
    tasks = [_task("a"), _task("b"), _task("c")]
    config = EvalConfig(model="fixture", concurrency=1, max_retries=0)
    first = run_eval(
        tasks,
        FaultInjectionAdapter(
            DeterministicAdapter(),
            {"b": "timeout"},
            fail_times={"b": 1},
        ),
        config,
        output_dir=tmp_path,
    )
    assert first["counts"]["success"] == 2
    assert first["counts"]["failures"] == {"timeout": 1}

    called: list[str] = []

    class CountingAdapter:
        def complete(self, task: EvalTask, _config: EvalConfig) -> ProviderResponse:
            called.append(task.id)
            return ProviderResponse("A", "counting", "test", 0.0)

    resumed = run_eval(
        tasks,
        CountingAdapter(),
        config,
        output_dir=tmp_path,
        resume=True,
        retry_failed=True,
    )

    records = read_records(tmp_path / "records.jsonl")
    assert called == ["b"]
    assert resumed["counts"]["success"] == 3
    assert len(records) == 3
    assert len({record.task_id for record in records}) == 3
    assert next(record for record in records if record.task_id == "b").status == "success"


def test_resume_rejects_config_or_task_set_drift(tmp_path: Path) -> None:
    tasks = [_task("a"), _task("b")]
    adapter = DeterministicAdapter()
    config = EvalConfig(model="fixture", concurrency=1, max_retries=0)
    run_eval(tasks, adapter, config, output_dir=tmp_path)
    records_bytes = (tmp_path / "records.jsonl").read_bytes()

    with pytest.raises(EvalOpsError, match="config hash"):
        run_eval(
            tasks,
            DeterministicAdapter(),
            EvalConfig(model="different", concurrency=1, max_retries=0),
            output_dir=tmp_path,
            resume=True,
        )
    assert (tmp_path / "records.jsonl").read_bytes() == records_bytes

    drifted_tasks = [
        EvalTask(
            id=task.id,
            subject=task.subject,
            question=task.question.replace("Question", "Changed question"),
            choices=task.choices,
            gold=task.gold,
        )
        for task in tasks
    ]
    with pytest.raises(EvalOpsError, match="task-set hash"):
        run_eval(
            drifted_tasks,
            DeterministicAdapter(),
            config,
            output_dir=tmp_path,
            resume=True,
        )
    assert (tmp_path / "records.jsonl").read_bytes() == records_bytes


def test_concurrency_is_bounded_and_actually_parallel() -> None:
    tasks = [_task(f"t{index}") for index in range(16)]
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    class ParallelAdapter:
        def complete(self, _task: EvalTask, _config: EvalConfig) -> ProviderResponse:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return ProviderResponse("A", "parallel", "test", 10.0)

    report = run_eval(
        tasks,
        ParallelAdapter(),
        EvalConfig(model="fixture", concurrency=4, max_retries=0),
    )
    assert report["counts"]["success"] == 16
    assert maximum_active == 4


def test_report_contains_hashes_metrics_and_subject_rows(tmp_path: Path) -> None:
    tasks = [_task("a"), _task("b", subject="logic")]
    adapter = DeterministicAdapter(answer_by_id={"a": "A", "b": "C"})
    report = run_eval(
        tasks,
        adapter,
        EvalConfig(model="fixture", concurrency=2, max_retries=0),
        output_dir=tmp_path,
    )

    assert report["schema_version"] == "evalops.report.v1"
    assert len(report["answer_hash"]) == 64
    assert len(report["task_set_hash"]) == 64
    assert report["subject_metrics"]["math"]["total"] == 1
    assert report["subject_metrics"]["logic"]["total"] == 1
    assert report["latency_ms"]["all"]["p50"] >= 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert len(read_records(tmp_path / "records.jsonl")) == 2


def test_malformed_record_file_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(path)


def test_evalops_cli_json_output_is_parseable(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from llm_lab import cli as cli_module

    task_file = tmp_path / "tasks-v1.jsonl"
    task_file.write_text(
        "\n".join(json.dumps(_task(f"t{index}").to_dict()) for index in range(3)) + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli_module.app,
        [
            "evalops",
            "run",
            str(task_file),
            "--output",
            str(tmp_path / "run"),
            "--adapter",
            "deterministic",
            "--concurrency",
            "2",
            "--json",
        ],
        env={"COLUMNS": "40"},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["counts"]["total"] == 3
