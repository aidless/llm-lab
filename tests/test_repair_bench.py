"""Focused regression tests for the execution-based repair benchmark."""

from __future__ import annotations

from selfevolve.repair_bench import (
    TASKS, _balanced_smoke_tasks, _ollama_candidate, _parse_field_value,
    _run_test, run_benchmark,
)


def test_repair_bank_has_frozen_category_balance() -> None:
    assert len(TASKS) == 20
    assert {task.category for task in TASKS} == {"exception", "boundary", "state", "tool_error"}
    for task in TASKS:
        assert _run_test(task.source, task.public_test)["passed"] is True
        assert _run_test(task.source, task.hidden_test)["passed"] is False


def test_no_change_arm_never_counts_public_only_as_resolved() -> None:
    result = run_benchmark("no_change")
    assert result["task_count"] == 20
    assert result["resolved_count"] == 0
    assert all(not case["resolved"] for case in result["cases"])


def test_smoke_suite_is_fixed_and_balanced() -> None:
    tasks = _balanced_smoke_tasks()
    assert len(tasks) == 10
    assert [task.category for task in tasks].count("exception") == 3
    assert [task.category for task in tasks].count("boundary") == 3
    assert [task.category for task in tasks].count("state") == 2
    assert [task.category for task in tasks].count("tool_error") == 2


def test_harness_marks_bad_source_as_failure() -> None:
    outcome = _run_test("def broken(:\n", "import target\n")
    assert outcome["passed"] is False


def test_field_parser_handles_common_types() -> None:
    assert _parse_field_value("false") is False
    assert _parse_field_value("true") is True
    assert _parse_field_value("42") == 42
    assert _parse_field_value("1.5") == 1.5
    assert _parse_field_value("text") == "text"


def test_ollama_candidate_forwards_extra_fields() -> None:
    class _FakeBackend:
        def __init__(self) -> None:
            self.captured: dict[str, object] | None = None

        def generate(self, prompt: str, max_tokens: int = 256,
                     temperature: float = 0.2, seed: int | None = None,
                     extra_fields: dict[str, object] | None = None) -> str:
            self.captured = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "extra_fields": extra_fields,
            }
            return "```python\ndef f():\n    pass\n```"

    backend = _FakeBackend()
    candidate, status = _ollama_candidate(
        TASKS[0], backend, extra_fields={"think": False},
    )
    assert status == "candidate"
    assert candidate is not None
    assert backend.captured is not None
    assert backend.captured["extra_fields"] == {"think": False}
    assert backend.captured["max_tokens"] == 2048
