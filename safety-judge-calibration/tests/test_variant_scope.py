from __future__ import annotations

from judge_calibration.cli import _slice
from judge_calibration.metrics import summarize


def _record(case_id: str, judge: str, variant: str, repeat_index: int) -> dict:
    return {
        "case_id": case_id,
        "judge": judge,
        "variant": variant,
        "presentation": variant,
        "repeat_index": repeat_index,
        "slice": "normal_workflow",
        "gold_label": "safe",
        "predicted_label": "safe",
        "confidence": 0.8,
        "status": "success",
        "latency_ms": 10.0,
        "reason": "bounded",
        "raw": '{"label":"safe"}',
        "token_usage": {},
    }


def _condition(report: dict, key: str) -> dict:
    matches = [item for item in report["hold_conditions"] if item["key"] == key]
    assert matches, f"missing hold condition {key}"
    return matches[0]


def test_limited_variant_scope_is_recorded_and_still_holds_full_coverage() -> None:
    cases = ["c1", "c2", "c3", "c4"]
    records = []
    for repeat_index in (0, 1):
        for case_id in cases:
            records.append(_record(case_id, "m1", "baseline", repeat_index))
    for case_id in cases[:2]:
        records.append(_record(case_id, "m1", "authority", 0))

    report = summarize(
        records,
        ground_truth_source="human_review",
        expected_case_count=len(cases),
        required_repeats=2,
        expected_variants=["baseline", "authority"],
        variant_case_limit=2,
    )

    assert report["variant_scope"] == {
        "declared_variant_case_limit": 2,
        "expected_variant_cases": 2,
        "full_case_count": 4,
        "scope_limited": True,
    }
    # The declared scope was executed exactly as declared.
    assert _condition(report, "authority:declared_scope_coverage")["passed"] is True
    # Full coverage is still a separate, failing claim.
    assert _condition(report, "authority:full_case_coverage")["passed"] is False
    assert "authority:full_case_coverage" in report["hold_reasons"]
    assert report["decision"] == "HOLD"


def test_full_variant_scope_passes_both_scope_and_coverage() -> None:
    cases = ["c1", "c2"]
    records = []
    for repeat_index in (0, 1):
        for case_id in cases:
            records.append(_record(case_id, "m1", "baseline", repeat_index))
    for case_id in cases:
        records.append(_record(case_id, "m1", "authority", 0))

    report = summarize(
        records,
        ground_truth_source="human_review",
        expected_case_count=len(cases),
        required_repeats=2,
        expected_variants=["baseline", "authority"],
        variant_case_limit=0,
    )

    assert report["variant_scope"]["scope_limited"] is False
    assert _condition(report, "authority:declared_scope_coverage")["passed"] is True
    assert _condition(report, "authority:full_case_coverage")["passed"] is True


def test_case_slice_is_derived_from_the_frozen_case_id() -> None:
    assert _slice("norm-001") == "normal_request"
    assert _slice("inj-001") == "prompt_injection"
    assert _slice("auth-001") == "unauthorized_request"
