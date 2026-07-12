"""Tests for runner.py — uses conftest.py fixtures."""

from llm_lab.tests.helpers import make_verdict


def test_run_plan_returns_expected_keys(mock_llm, mock_verifier):
    from llm_lab.planner import plan as build_plan
    from llm_lab.runner import run_plan

    p = build_plan("test goal", "gpt-4o")
    num_steps = len(p.steps)

    r = run_plan("test goal", "gpt-4o")
    assert r["intent_id"] is not None
    assert r["goal"] == "test goal"
    assert r["model"] == "gpt-4o"
    assert r["steps"] == num_steps
    assert r["total_tokens"] == 10 * num_steps
    assert r["total_cost_usd"] == 0.0001 * num_steps
    assert r["all_passed"] is True
    assert len(r["steps_detail"]) == r["steps"]


def test_run_plan_fail_verdict(mock_llm, mock_verifier):
    from llm_lab.runner import run_plan

    mock_verifier.return_value.verify.return_value = make_verdict("fail", "bad output")

    r = run_plan("test goal", "gpt-4o")
    assert r["all_passed"] is False


def test_compare_returns_both_models(mock_llm, mock_verifier):
    from llm_lab.runner import compare

    c = compare("test goal", "gpt-4o", "gpt-4o-mini")
    assert c["model_a"]["model"] == "gpt-4o"
    assert c["model_b"]["model"] == "gpt-4o-mini"
    assert "summary" in c
    assert c["summary"]["winner"] in ("a", "b", "tie")


def test_batch_returns_sorted_models(mock_llm, mock_verifier):
    from llm_lab.runner import batch

    r = batch("test goal", ["gpt-4o", "gpt-4o-mini", "claude-3"])
    assert r["count"] == 3
    assert len(r["models"]) == 3


def test_batch_single_model(mock_llm, mock_verifier):
    from llm_lab.runner import batch

    r = batch("test goal", ["gpt-4o"])
    assert r["count"] == 1


# Use a goal that doesn't match template keywords so fallback produces 1 step.
_NON_MATCHING_GOAL = "custom analysis of dataset XYZ123"


def test_compare_b_wins(mock_llm, mock_verifier):
    from llm_lab.runner import compare
    from llm_lab.tests.helpers import make_verdict

    mock_verifier.return_value.verify.side_effect = [
        make_verdict("fail", "bad"),
        make_verdict("pass", "ok"),
    ]
    c = compare(_NON_MATCHING_GOAL, "gpt-4o", "gpt-4o-mini")
    assert c["summary"]["winner"] == "b"


def test_compare_both_fail_tie(mock_llm, mock_verifier):
    from llm_lab.runner import compare
    from llm_lab.tests.helpers import make_verdict

    mock_verifier.return_value.verify.side_effect = [
        make_verdict("fail", "bad"),
        make_verdict("fail", "bad"),
    ]
    c = compare(_NON_MATCHING_GOAL, "gpt-4o", "gpt-4o-mini")
    assert c["summary"]["winner"] == "tie"


def test_compare_both_pass_tie(mock_llm, mock_verifier):
    from llm_lab.runner import compare

    c = compare(_NON_MATCHING_GOAL, "gpt-4o", "gpt-4o-mini")
    assert c["summary"]["winner"] == "tie"


def test_batch_empty_models(mock_llm, mock_verifier):
    from llm_lab.runner import batch

    r = batch(_NON_MATCHING_GOAL, [])
    assert r["count"] == 0
    assert r["models"] == []


def test_run_plan_with_structural_verifier(mock_llm):
    from llm_lab.runner import run_plan

    r = run_plan(_NON_MATCHING_GOAL, "gpt-4o", verifier_name="structural")
    assert r["verifier"] == "structural"
    assert r["all_passed"] is True


def test_batch_parallel_returns_same_shape(mock_llm, mock_verifier):
    from llm_lab.runner import batch_parallel

    result = batch_parallel("test parallel", models=["gpt-4o"])
    assert "goal" in result
    assert "models" in result
    assert result["count"] == 1
    assert result.get("parallel") is True
    assert "all_passed" in result["models"][0]


def test_batch_parallel_multiple_models(mock_llm, mock_verifier):
    from llm_lab.runner import batch_parallel

    result = batch_parallel("test multi", models=["gpt-4o", "gpt-4o-mini"])
    assert len(result["models"]) == 2


def test_batch_parallel_empty_models(mock_llm, mock_verifier):
    from llm_lab.runner import batch_parallel

    result = batch_parallel("empty", models=[])
    assert result["count"] == 0
    assert result["models"] == []
    assert result["parallel"] is True


def test_batch_parallel_respects_max_workers(mock_llm, mock_verifier):
    from llm_lab.runner import batch_parallel

    result = batch_parallel("workers", models=["gpt-4o", "gpt-4o-mini"], max_workers=2)
    assert len(result["models"]) == 2


def test_batch_parallel_sorts_by_all_passed(mock_llm, mock_verifier):
    from llm_lab.runner import batch_parallel
    from llm_lab.tests.helpers import make_verdict

    mock_verifier.return_value.verify.side_effect = [
        make_verdict("fail", "bad"),
        make_verdict("pass", "ok"),
    ]
    result = batch_parallel("sort check", models=["gpt-4o", "gpt-4o-mini"])
    assert len(result["models"]) == 2
    assert result["models"][0]["all_passed"] is True
    assert result["models"][1]["all_passed"] is False


def test_run_plan_custom_metrics_fail(mock_llm, mock_verifier):
    from unittest.mock import patch

    from llm_lab.models import Verdict
    from llm_lab.runner import run_plan

    with patch("llm_lab.runner.vrf.check_custom_metrics", return_value=Verdict(label="fail", reason="too short")):
        r = run_plan("test goal", "gpt-4o")
    assert r["all_passed"] is False
    assert r["steps_detail"][-1]["metric_check"]["label"] == "fail"


def test_batch_parallel_worker_exception(mock_llm, mock_verifier):
    from llm_lab.runner import batch_parallel

    mock_llm.side_effect = RuntimeError("llm call failed")
    result = batch_parallel("fail goal", models=["gpt-4o", "gpt-4o-mini"], max_workers=1)
    assert result["count"] == 2
    for m in result["models"]:
        assert m["all_passed"] is False
        assert m["error"] == "llm call failed"
        assert m["total_tokens"] == 0
