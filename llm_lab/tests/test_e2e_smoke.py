"""E2E smoke test — exercises the full pipeline with mocked LLM + verifier."""

import json

_NON_MATCHING_GOAL = "custom analysis of dataset XYZ123"


def test_run_plan_full_pipeline(mock_llm, mock_verifier):
    from llm_lab.runner import run_plan

    r = run_plan(_NON_MATCHING_GOAL, "gpt-4o")
    assert r["intent_id"] is not None
    assert r["goal"] == _NON_MATCHING_GOAL
    assert r["model"] == "gpt-4o"
    assert r["verifier"] == "deepeval"
    assert r["steps"] >= 1
    assert r["all_passed"] is True
    assert len(r["steps_detail"]) == r["steps"]
    for s in r["steps_detail"]:
        assert s["action"] is not None
        assert s["output"] == "mock output"
        assert s["model"] == "gpt-4o"
        assert s["verdict"]["label"] == "pass"
        assert s["tokens"] == 10
        assert s["cost"] == 0.0001


def test_run_plan_with_export(mock_llm, mock_verifier):
    from llm_lab.export import export_csv, export_json
    from llm_lab.runner import run_plan

    r = run_plan(_NON_MATCHING_GOAL, "gpt-4o")
    steps = r["steps_detail"]

    events = [
        {
            "id": i + 1,
            "intent_id": r["intent_id"],
            "seq": i + 1,
            "timestamp": "2025-01-01T00:00:00",
            "action": s["action"],
            "model": s["model"],
            "detail": s["output"],
            "cost_usd": s["cost"],
        }
        for i, s in enumerate(steps)
    ]

    json_blob = export_json(r["intent_id"], events)
    parsed = json.loads(json_blob)
    assert parsed["intent_id"] == r["intent_id"]
    assert len(parsed["events"]) == len(events)

    csv_blob = export_csv(events)
    assert "intent_id" in csv_blob
    assert r["intent_id"] in csv_blob


def test_batch_pipeline(mock_llm, mock_verifier):
    from llm_lab.runner import batch

    r = batch(_NON_MATCHING_GOAL, ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"])
    assert r["goal"] == _NON_MATCHING_GOAL
    assert r["count"] == 3
    assert len(r["models"]) == 3
    for m in r["models"]:
        assert "model" in m
        assert "all_passed" in m
        assert "total_tokens" in m
        assert "total_cost_usd" in m


def test_compare_pipeline(mock_llm, mock_verifier):
    from llm_lab.runner import compare

    c = compare(_NON_MATCHING_GOAL, "gpt-4o", "gpt-4o-mini")
    assert c["goal"] == _NON_MATCHING_GOAL
    assert c["model_a"]["model"] == "gpt-4o"
    assert c["model_b"]["model"] == "gpt-4o-mini"
    assert c["summary"]["winner"] in ("a", "b", "tie")
    assert "cost_delta" in c["summary"]
    assert "token_delta" in c["summary"]


def test_runner_integrates_with_planner_fallback(mock_llm, mock_verifier):
    from llm_lab.planner import plan as build_plan
    from llm_lab.runner import run_plan

    p = build_plan(_NON_MATCHING_GOAL, "gpt-4o")
    assert p.template_id is None
    assert len(p.steps) >= 1

    r = run_plan(_NON_MATCHING_GOAL, "gpt-4o")
    assert r["plan_template"] is None
    assert r["steps"] == len(p.steps)


def test_runner_matches_template_goal(mock_llm, mock_verifier):
    from llm_lab.planner import plan as build_plan
    from llm_lab.runner import run_plan

    p = build_plan("evaluate model X on benchmark Y", "gpt-4o")
    assert p.template_id == "eval-model"
    assert len(p.steps) >= 1

    r = run_plan("evaluate model X on benchmark Y", "gpt-4o")
    assert r["plan_template"] == "eval-model"
