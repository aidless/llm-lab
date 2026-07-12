"""Tests for planner module — uses conftest.py sys.path."""

from unittest.mock import patch

from llm_lab.planner.engine import _load_templates, plan


def test_import_smoke():
    """Verify all public planner exports resolve."""
    from llm_lab.planner import (
        delete_custom_template,
        get_template_def,
        list_templates,
        plan,
        reload_templates,
        save_custom_template,
    )

    assert callable(plan)
    assert callable(get_template_def)
    assert callable(list_templates)
    assert callable(save_custom_template)
    assert callable(delete_custom_template)
    assert callable(reload_templates)


def test_templates_loaded():
    templates = _load_templates()
    assert len(templates) >= 3, f"expected ≥3 templates, got {len(templates)}"


def test_match_eval_model():
    p = plan("evaluate gpt-4o on mmlu")
    assert p.template_id == "eval-model"
    assert len(p.steps) > 0


def test_match_compare_ab():
    p = plan("compare gpt-4o vs claude on coding")
    assert p.template_id == "compare-ab"
    assert len(p.steps) > 0


def test_match_summarize():
    p = plan("summarize this research paper")
    assert p.template_id == "summarize-paper"
    assert len(p.steps) > 0


def test_fallback_does_not_crash():
    with patch("llm_lab.worker.call_llm") as mock_call:
        mock_call.return_value = {"output": '[{"action": "research", "prompt": "do something completely random blah"}]'}
        p = plan("do something completely random blah")
        assert len(p.steps) > 0
        assert p.template_id is None


# ── planner/engine.py uncovered-branch coverage ──────────────────────────


def test_load_templates_skips_bad_yaml():
    import os
    import warnings

    from llm_lab.planner.engine import _CUSTOM_TEMPLATES_DIR, reload_templates

    os.makedirs(_CUSTOM_TEMPLATES_DIR, exist_ok=True)
    bad_fpath = os.path.join(_CUSTOM_TEMPLATES_DIR, "_bad_test_yaml_.yaml")
    try:
        with open(bad_fpath, "w") as f:
            f.write(": bad yaml : [\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reload_templates()
            assert any("Skipping template" in str(x.message) for x in w)
    finally:
        if os.path.isfile(bad_fpath):
            os.remove(bad_fpath)


def test_fallback_json_error_returns_single_step():
    with patch("llm_lab.worker.call_llm") as mock_call:
        mock_call.return_value = {"output": "this is not json"}
        p = plan("do something completely random blah")
        assert p.template_id is None
        assert len(p.steps) == 1
        assert p.steps[0].action == "call_llm"


def test_get_template_def_not_found():
    from llm_lab.planner.engine import get_template_def

    assert get_template_def("nonexistent-template-id") is None


# ── planner/engine.py remaining branch coverage ──────────────────────────


def test_load_templates_skip_missing_custom_dir():
    """engine.py line 17: continue when custom templates dir is missing."""
    import os

    from llm_lab.planner.engine import _CUSTOM_TEMPLATES_DIR, _load_templates

    removed = False
    if os.path.isdir(_CUSTOM_TEMPLATES_DIR):
        os.rename(_CUSTOM_TEMPLATES_DIR, _CUSTOM_TEMPLATES_DIR + "_backup")
        removed = True
    try:
        templates = _load_templates()
        assert len(templates) >= 3
    finally:
        if removed:
            os.rename(_CUSTOM_TEMPLATES_DIR + "_backup", _CUSTOM_TEMPLATES_DIR)


def test_save_custom_template_failure():
    """engine.py lines 52-53: save_custom_template raises on write error."""
    from unittest.mock import patch

    from llm_lab.planner.engine import save_custom_template

    with patch("builtins.open", side_effect=OSError("permission denied")):
        import pytest

        with pytest.raises(RuntimeError, match="Failed to save template"):
            save_custom_template("test_fail", {"template_id": "test_fail", "steps": ["a"]})


def test_get_template_def_found():
    """engine.py line 110: get_template_def returns matching template."""
    from llm_lab.planner.engine import get_template_def

    tmpl = get_template_def("eval-model")
    assert tmpl is not None
    assert tmpl["template_id"] == "eval-model"


# ── planner/engine.py delete_custom_template ──────────────────────────────


def test_delete_custom_template_success():
    from llm_lab.planner.engine import (
        _CUSTOM_TEMPLATES_DIR,
        delete_custom_template,
        reload_templates,
        save_custom_template,
    )

    import os

    os.makedirs(_CUSTOM_TEMPLATES_DIR, exist_ok=True)
    tid = "_test_delete_me_"
    save_custom_template(tid, {"template_id": tid, "steps": ["test"]})
    fpath = os.path.join(_CUSTOM_TEMPLATES_DIR, f"{tid}.yaml")
    assert os.path.isfile(fpath)
    result = delete_custom_template(tid)
    assert result is True
    assert not os.path.isfile(fpath)


def test_delete_custom_template_not_found():
    from llm_lab.planner.engine import delete_custom_template

    result = delete_custom_template("_nonexistent_id_")
    assert result is False


# ── planner/engine.py LLM fallback edge cases ─────────────────────────────


def test_fallback_missing_action_key():
    """LLM returns valid JSON but items lack 'action' key → caught by except."""
    from unittest.mock import patch

    from llm_lab.planner.engine import plan

    with patch("llm_lab.worker.call_llm") as mock_call:
        mock_call.return_value = {"output": '[{"prompt": "do something"}]'}
        p = plan("do something completely random blah")
        assert p.template_id is None
        assert len(p.steps) == 1


def test_fallback_missing_prompt_key():
    """LLM returns valid JSON with 'action' but no 'prompt' key → blank prompt."""
    from unittest.mock import patch

    from llm_lab.planner.engine import plan

    with patch("llm_lab.worker.call_llm") as mock_call:
        mock_call.return_value = {
            "output": '[{"action": "research"}]'
        }
        p = plan("do something completely random blah")
        assert p.template_id is None
        assert len(p.steps) == 1
        assert p.steps[0].action == "research"
        assert p.steps[0].prompt == ""


# ── planner/engine.py _match_template no match → fallback ─────────────────


def test_match_template_no_keywords_falls_back():
    from llm_lab.planner.engine import plan

    p = plan("a totally random unrelated string that nothing matches")
    assert p.template_id is None
    assert len(p.steps) > 0


def test_plan_with_metrics_no_template():
    """When fallback LLM response includes metrics, Plan.metrics is None."""
    from llm_lab.planner.engine import plan

    from unittest.mock import patch

    with patch("llm_lab.worker.call_llm") as mock_call:
        mock_call.return_value = {
            "output": '[{"action": "call_llm", "prompt": "test", "metrics": {"min_output_length": 10}}]'
        }
        p = plan("random fallback metric")
        assert p.metrics is None  # fallback path ignores metrics


def test_plan_from_template_without_metrics():
    """Built-in templates have no metrics key → Plan.metrics is None."""
    from llm_lab.planner.engine import plan

    p = plan("evaluate gpt-4o on mmlu")
    assert p.metrics is None


def test_plan_from_custom_template_with_metrics():
    """Save a custom template with metrics, then verify Plan.metrics is parsed."""
    from llm_lab.planner.engine import (
        _CUSTOM_TEMPLATES_DIR,
        delete_custom_template,
        plan,
        save_custom_template,
    )

    import os

    tid = "_test_metrics_tmpl_"
    try:
        os.makedirs(_CUSTOM_TEMPLATES_DIR, exist_ok=True)
        save_custom_template(tid, {
            "template_id": tid,
                "intent_keywords": ["metrics-test"],
            "steps": [{"action": "call_llm", "prompt": "do something"}],
            "metrics": {
                "min_output_length": 50,
                "must_contain": ["result"],
            },
        })
        p = plan("metrics-test some goal")
        assert p.template_id == tid
        assert p.metrics is not None
        assert p.metrics.min_output_length == 50
        assert p.metrics.must_contain == ["result"]
    finally:
        delete_custom_template(tid)
        fpath = os.path.join(_CUSTOM_TEMPLATES_DIR, f"{tid}.yaml")
        if os.path.isfile(fpath):
            os.remove(fpath)


def test_plan_from_custom_template_with_empty_metrics():
    """Save a custom template with empty metrics → Plan.metics is None or has None values."""
    from llm_lab.planner.engine import (
        _CUSTOM_TEMPLATES_DIR,
        delete_custom_template,
        plan,
        save_custom_template,
    )

    import os

    tid = "_test_empty_metrics_"
    try:
        os.makedirs(_CUSTOM_TEMPLATES_DIR, exist_ok=True)
        save_custom_template(tid, {
            "template_id": tid,
                "intent_keywords": ["empty-metrics"],
            "steps": [{"action": "call_llm", "prompt": "test"}],
            "metrics": {},
        })
        p = plan("empty-metrics test")
        assert p.template_id == tid
        # empty dict may produce None or a metrics object with all-None fields
        if p.metrics is not None:
            assert p.metrics.min_output_length is None
            assert p.metrics.must_contain is None
    finally:
        delete_custom_template(tid)
        fpath = os.path.join(_CUSTOM_TEMPLATES_DIR, f"{tid}.yaml")
        if os.path.isfile(fpath):
            os.remove(fpath)


def test_list_templates_returns_expected_structure():
    """list_templates() returns entries with template_id and _source (engine.py lines 126-131)."""
    from llm_lab.planner.engine import list_templates

    tms = list_templates()
    assert len(tms) >= 3
    for entry in tms:
        assert "template_id" in entry
        assert "_source" in entry
