"""Tests for verifier.py — structural, keyword, schema, and deepeval verifiers."""

from unittest.mock import patch

import pytest

from llm_lab.verifier import (
    DeepEvalVerifier,
    KeywordVerifier,
    SchemaVerifier,
    StructuralVerifier,
    get_verifier,
    register_verifier,
)


class TestStructuralVerifier:
    def test_empty_output_fails(self):
        v = StructuralVerifier()
        verdict = v.verify("")
        assert verdict.label == "fail"

    def test_whitespace_only_fails(self):
        v = StructuralVerifier()
        verdict = v.verify("   ")
        assert verdict.label == "fail"

    def test_non_empty_passes(self):
        v = StructuralVerifier()
        verdict = v.verify("hello")
        assert verdict.label == "pass"

    def test_keywords_all_present(self):
        v = StructuralVerifier()
        verdict = v.verify("foo bar baz", expected_keywords=["foo", "bar"])
        assert verdict.label == "pass"

    def test_keyword_missing(self):
        v = StructuralVerifier()
        verdict = v.verify("foo", expected_keywords=["foo", "bar"])
        assert verdict.label == "fail"

    def test_json_schema_valid(self):
        v = StructuralVerifier()
        verdict = v.verify('{"a": 1}', schema_type="json")
        assert verdict.label == "pass"

    def test_json_schema_invalid(self):
        v = StructuralVerifier()
        verdict = v.verify("not json", schema_type="json")
        assert verdict.label == "fail"


class TestKeywordVerifier:
    @pytest.fixture
    def verifier(self):
        return KeywordVerifier(keywords=["foo", "bar"], require_all=True)

    def test_all_keywords_found(self, verifier):
        verdict = verifier.verify("foo and bar here")
        assert verdict.label == "pass"

    def test_some_keywords_missing(self, verifier):
        verdict = verifier.verify("only foo")
        assert verdict.label == "fail"

    def test_no_keywords_found(self, verifier):
        verdict = verifier.verify("nothing")
        assert verdict.label == "fail"

    def test_not_require_all_any_found(self):
        v = KeywordVerifier(keywords=["foo", "bar"], require_all=False)
        verdict = v.verify("just foo")
        assert verdict.label == "pass"

    def test_not_require_all_none_found(self):
        v = KeywordVerifier(keywords=["foo", "bar"], require_all=False)
        verdict = v.verify("nothing")
        assert verdict.label == "fail"

    def test_empty_keywords_list(self):
        v = KeywordVerifier(keywords=[])
        verdict = v.verify("anything")
        assert verdict.label == "pass"

    def test_empty_output(self):
        v = KeywordVerifier(keywords=["something"])
        verdict = v.verify("")
        assert verdict.label == "fail"

    def test_reason_on_fail(self):
        v = KeywordVerifier(keywords=["x", "y"])
        verdict = v.verify("z")
        assert "missing" in verdict.reason.lower()

    def test_reason_on_pass(self):
        v = KeywordVerifier(keywords=["x"])
        verdict = v.verify("x")
        assert "found" in verdict.reason.lower()


class TestSchemaVerifier:
    def test_all_fields_present(self):
        v = SchemaVerifier(required_fields=["name", "age"])
        verdict = v.verify('{"name": "Alice", "age": 30}')
        assert verdict.label == "pass"

    def test_missing_field(self):
        v = SchemaVerifier(required_fields=["name", "age"])
        verdict = v.verify('{"name": "Alice"}')
        assert verdict.label == "fail"

    def test_invalid_json(self):
        v = SchemaVerifier(required_fields=["name"])
        verdict = v.verify("not json")
        assert verdict.label == "fail"

    def test_empty_required_fields(self):
        v = SchemaVerifier(required_fields=[])
        verdict = v.verify("{}")
        assert verdict.label == "pass"

    def test_extra_fields_ignored(self):
        v = SchemaVerifier(required_fields=["a"])
        verdict = v.verify('{"a": 1, "b": 2}')
        assert verdict.label == "pass"

    def test_nested_field_not_checked(self):
        v = SchemaVerifier(required_fields=["meta"])
        verdict = v.verify('{"meta": {"nested": 1}}')
        assert verdict.label == "pass"

    def test_reason_on_fail(self):
        v = SchemaVerifier(required_fields=["missing"])
        verdict = v.verify("{}")
        assert "missing" in verdict.reason.lower()


class TestDeepEvalVerifier:
    def test_disabled_returns_heuristic_pass(self):
        v = DeepEvalVerifier(metric="faithfulness")
        verdict = v.verify("some output", input_text="some input")
        assert verdict.label == "pass"

    def test_disabled_fails_on_empty_output(self):
        v = DeepEvalVerifier(metric="faithfulness")
        verdict = v.verify("", input_text="input")
        assert verdict.label == "fail"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="unknown DeepEval metric"):
            DeepEvalVerifier(metric="nonexistent")

    def test_available_metrics(self):
        for metric in ("faithfulness", "answer_relevancy", "bias", "toxicity"):
            v = DeepEvalVerifier(metric=metric)
            assert v._metric == metric

    def test_enabled_but_no_deepeval_heuristic(self, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_ENABLED", "true")
        v = DeepEvalVerifier()
        verdict = v.verify("output")
        assert verdict.label in ("pass", "partial")

    def test_run_metric_faithfulness_passes(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setenv("DEEPEVAL_THRESHOLD", "0.5")

        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "faithful"
        mock_cls = MagicMock(return_value=mock_metric)

        v = DeepEvalVerifier(metric="faithfulness")
        verdict = v._run_metric("output", "context", mock_cls, MagicMock(), MagicMock(), MagicMock())
        assert verdict.label == "pass"
        mock_metric.measure.assert_called_once_with(output="output", context=["context"])

    def test_run_metric_answer_relevancy_fails(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setenv("DEEPEVAL_THRESHOLD", "0.5")

        mock_metric = MagicMock()
        mock_metric.score = 0.3
        mock_metric.reason = "not relevant"
        mock_cls = MagicMock(return_value=mock_metric)

        v = DeepEvalVerifier(metric="answer_relevancy")
        verdict = v._run_metric("output", "question", MagicMock(), mock_cls, MagicMock(), MagicMock())
        assert verdict.label == "fail"
        mock_metric.measure.assert_called_once_with(output="output", input="question")

    def test_run_metric_bias_measures_output_only(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setenv("DEEPEVAL_THRESHOLD", "0.5")

        mock_metric = MagicMock()
        mock_metric.score = 0.95
        mock_metric.reason = "no bias"
        mock_cls = MagicMock(return_value=mock_metric)

        v = DeepEvalVerifier(metric="bias")
        verdict = v._run_metric("output", None, MagicMock(), MagicMock(), mock_cls, MagicMock())
        assert verdict.label == "pass"
        mock_metric.measure.assert_called_once_with(output="output")

    def test_run_metric_toxicity_fails_below_threshold(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setenv("DEEPEVAL_THRESHOLD", "0.7")

        mock_metric = MagicMock()
        mock_metric.score = 0.5
        mock_metric.reason = "toxic content"
        mock_cls = MagicMock(return_value=mock_metric)

        v = DeepEvalVerifier(metric="toxicity")
        verdict = v._run_metric("bad output", None, MagicMock(), MagicMock(), MagicMock(), mock_cls)
        assert verdict.label == "fail"

    def test_run_metric_handles_deepeval_error(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setenv("DEEPEVAL_ENABLED", "true")

        with patch("deepeval.metrics.FaithfulnessMetric", side_effect=Exception("deepeval API error")):
            v = DeepEvalVerifier(metric="faithfulness")
            verdict = v.verify("output", input_text="ctx")
        assert verdict.label == "partial"
        assert "deepeval error" in verdict.reason


class TestRegistry:
    def test_get_verifier_structural_by_default(self):
        v = get_verifier()
        assert isinstance(v, StructuralVerifier)

    def test_get_verifier_by_name(self):
        v = get_verifier("deepeval")
        assert isinstance(v, DeepEvalVerifier)

    def test_get_verifier_raises_on_unknown(self):
        with pytest.raises(KeyError, match="unknown verifier"):
            get_verifier("nonexistent")

    def test_register_and_retrieve(self):
        dummy = StructuralVerifier()
        register_verifier("custom", dummy)
        assert get_verifier("custom") is dummy


def test_deepeval_import_error_causes_fallback(monkeypatch):
    """verifier.py line 84: ImportError inside verify() falls back to heuristic."""
    import builtins
    import sys

    monkeypatch.setenv("DEEPEVAL_ENABLED", "1")
    # Purge cached deepeval modules so import must call __import__
    for mod in list(sys.modules):
        if mod.startswith("deepeval"):
            sys.modules.pop(mod, None)
    real_import = builtins.__import__

    def _block_deepeval(name, *args, **kwargs):
        if name == "deepeval" or name.startswith("deepeval."):
            raise ImportError("No module named deepeval")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_block_deepeval):
        v = DeepEvalVerifier("faithfulness")
        verdict = v.verify("some output", "some input")
        assert verdict.label == "pass"


class TestCheckCustomMetrics:
    def make_metrics(self, **kwargs):
        from types import SimpleNamespace

        defaults = dict(
            min_output_length=None, max_output_length=None,
            min_tokens=None, max_tokens=None,
            must_contain=None, must_not_contain=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_none_metrics_returns_none(self):
        from llm_lab.verifier import check_custom_metrics
        assert check_custom_metrics("hello", None) is None

    def test_min_output_length_pass(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(min_output_length=5)
        v = check_custom_metrics("hello world", m)
        assert v.label == "pass"

    def test_min_output_length_fail(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(min_output_length=20)
        v = check_custom_metrics("too short", m)
        assert v.label == "fail"

    def test_max_output_length_pass(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(max_output_length=100)
        v = check_custom_metrics("short enough", m)
        assert v.label == "pass"

    def test_max_output_length_fail(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(max_output_length=5)
        v = check_custom_metrics("this is way too long", m)
        assert v.label == "fail"

    def test_min_tokens_pass(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(min_tokens=3)
        v = check_custom_metrics("one two three", m)
        assert v.label == "pass"

    def test_min_tokens_fail(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(min_tokens=10)
        v = check_custom_metrics("too few words", m)
        assert v.label == "fail"

    def test_max_tokens_pass(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(max_tokens=10)
        v = check_custom_metrics("short text", m)
        assert v.label == "pass"

    def test_max_tokens_fail(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(max_tokens=2)
        v = check_custom_metrics("one two three four", m)
        assert v.label == "fail"

    def test_must_contain_pass(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(must_contain=["foo", "bar"])
        v = check_custom_metrics("foo and bar here", m)
        assert v.label == "pass"

    def test_must_contain_fail(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(must_contain=["foo", "bar"])
        v = check_custom_metrics("only foo", m)
        assert v.label == "fail"

    def test_must_not_contain_pass(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(must_not_contain=["bad", "evil"])
        v = check_custom_metrics("good output", m)
        assert v.label == "pass"

    def test_must_not_contain_fail(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(must_not_contain=["evil"])
        v = check_custom_metrics("this is evil", m)
        assert v.label == "fail"

    def test_multiple_failures_reported(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(
            min_output_length=100, must_contain=["required_keyword"]
        )
        v = check_custom_metrics("short", m)
        assert v.label == "fail"
        assert "output length" in v.reason
        assert "missing required content" in v.reason

    def test_reason_on_pass_returns_ok(self):
        from llm_lab.verifier import check_custom_metrics
        m = self.make_metrics(min_output_length=1)
        v = check_custom_metrics("ok", m)
        assert v.label == "pass"
        assert v.reason == "custom metrics passed"

    def test_uses_tokenizer_not_word_count(self, monkeypatch):
        from llm_lab import verifier as verifier_mod
        from unittest.mock import patch

        # Force a tokenizer count that diverges from the word count to prove
        # the metric counts tokens, not whitespace-separated words.
        with patch.object(verifier_mod, "count_tokens", return_value=50):
            assert verifier_mod.check_custom_metrics("one two three", self.make_metrics(min_tokens=10)).label == "pass"
            assert verifier_mod.check_custom_metrics("one two three", self.make_metrics(max_tokens=5)).label == "fail"
