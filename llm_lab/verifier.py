import json
import logging
import os
from typing import Any

from llm_lab.models import Verdict

log = logging.getLogger(__name__)

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken optional at import time
    _ENC = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken when available, else fall back to a rough heuristic."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


class StructuralVerifier:
    def verify(
        self, output: str, expected_keywords: list[str] | None = None, schema_type: str | None = None, **kwargs: Any
    ) -> Verdict:
        if not output or not output.strip():
            return Verdict(label="fail", reason="output is empty")
        if expected_keywords:
            missing = [kw for kw in expected_keywords if kw not in output]
            if missing:
                return Verdict(label="fail", reason=f"missing expected keywords: {missing}")
            return Verdict(label="pass", reason=f"all {len(expected_keywords)} expected keywords found")
        if schema_type == "json":
            try:
                json.loads(output)
                return Verdict(label="pass", reason="valid JSON")
            except json.JSONDecodeError as e:
                return Verdict(label="fail", reason=f"invalid JSON: {e}")
        return Verdict(label="pass", reason="non-empty output")


class KeywordVerifier:
    def __init__(self, keywords: list[str] | None = None, require_all: bool = True):
        self._keywords = list(keywords or [])
        self._require_all = require_all

    def verify(
        self,
        output: str,
        expected_keywords: list[str] | None = None,
        **kwargs: Any,
    ) -> Verdict:
        keywords = self._keywords or list(expected_keywords or [])
        if not keywords:
            return Verdict(label="pass", reason="no keywords to check")
        found = [kw for kw in keywords if kw in output]
        if self._require_all:
            missing = [kw for kw in keywords if kw not in output]
            if missing:
                return Verdict(label="fail", reason=f"missing keywords: {missing}")
            return Verdict(label="pass", reason="all keywords found")
        if found:
            return Verdict(label="pass", reason=f"found keywords: {found}")
        return Verdict(label="fail", reason="no expected keywords found")


class SchemaVerifier:
    def __init__(self, required_fields: list[str] | None = None):
        self._fields = list(required_fields or [])

    def verify(
        self,
        output: str,
        required_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Verdict:
        fields = self._fields or list(required_fields or [])
        if not fields:
            return Verdict(label="pass", reason="no required fields to check")
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return Verdict(label="fail", reason="output is not valid JSON")
        missing = [f for f in fields if f not in data]
        if missing:
            return Verdict(label="fail", reason=f"missing required fields: {missing}")
        return Verdict(label="pass", reason=f"all {len(fields)} required fields present")


class DeepEvalVerifier:
    METRICS = {
        "faithfulness": "output factually follows from input context",
        "answer_relevancy": "output is relevant to the input question",
        "bias": "output is free from unwanted bias",
        "toxicity": "output is non-toxic",
    }

    def __init__(self, metric: str = "faithfulness"):
        if metric not in self.METRICS:
            raise ValueError(f"unknown DeepEval metric: {metric}. choose from {list(self.METRICS)}")
        self._metric = metric

    def verify(self, output: str, input_text: str | None = None) -> Verdict:
        enabled = os.getenv("DEEPEVAL_ENABLED", "").lower() in ("1", "true", "yes")
        if not enabled:
            return self._heuristic_fallback(output, input_text)
        try:
            from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, FaithfulnessMetric, ToxicityMetric

            return self._run_metric(
                output, input_text, FaithfulnessMetric, AnswerRelevancyMetric, BiasMetric, ToxicityMetric
            )
        except ImportError:
            return self._heuristic_fallback(output, input_text)
        except Exception as e:
            log.warning("deepeval metric '%s' failed: %s", self._metric, e)
            return Verdict(label="partial", reason=f"[deepeval error: {e}]")

    def _heuristic_fallback(self, output: str, _input_text: str | None) -> Verdict:
        if not output or not output.strip():
            return Verdict(label="fail", reason="output is empty")
        return Verdict(label="pass", reason="deepeval unavailable or disabled; heuristic pass")

    def _run_metric(self, output: str, input_text: str | None, *metric_classes) -> Verdict:
        threshold = float(os.getenv("DEEPEVAL_THRESHOLD", "0.5"))
        eval_model = os.getenv("DEEPEVAL_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"

        m = {
            "faithfulness": metric_classes[0](threshold=threshold, model=eval_model),
            "answer_relevancy": metric_classes[1](threshold=threshold, model=eval_model),
            "bias": metric_classes[2](threshold=threshold, model=eval_model),
            "toxicity": metric_classes[3](threshold=threshold, model=eval_model),
        }[self._metric]

        if self._metric == "faithfulness":
            m.measure(output=output, context=[input_text or ""])
        elif self._metric == "answer_relevancy":
            m.measure(output=output, input=input_text or "")
        else:
            m.measure(output=output)

        return Verdict(
            label="pass" if m.score >= threshold else "fail",
            reason=m.reason if hasattr(m, "reason") else f"score={m.score:.3f}",
        )


_VERIFIERS = {
    "structural": StructuralVerifier(),
    "keyword": KeywordVerifier(),
    "schema": SchemaVerifier(),
    "deepeval": DeepEvalVerifier(),
}


def get_verifier(name: str = "structural") -> Any:
    if name not in _VERIFIERS:
        raise KeyError(f"unknown verifier: {name!r}. available: {list(_VERIFIERS)}")
    return _VERIFIERS[name]


def register_verifier(name: str, verifier: Any) -> None:
    _VERIFIERS[name] = verifier


def check_custom_metrics(output: str, metrics: Any) -> Verdict | None:
    if metrics is None:
        return None
    failures: list[str] = []
    if metrics.min_output_length is not None and len(output) < metrics.min_output_length:
        failures.append(
            f"output length {len(output)} < min {metrics.min_output_length}"
        )
    if metrics.max_output_length is not None and len(output) > metrics.max_output_length:
        failures.append(
            f"output length {len(output)} > max {metrics.max_output_length}"
        )
    if metrics.min_tokens is not None:
        t = count_tokens(output)
        if t < metrics.min_tokens:
            failures.append(f"token count {t} < min tokens {metrics.min_tokens}")
    if metrics.max_tokens is not None:
        t = count_tokens(output)
        if t > metrics.max_tokens:
            failures.append(f"token count {t} > max tokens {metrics.max_tokens}")
    if metrics.must_contain:
        missing = [kw for kw in metrics.must_contain if kw not in output]
        if missing:
            failures.append(f"missing required content: {missing}")
    if metrics.must_not_contain:
        found = [kw for kw in metrics.must_not_contain if kw in output]
        if found:
            failures.append(f"contains forbidden content: {found}")
    if failures:
        return Verdict(label="fail", reason="; ".join(failures))
    return Verdict(label="pass", reason="custom metrics passed")
