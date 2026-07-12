"""Core runner — sync logic shared between API and CLI."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

from llm_lab import settings as app_settings  # noqa: E402
from llm_lab import tracer  # noqa: E402
from llm_lab import verifier as vrf  # noqa: E402
from llm_lab import worker as wrk  # noqa: E402
from llm_lab.planner import plan as build_plan  # noqa: E402


def _resolve_verifier(name: str) -> str:
    """Resolve the verifier that will actually run.

    If the requested verifier is ``deepeval`` but DeepEval is disabled or the
    eval model is unavailable (common in CI / offline runs), fall back to the
    ``structural`` verifier so verification still runs instead of silently
    passing. The requested name is preserved in the result dict.
    """
    if name == "deepeval" and not app_settings.get_settings().deepeval_enabled:
        return "structural"
    return name


def run_plan(goal: str, model: str | None = None, verifier_name: str = "deepeval") -> dict[str, Any]:
    model = model or os.getenv("LLM_MODEL", "gpt-4o")
    intent_id = uuid.uuid4().hex[:12]

    p = build_plan(goal, model)

    all_passed = True
    total_tokens = 0
    total_cost = 0.0
    steps_detail: list[dict[str, Any]] = []

    actual_verifier = _resolve_verifier(verifier_name)

    for step in p.steps:
        result = wrk.call_llm(step.prompt, model=model)
        step.output = result["output"]
        step.model = result["model"]

        tu = result["token_usage"]
        total_tokens += tu.get("total_tokens", 0)
        total_cost += result["cost_usd"]

        verifier = vrf.get_verifier(actual_verifier)
        v = verifier.verify(result["output"], input_text=step.prompt)

        if v.label == "fail":
            all_passed = False

        mv = vrf.check_custom_metrics(result["output"], p.metrics)
        if mv and mv.label == "fail":
            all_passed = False

        seq = len(steps_detail) + 1
        tracer.trace_call_sync(
            intent_id,
            seq,
            step.model,
            step.prompt,
            step.output,
            tu,
            result["cost_usd"],
            v.label,
        )

        steps_detail.append(
            {
                "action": step.action,
                "prompt": step.prompt,
                "output": step.output,
                "model": step.model,
                "verdict": {"label": v.label, "reason": v.reason},
                "tokens": tu.get("total_tokens", 0),
                "cost": result["cost_usd"],
                "metric_check": mv.model_dump() if mv else None,
            }
        )

    return {
        "run_id": intent_id,
        "intent_id": intent_id,
        "goal": goal,
        "model": model,
        "verifier": verifier_name,
        "steps": len(p.steps),
        "plan_template": p.template_id,
        "steps_detail": steps_detail,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "all_passed": all_passed,
    }


def batch(goal: str, models: list[str], verifier_name: str = "deepeval") -> dict[str, Any]:
    results = []
    for model in models:
        r = run_plan(goal, model, verifier_name)
        results.append(
            {
                "model": r["model"],
                "plan_template": r["plan_template"],
                "total_tokens": r["total_tokens"],
                "total_cost_usd": r["total_cost_usd"],
                "all_passed": r["all_passed"],
                "steps": r["steps_detail"],
            }
        )
    results.sort(key=lambda x: (-x["all_passed"], x["total_cost_usd"], x["total_tokens"]))
    return {"goal": goal, "models": results, "count": len(results)}


def batch_parallel(
    goal: str, models: list[str], verifier_name: str = "deepeval", max_workers: int = 4
) -> dict[str, Any]:
    results = []

    def _run(m: str) -> dict:
        return run_plan(goal, m, verifier_name)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {pool.submit(_run, m): m for m in models}
        for fut in as_completed(fut_map):
            try:
                r = fut.result()
                results.append(
                    {
                        "model": r["model"],
                        "plan_template": r["plan_template"],
                        "total_tokens": r["total_tokens"],
                        "total_cost_usd": r["total_cost_usd"],
                        "all_passed": r["all_passed"],
                        "steps": r["steps_detail"],
                    }
                )
            except Exception as exc:
                m = fut_map[fut]
                results.append(
                    {
                        "model": m,
                        "plan_template": None,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                        "all_passed": False,
                        "error": str(exc),
                    }
                )

    results.sort(key=lambda x: (-x["all_passed"], x["total_cost_usd"], x["total_tokens"]))
    return {"goal": goal, "models": results, "count": len(results), "parallel": True}


def compare(goal: str, model_a: str | None = None, model_b: str | None = None) -> dict[str, Any]:
    model_a = model_a or os.getenv("LLM_MODEL", "gpt-4o")
    model_b = model_b or os.getenv("LLM_MODEL_2", "gpt-4o-mini")

    result_a = run_plan(goal, model_a)
    result_b = run_plan(goal, model_b)

    return {
        "goal": goal,
        "model_a": {
            "model": result_a["model"],
            "plan_template": result_a["plan_template"],
            "total_tokens": result_a["total_tokens"],
            "total_cost_usd": result_a["total_cost_usd"],
            "all_passed": result_a["all_passed"],
            "steps": result_a["steps_detail"],
        },
        "model_b": {
            "model": result_b["model"],
            "plan_template": result_b["plan_template"],
            "total_tokens": result_b["total_tokens"],
            "total_cost_usd": result_b["total_cost_usd"],
            "all_passed": result_b["all_passed"],
            "steps": result_b["steps_detail"],
        },
        "summary": {
            "winner": "a"
            if result_a["all_passed"] and not result_b["all_passed"]
            else "b"
            if result_b["all_passed"] and not result_a["all_passed"]
            else "tie",
            "cost_delta": round(result_a["total_cost_usd"] - result_b["total_cost_usd"], 6),
            "token_delta": result_a["total_tokens"] - result_b["total_tokens"],
        },
    }
