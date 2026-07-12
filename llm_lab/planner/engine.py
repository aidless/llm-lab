import glob
import os
import re
from typing import Any

import yaml

from llm_lab.models import CustomMetrics, Plan, Step

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CUSTOM_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates_custom")


def _load_templates() -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for d in (_TEMPLATES_DIR, _CUSTOM_TEMPLATES_DIR):
        if not os.path.isdir(d):
            continue
        for fpath in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and isinstance(data, dict):
                        data["_source"] = "builtin" if d == _TEMPLATES_DIR else "custom"
                        templates.append(data)
            except (OSError, yaml.YAMLError) as exc:
                import warnings

                warnings.warn(f"Skipping template {fpath}: {exc}", stacklevel=2)
    return templates


_TEMPLATES = _load_templates()


def reload_templates() -> None:
    global _TEMPLATES
    _TEMPLATES = _load_templates()


def _template_store_path() -> str:
    os.makedirs(_CUSTOM_TEMPLATES_DIR, exist_ok=True)
    return _CUSTOM_TEMPLATES_DIR


_TEMPLATE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_template_path(template_id: str) -> str:
    if not _TEMPLATE_ID_RE.match(template_id):
        raise ValueError(f"invalid template_id: {template_id!r}")
    store = _template_store_path()
    fpath = os.path.join(store, f"{template_id}.yaml")
    store_real = os.path.realpath(store)
    fpath_real = os.path.realpath(fpath)
    if os.path.commonpath([store_real, fpath_real]) != store_real:
        raise ValueError(f"invalid template_id: {template_id!r}")
    return fpath


def save_custom_template(template_id: str, data: dict[str, Any]) -> str:
    fpath = _safe_template_path(template_id)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        reload_templates()
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to save template {template_id}: {exc}") from exc
    return fpath


def delete_custom_template(template_id: str) -> bool:
    fpath = _safe_template_path(template_id)
    if os.path.isfile(fpath):
        os.remove(fpath)
        reload_templates()
        return True
    return False


def _match_template(goal: str) -> dict[str, Any] | None:
    goal_lower = goal.lower()
    custom: list[dict[str, Any]] = []
    builtin: list[dict[str, Any]] = []
    for tmpl in _TEMPLATES:
        (custom if tmpl.get("_source") == "custom" else builtin).append(tmpl)
    for tmpl in custom + builtin:
        for kw in tmpl["intent_keywords"]:
            if kw.lower() in goal_lower:
                return tmpl
    return None


def _llm_fallback(goal: str, model: str | None = None) -> Plan:
    try:
        import json

        from llm_lab.worker import call_llm

        prompt = (
            "You are a task planner. Given a user goal, output a JSON array of step objects. "
            "Each step has keys: 'action' (short verb phrase) and 'prompt' (what to ask the LLM). "
            f"Goal: {goal}\n\nReturn ONLY valid JSON."
        )
        result = call_llm(prompt, model=model)
        steps_data = json.loads(result["output"])
        steps = [Step(action=s["action"], prompt=s.get("prompt", "")) for s in steps_data]
    except Exception:
        steps = [Step(action="call_llm", prompt=goal)]
    return Plan(template_id=None, steps=steps)


def plan(goal: str, preferred_model: str | None = None) -> Plan:
    tmpl = _match_template(goal)

    if tmpl is not None:
        model = preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
        raw_steps = tmpl["steps"]
        steps = []
        for s in raw_steps:
            if isinstance(s, dict):
                steps.append(Step(action=s["action"], prompt=s.get("prompt", goal), model=model))
            else:
                steps.append(Step(action=s, prompt=goal, model=model))
        m = tmpl.get("metrics")
        metrics = CustomMetrics(**m) if m else None
        return Plan(template_id=tmpl["template_id"], steps=steps, metrics=metrics)

    return _llm_fallback(goal, preferred_model)


def get_template_def(template_id: str) -> dict[str, Any] | None:
    for tmpl in _TEMPLATES:
        if tmpl["template_id"] == template_id:
            return tmpl
    return None


def list_templates() -> list[dict[str, Any]]:
    result = []
    for tmpl in _TEMPLATES:
        entry = {k: v for k, v in tmpl.items() if not k.startswith("_")}
        entry["_source"] = tmpl.get("_source", "builtin")
        result.append(entry)
    return result
