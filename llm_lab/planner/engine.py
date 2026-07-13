import glob
import os
import re
import uuid
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
    """Resolve a template id to a safe filesystem path.

    Security contract (enforced by THREAT_MODEL.md §S4 "Template path
    traversal"):

    1. ``template_id`` must match ``_TEMPLATE_ID_RE``
       (``^[A-Za-z0-9_-]{1,64}$``). Anything else raises
       ``ValueError`` *before* the path is constructed.
    2. The candidate path is normalised via ``os.path.realpath``.
    3. ``os.path.commonpath`` verifies the resolved path is inside
       the template store. Any traversal (e.g. ``..`` segments or
       absolute paths) raises ``ValueError``.

    The first check is the load-bearing one: it constrains the
    input to a 65-character alphabet that cannot contain path
    separators, parent refs, or null bytes. CodeQL's
    ``py/path-injection`` query flags ``os.path.join`` /
    ``open(fpath)`` / ``os.remove(fpath)`` calls below because
    it does not see the upstream sanitiser across function
    boundaries; the suppressions on those call sites point back
    to this function as the authoritative reason.
    """
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
        # CodeQL: template_id is regex-validated by _safe_template_path against
        # _TEMPLATE_ID_RE (^[A-Za-z0-9_-]{1,64}$) AND the resolved path is
        # verified to stay within the templates store via os.path.commonpath.
        # See THREAT_MODEL.md §S4. The "uncontrolled data" alert is a
        # data-flow false positive that does not see the upstream sanitiser.
        with open(fpath, "w", encoding="utf-8") as f:  # codeql[py/path-injection]
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        reload_templates()
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to save template {template_id}: {exc}") from exc
    return fpath


def delete_custom_template(template_id: str) -> bool:
    fpath = _safe_template_path(template_id)
    # See save_custom_template for the suppression justification.
    if os.path.isfile(fpath):  # codeql[py/path-injection]
        os.remove(fpath)  # codeql[py/path-injection]
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


def _llm_fallback(goal: str, model: str | None = None, intent_id: str | None = None) -> Plan:
    try:
        import json

        from llm_lab import tracer
        from llm_lab.worker import call_llm

        prompt = (
            "You are a task planner. Given a user goal, output a JSON array of step objects. "
            "Each step has keys: 'action' (short verb phrase) and 'prompt' (what to ask the LLM). "
            f"Goal: {goal}\n\nReturn ONLY valid JSON."
        )
        result = call_llm(prompt, model=model)
        trace_id = intent_id or uuid.uuid4().hex[:12]
        tracer.trace_call_sync(
            trace_id,
            0,
            result.get("model", model or ""),
            prompt,
            result["output"],
            result.get("token_usage", {}),
            result.get("cost_usd", 0.0),
            "pass",
        )
        steps_data = json.loads(result["output"])
        steps = [Step(action=s["action"], prompt=s.get("prompt", "")) for s in steps_data]
    except Exception:
        steps = [Step(action="call_llm", prompt=goal)]
    return Plan(template_id=None, steps=steps)


def plan(goal: str, preferred_model: str | None = None, intent_id: str | None = None) -> Plan:
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

    return _llm_fallback(goal, preferred_model, intent_id)


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
