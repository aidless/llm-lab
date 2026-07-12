"""Tracer: Langfuse-backed with local SQLite fallback."""

import os
from typing import Any

from llm_lab import db as _local_db
from llm_lab.models import Verdict

_lf = None
_trace_cache: dict[str, Any] = {}


def _get_lf() -> Any | None:
    global _lf
    if _lf is not None:
        return _lf
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not secret or not public:
        return None
    try:
        from langfuse import Langfuse

        _lf = Langfuse(
            secret_key=secret,
            public_key=public,
            host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
        )
        return _lf
    except ImportError:
        return None


def _ensure_trace(intent_id: str, goal: str = "") -> None:
    lf = _get_lf()
    if lf and intent_id not in _trace_cache:
        _trace_cache[intent_id] = lf.trace(name=intent_id, input=goal or intent_id)


async def trace_call(
    intent_id: str,
    seq: int,
    model: str,
    prompt: str,
    output: str,
    token_usage: dict | None = None,
    cost_usd: float | None = None,
    verdict: str | None = None,
) -> None:
    lf = _get_lf()
    if lf:
        trace = _trace_cache.get(intent_id)
        if trace:
            trace.generation(
                name=f"step-{seq}",
                model=model or "unknown",
                input=prompt,
                output=output,
                usage={
                    "input": (token_usage or {}).get("prompt_tokens", 0),
                    "output": (token_usage or {}).get("completion_tokens", 0),
                }
                if token_usage
                else None,
                metadata={"cost_usd": cost_usd, "seq": seq},
            )
    else:
        await _local_db.append_event(
            intent_id=intent_id,
            seq=seq,
            action="call",
            model=model,
            detail={"prompt_preview": prompt[:200]},
            input_text=prompt,
            output_text=output,
            token_usage=token_usage,
            cost_usd=cost_usd,
            verdict=verdict,
        )


async def trace_event(intent_id: str, seq: int, action: str, detail: str | None = None) -> None:
    lf = _get_lf()
    if lf:
        _ensure_trace(intent_id, detail or action)
        trace = _trace_cache.get(intent_id)
        if trace:
            trace.span(name=f"{action}-{seq}", input={"action": action, "detail": detail})
    else:
        await _local_db.append_event(
            intent_id=intent_id,
            seq=seq,
            action=action,
            detail=detail,
        )


async def trace_verdict(intent_id: str, seq: int, verdict: Verdict) -> None:
    lf = _get_lf()
    if lf:
        trace = _trace_cache.get(intent_id)
        if trace:
            trace.score(
                name="verdict",
                value=1.0 if verdict.label == "pass" else (0.0 if verdict.label == "fail" else 0.5),
                comment=verdict.reason,
            )
    else:
        await _local_db.append_event(
            intent_id=intent_id,
            seq=seq,
            action="verify",
            detail=verdict.model_dump_json(),
            verdict=verdict.label,
        )


async def get_trace(intent_id: str) -> list[dict[str, Any]] | dict[str, str]:
    lf = _get_lf()
    if lf:
        return {"intent_id": intent_id, "source": "langfuse", "note": "view full trace in Langfuse dashboard"}
    rows = await _local_db.get_events(intent_id)
    return [dict(r) for r in rows]


async def get_summary(intent_id: str) -> dict[str, Any]:
    return await _local_db.get_run_summary(intent_id)


def shutdown() -> None:
    lf = _get_lf()
    if lf:
        lf.flush()
