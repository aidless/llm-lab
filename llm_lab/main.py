import asyncio
import json
import os
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from llm_lab import (  # noqa: E402
    db,
    export,
    tracer,
)
from llm_lab import runner as core  # noqa: E402
from llm_lab.models import BatchRequest, BatchResult, CompareResult, IntentRequest, RunResult, TemplateDef  # noqa: E402
from llm_lab.planner import delete_custom_template, list_templates, save_custom_template  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────

_HTML_DIR = Path(__file__).parent / "templates"


def _load_html(name: str) -> str:
    return (_HTML_DIR / name).read_text(encoding="utf-8")


_UI_HTML = _load_html("ui.html")
_COMPARE_REPORT_HTML = _load_html("compare_report.html")


async def _try(label: str, coro):
    try:
        return await coro
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to {label}: {exc}") from exc


async def _store_task(
    task_id: str, status: str, result: Any | None = None, error: str | None = None
) -> None:
    await db.save_task(task_id, {"status": status, "result": result, "error": error})


async def _run_task(
    task_id: str, goal: str, model: str | None, mode: str = "submit", model_b: str | None = None
) -> None:
    await _store_task(task_id, "running")
    try:
        if mode == "compare":
            result = await asyncio.to_thread(core.compare, goal, model, model_b)
        elif mode == "batch":
            result = await asyncio.to_thread(core.batch, goal, model.split(",") if model else [])
        else:
            result = await asyncio.to_thread(core.run_plan, goal, model)
        await _store_task(task_id, "done", result=result)
        try:
            if mode == "compare":
                verdict = (
                    "pass"
                    if result.get("model_a", {}).get("all_passed")
                    and result.get("model_b", {}).get("all_passed")
                    else "fail"
                )
                cost = result.get("model_a", {}).get("total_cost_usd", 0.0)
                tokens = result.get("model_a", {}).get("total_tokens", 0)
            elif mode == "batch":
                verdict = "pass" if all(m.get("all_passed", False) for m in result.get("models", [])) else "fail"
                cost = sum(m.get("total_cost_usd", 0.0) for m in result.get("models", []))
                tokens = sum(m.get("total_tokens", 0) for m in result.get("models", []))
            else:
                verdict = "pass" if result.get("all_passed") else "fail"
                cost = result.get("total_cost_usd", 0.0)
                tokens = result.get("total_tokens", 0)
            await tracer.trace_call(
                task_id,
                1,
                model or "",
                goal,
                json.dumps(result),
                {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
                cost,
                verdict,
            )
        except Exception as exc:
            warnings.warn(f"tracer.trace_call failed: {exc}", stacklevel=2)
    except Exception as exc:
        await _store_task(task_id, "error", error=str(exc))


# ── App ─────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.init_db()
    yield
    tracer.shutdown()


app = FastAPI(title="llm-lab M0", version="0.1.0", lifespan=lifespan)


if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Submit (sync) ───────────────────────────────────────────────────────────


@app.post("/submit", response_model=RunResult)
async def submit(req: IntentRequest):
    model = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    result = core.run_plan(req.goal, model)
    try:
        await tracer.trace_call(
            result["run_id"],
            1,
            model or "",
            result["goal"],
            json.dumps(result),
            {"prompt_tokens": result["total_tokens"], "completion_tokens": 0, "total_tokens": result["total_tokens"]},
            result["total_cost_usd"],
            "pass" if result["all_passed"] else "fail",
        )
    except Exception as exc:
        warnings.warn(f"tracer.trace_call failed: {exc}", stacklevel=2)
    return result


# ── Submit (async) ──────────────────────────────────────────────────────────


@app.post("/submit/async")
async def submit_async(req: IntentRequest, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex[:12]
    await db.save_task(task_id, {"status": "queued", "result": None, "error": None})
    model = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    background_tasks.add_task(_run_task, task_id, req.goal, model, "submit")
    return {"task_id": task_id, "status": "queued"}


# ── Compare (sync) ──────────────────────────────────────────────────────────


@app.post("/compare", response_model=CompareResult)
async def compare(req: IntentRequest):
    model_a = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    model_b = req.preferred_model_2 or os.getenv("LLM_MODEL_2", "gpt-4o-mini")
    result = core.compare(req.goal, model_a, model_b)
    intent_id = db._sha16(req.goal)
    verdict = (
        "pass"
        if result["model_a"]["all_passed"] and result["model_b"]["all_passed"]
        else "fail"
    )
    cost = result["model_a"]["total_cost_usd"]
    tokens = result["model_a"]["total_tokens"]
    try:
        await tracer.trace_call(
            intent_id,
            1,
            model_a or "",
            result["goal"],
            json.dumps(result),
            {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
            cost,
            verdict,
        )
    except Exception as exc:
        warnings.warn(f"tracer.trace_call failed: {exc}", stacklevel=2)
    result["intent_id"] = intent_id
    return result


# ── Compare (async) ─────────────────────────────────────────────────────────


@app.post("/compare/async")
async def compare_async(req: IntentRequest, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex[:12]
    await db.save_task(task_id, {"status": "queued", "result": None, "error": None})
    model_a = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    model_b = req.preferred_model_2 or os.getenv("LLM_MODEL_2", "gpt-4o-mini")
    background_tasks.add_task(_run_task, task_id, req.goal, model_a, "compare", model_b)
    return {"task_id": task_id, "status": "queued"}


# ── Batch ───────────────────────────────────────────────────────────────────


@app.post("/batch", response_model=BatchResult)
async def batch_run(req: BatchRequest):
    result = core.batch(req.goal, req.models)
    intent_id = db._sha16(req.goal)
    verdict = "pass" if all(m.get("all_passed", False) for m in result["models"]) else "fail"
    cost = sum(m.get("total_cost_usd", 0.0) for m in result["models"])
    tokens = sum(m.get("total_tokens", 0) for m in result["models"])
    try:
        await tracer.trace_call(
            intent_id,
            1,
            ",".join(req.models),
            result["goal"],
            json.dumps(result),
            {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
            cost,
            verdict,
        )
    except Exception as exc:
        warnings.warn(f"tracer.trace_call failed: {exc}", stacklevel=2)
    result["intent_id"] = intent_id
    return result


@app.post("/batch/async")
async def batch_async(req: BatchRequest, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex[:12]
    await db.save_task(task_id, {"status": "queued", "result": None, "error": None})
    background_tasks.add_task(_run_task, task_id, req.goal, ",".join(req.models), "batch")
    return {"task_id": task_id, "status": "queued"}


# ── Templates ────────────────────────────────────────────────────────────────


@app.get("/templates")
async def get_templates() -> dict[str, Any]:
    return {"templates": list_templates()}


@app.post("/templates")
async def create_template(tmpl: TemplateDef):
    for existing in list_templates():
        if existing["template_id"] == tmpl.template_id and existing["_source"] == "builtin":
            raise HTTPException(status_code=409, detail=f"builtin template '{tmpl.template_id}' cannot be overwritten")
    save_custom_template(tmpl.template_id, tmpl.model_dump())
    return {"status": "created", "template_id": tmpl.template_id}


@app.delete("/templates/{template_id}")
async def remove_template(template_id: str):
    for tmpl in list_templates():
        if tmpl["template_id"] == template_id and tmpl["_source"] == "builtin":
            raise HTTPException(status_code=403, detail="cannot delete builtin template")
    ok = delete_custom_template(template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="template not found")
    return {"status": "deleted", "template_id": template_id}


# ── Task status ─────────────────────────────────────────────────────────────


@app.get("/status/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task_id not found")
    return task


# ── Compare HTML report ─────────────────────────────────────────────────────


@app.get("/compare/report/{id_a}/{id_b}", response_class=HTMLResponse)
async def compare_report(id_a: str, id_b: str):
    events_a = await _try("read trace A", tracer.get_trace(id_a))
    events_b = await _try("read trace B", tracer.get_trace(id_b))
    if not events_a or not events_b:
        raise HTTPException(status_code=404, detail="one or both runs not found")
    return HTMLResponse(
        content=_COMPARE_REPORT_HTML.format(
            id_a=id_a,
            id_b=id_b,
            json_a=json.dumps(events_a, indent=2, ensure_ascii=False),
            json_b=json.dumps(events_b, indent=2, ensure_ascii=False),
        )
    )


# ── Result / Trace ─────────────────────────────────────────────────────────


@app.get("/result/{intent_id}")
async def get_result(intent_id: str) -> dict[str, Any]:
    summary = await _try("read result", tracer.get_summary(intent_id))
    if summary["events"] == 0:
        raise HTTPException(status_code=404, detail="intent_id not found")
    return summary


@app.get("/trace/{intent_id}")
async def get_trace(intent_id: str) -> Any:
    events = await _try("read trace", tracer.get_trace(intent_id))
    if not events:
        raise HTTPException(status_code=404, detail="intent_id not found")
    return events


# ── Export ───────────────────────────────────────────────────────────────────


@app.get("/export/json/{intent_id}")
async def export_json_endpoint(intent_id: str) -> PlainTextResponse:
    events = await _try("read trace", tracer.get_trace(intent_id))
    if not events:
        raise HTTPException(status_code=404, detail="intent_id not found")
    return PlainTextResponse(
        content=export.export_json(intent_id, events),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{intent_id}.json"'},
    )


@app.get("/export/csv")
async def export_csv_endpoint() -> PlainTextResponse:
    rows = await _try("read events", db.get_all_events())
    if not rows:
        raise HTTPException(status_code=404, detail="no data found")
    return PlainTextResponse(
        content=export.export_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="llm_lab_export.csv"'},
    )


@app.get("/export/xlsx/{intent_id}")
async def export_xlsx_by_intent(intent_id: str) -> Response:
    events = await _try("read trace", tracer.get_trace(intent_id))
    if not events:
        raise HTTPException(status_code=404, detail="intent_id not found")
    try:
        content = export.export_xlsx(events)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{intent_id}.xlsx"'},
    )


@app.get("/export/xlsx")
async def export_xlsx_all() -> Response:
    rows = await _try("read events", db.get_all_events())
    if not rows:
        raise HTTPException(status_code=404, detail="no data found")
    try:
        content = export.export_xlsx(rows)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="llm_lab_export.xlsx"'},
    )


# ── History (list recent runs) ──────────────────────────────────────────────


@app.get("/history")
async def list_history(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    rows = await _try("list intents", db.list_intents(limit))
    return {"runs": rows}


# ── Web UI ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return HTMLResponse(content=_UI_HTML)



