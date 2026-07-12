import asyncio
import hmac
import html
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, status  # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from llm_lab import (  # noqa: E402
    db,
    export,
    tracer,
)
from llm_lab import runner as core  # noqa: E402
from llm_lab.models import BatchRequest, BatchResult, CompareResult, IntentRequest, RunResult, TemplateDef  # noqa: E402
from llm_lab.planner import _TEMPLATE_ID_RE, delete_custom_template, list_templates, save_custom_template  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────

_HTML_DIR = Path(__file__).parent / "templates"


# ── Auth ─────────────────────────────────────────────────────────────────────
# Optional API-key gate. When LLM_LAB_API_KEY is set, every mutating endpoint
# requires a matching ``X-API-Key`` header (or ``Authorization: Bearer <key>``).
# When unset (default for local/dev), the gate is a no-op so existing usage and
# tests are unaffected.

_API_KEY = os.getenv("LLM_LAB_API_KEY")


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    if not _API_KEY:
        return
    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    if not hmac.compare_digest(provided or "", _API_KEY or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


API_KEY_DEP = Depends(require_api_key)


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
    except Exception as exc:
        await _store_task(task_id, "error", error=str(exc))


# ── App ─────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.init_db()
    yield
    tracer.shutdown()


app = FastAPI(title="llm-lab M0", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Submit (sync) ───────────────────────────────────────────────────────────


@app.post("/submit", response_model=RunResult)
async def submit(req: IntentRequest, _: Any = API_KEY_DEP):
    model = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    result = await asyncio.to_thread(core.run_plan, req.goal, model)
    return result


# ── Submit (async) ──────────────────────────────────────────────────────────


@app.post("/submit/async")
async def submit_async(req: IntentRequest, background_tasks: BackgroundTasks, _: Any = API_KEY_DEP):
    task_id = uuid.uuid4().hex[:12]
    await db.save_task(task_id, {"status": "queued", "result": None, "error": None})
    model = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    background_tasks.add_task(_run_task, task_id, req.goal, model, "submit")
    return {"task_id": task_id, "status": "queued"}


# ── Compare (sync) ──────────────────────────────────────────────────────────


@app.post("/compare", response_model=CompareResult)
async def compare(req: IntentRequest, _: Any = API_KEY_DEP):
    model_a = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    model_b = req.preferred_model_2 or os.getenv("LLM_MODEL_2", "gpt-4o-mini")
    result = await asyncio.to_thread(core.compare, req.goal, model_a, model_b)
    return result


# ── Compare (async) ─────────────────────────────────────────────────────────


@app.post("/compare/async")
async def compare_async(req: IntentRequest, background_tasks: BackgroundTasks, _: Any = API_KEY_DEP):
    task_id = uuid.uuid4().hex[:12]
    await db.save_task(task_id, {"status": "queued", "result": None, "error": None})
    model_a = req.preferred_model or os.getenv("LLM_MODEL", "gpt-4o")
    model_b = req.preferred_model_2 or os.getenv("LLM_MODEL_2", "gpt-4o-mini")
    background_tasks.add_task(_run_task, task_id, req.goal, model_a, "compare", model_b)
    return {"task_id": task_id, "status": "queued"}


# ── Batch ───────────────────────────────────────────────────────────────────


@app.post("/batch", response_model=BatchResult)
async def batch_run(req: BatchRequest, _: Any = API_KEY_DEP):
    result = await asyncio.to_thread(core.batch, req.goal, req.models)
    return result


@app.post("/batch/async")
async def batch_async(req: BatchRequest, background_tasks: BackgroundTasks, _: Any = API_KEY_DEP):
    task_id = uuid.uuid4().hex[:12]
    await db.save_task(task_id, {"status": "queued", "result": None, "error": None})
    background_tasks.add_task(_run_task, task_id, req.goal, ",".join(req.models), "batch")
    return {"task_id": task_id, "status": "queued"}


# ── Templates ────────────────────────────────────────────────────────────────


@app.get("/templates")
async def get_templates(_: Any = API_KEY_DEP) -> dict[str, Any]:
    return {"templates": list_templates()}


@app.post("/templates")
async def create_template(tmpl: TemplateDef, _: Any = API_KEY_DEP):
    for existing in list_templates():
        if existing["template_id"] == tmpl.template_id and existing["_source"] == "builtin":
            raise HTTPException(status_code=409, detail=f"builtin template '{tmpl.template_id}' cannot be overwritten")
    save_custom_template(tmpl.template_id, tmpl.model_dump())
    return {"status": "created", "template_id": tmpl.template_id}


@app.delete("/templates/{template_id}")
async def remove_template(template_id: str, _: Any = API_KEY_DEP):
    if not _TEMPLATE_ID_RE.match(template_id):
        raise HTTPException(status_code=400, detail="invalid template_id")
    for tmpl in list_templates():
        if tmpl["template_id"] == template_id and tmpl["_source"] == "builtin":
            raise HTTPException(status_code=403, detail="cannot delete builtin template")
    try:
        ok = delete_custom_template(template_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid template_id") from None
    if not ok:
        raise HTTPException(status_code=404, detail="template not found")
    return {"status": "deleted", "template_id": template_id}


# ── Task status ─────────────────────────────────────────────────────────────


@app.get("/status/{task_id}")
async def get_task_status(task_id: str, _: Any = API_KEY_DEP) -> dict[str, Any]:
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task_id not found")
    return task


# ── Compare HTML report ─────────────────────────────────────────────────────


@app.get("/compare/report/{id_a}/{id_b}", response_class=HTMLResponse)
async def compare_report(id_a: str, id_b: str, _: Any = API_KEY_DEP):
    events_a = await _try("read trace A", tracer.get_trace(id_a))
    events_b = await _try("read trace B", tracer.get_trace(id_b))
    if not events_a or not events_b:
        raise HTTPException(status_code=404, detail="one or both runs not found")
    return HTMLResponse(
        content=_COMPARE_REPORT_HTML.format(
            id_a=id_a,
            id_b=id_b,
            json_a=html.escape(json.dumps(events_a, indent=2, ensure_ascii=False)),
            json_b=html.escape(json.dumps(events_b, indent=2, ensure_ascii=False)),
        )
    )


# ── Result / Trace ─────────────────────────────────────────────────────────


@app.get("/result/{intent_id}")
async def get_result(intent_id: str, _: Any = API_KEY_DEP) -> dict[str, Any]:
    summary = await _try("read result", tracer.get_summary(intent_id))
    if summary["events"] == 0:
        raise HTTPException(status_code=404, detail="intent_id not found")
    return summary


@app.get("/trace/{intent_id}")
async def get_trace(intent_id: str, _: Any = API_KEY_DEP) -> Any:
    events = await _try("read trace", tracer.get_trace(intent_id))
    if not events:
        raise HTTPException(status_code=404, detail="intent_id not found")
    return events


# ── Export ───────────────────────────────────────────────────────────────────


@app.get("/export/json/{intent_id}")
async def export_json_endpoint(intent_id: str, _: Any = API_KEY_DEP) -> PlainTextResponse:
    events = await _try("read trace", tracer.get_trace(intent_id))
    if not events:
        raise HTTPException(status_code=404, detail="intent_id not found")
    return PlainTextResponse(
        content=export.export_json(intent_id, events),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{intent_id}.json"'},
    )


@app.get("/export/csv")
async def export_csv_endpoint(_: Any = API_KEY_DEP) -> PlainTextResponse:
    rows = await _try("read events", db.get_all_events())
    if not rows:
        raise HTTPException(status_code=404, detail="no data found")
    return PlainTextResponse(
        content=export.export_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="llm_lab_export.csv"'},
    )


@app.get("/export/xlsx/{intent_id}")
async def export_xlsx_by_intent(intent_id: str, _: Any = API_KEY_DEP) -> Response:
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
async def export_xlsx_all(_: Any = API_KEY_DEP) -> Response:
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
async def list_history(limit: int = Query(20, ge=1, le=200), _: Any = API_KEY_DEP) -> dict[str, Any]:
    rows = await _try("list intents", db.list_intents(limit))
    return {"runs": rows}


# ── Web UI ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return HTMLResponse(content=_UI_HTML)



