"""Human-label packet for the 200 safety-judge cases."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "safety-judge.review-export.v1"


def build_packet(
    cases: list[dict[str, Any]],
    output: Path,
    *,
    case_set_hash: str,
) -> dict[str, Any]:
    items = [
        {
            "case_id": case["id"],
            "slice": _case_slice(case["id"]),
            "request": case["request"],
            "suggested_label": case["draft_label"],
            "rationale": case.get("rationale", ""),
            "decision": "pending",
            "gold_label": "",
            "reviewer": "",
            "reviewed_at_utc": "",
            "notes": "",
        }
        for case in cases
    ]
    payload = {
        "schema_version": "safety-judge.review-queue.v1",
        "case_count": len(items),
        "case_set_hash": case_set_hash,
        "items": items,
    }
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"review output is not a regular directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"review directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "review-queue.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "review.html").write_text(_html(payload), encoding="utf-8")
    return {"items": len(items), "file": str(output / "review.html")}


def _case_slice(case_id: str) -> str:
    if case_id.startswith(("inj-", "pi-", "pis-")):
        return "prompt_injection"
    if case_id.startswith(("auth-", "ua-", "uas-")):
        return "unauthorized_request"
    return "normal_request"


def validate_export(
    path: Path,
    *,
    expected_case_ids: set[str],
    expected_case_set_hash: str,
    expected_count: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported review export schema")
    if payload.get("case_set_hash") != expected_case_set_hash:
        raise ValueError("review export case-set hash does not match the frozen cases")
    rows = payload.get("items")
    if not isinstance(rows, list):
        raise TypeError("review items must be a list")
    completed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise TypeError(f"review item {index} is not an object")
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"review item {index} has a missing or duplicate case id")
        seen.add(case_id)
        decision = str(row.get("decision", "pending"))
        if decision == "pending":
            continue
        if decision not in {"safe", "unsafe"}:
            raise ValueError(f"review {case_id} has an invalid decision")
        label = str(row.get("gold_label", ""))
        reviewer = str(row.get("reviewer", "")).strip()
        reviewed_at = str(row.get("reviewed_at_utc", "")).strip()
        if label not in {"safe", "unsafe"}:
            raise ValueError(f"review {case_id} has an invalid gold label")
        if not reviewer:
            raise ValueError(f"review {case_id} has no reviewer")
        try:
            reviewed_dt = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"review {case_id} has an invalid timestamp") from exc
        if reviewed_dt.tzinfo is None:
            raise ValueError(f"review {case_id} timestamp must include a timezone")
        completed.append(
            {
                "case_id": case_id,
                "slice": str(row.get("slice")),
                "gold_label": label,
                "reviewer": reviewer,
                "reviewed_at_utc": reviewed_at,
                "notes": str(row.get("notes", "")),
                "decision": decision,
            }
        )
    completed_ids = {str(item["case_id"]) for item in completed}
    unexpected = completed_ids - expected_case_ids
    if unexpected:
        raise ValueError(f"review export contains unknown cases: {sorted(unexpected)}")
    missing = expected_case_ids - completed_ids
    summary = {
        "schema_version": "safety-judge.review-summary.v1",
        "case_set_hash": expected_case_set_hash,
        "expected_cases": expected_count,
        "reviewed_cases": len(completed),
        "missing_cases": len(missing),
        "human_gate_200_cases": (len(completed) == expected_count and not missing and not unexpected),
        "reviewer_ids": sorted({str(item["reviewer"]) for item in completed}),
    }
    return completed, summary


def _html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety Judge Human Labels</title>
<style>
:root {{ --ink:#15191b; --paper:#eef0ed; --card:#fffdf8; --line:#d0d1ca;
  --safe:#17634b; --unsafe:#ae3b31; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.5 "Aptos",sans-serif}}
header{{position:sticky;top:0;background:#f7f3e9;border-bottom:2px solid var(--ink);
padding:18px 24px;z-index:2;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}}
h1{{font:800 30px/1 Georgia,serif;margin:0}} main{{max-width:1050px;margin:30px auto;
padding:0 20px 80px;display:grid;grid-template-columns:1fr 300px;gap:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px}}
.eyebrow{{font-size:12px;letter-spacing:.15em;text-transform:uppercase;font-weight:900;
color:var(--unsafe)}} h2{{font:700 28px/1.2 Georgia,serif;margin:8px 0 14px}}
.request{{white-space:pre-wrap;background:#f3eee3;border-left:5px solid var(--unsafe);
padding:16px;border-radius:8px}} .suggest{{background:#e8f1ed;color:var(--safe);
padding:12px;border-radius:10px;margin:14px 0}} .actions{{display:flex;gap:10px;flex-wrap:wrap}}
button{{padding:11px 15px;border:1px solid var(--ink);border-radius:999px;background:white;
font-weight:800;cursor:pointer}} button.safe{{background:var(--safe);color:white}}
button.unsafe{{background:var(--unsafe);color:white}} aside{{position:sticky;top:100px;
align-self:start}} progress{{width:100%;height:14px}} label{{display:block;font-weight:800;
margin-top:12px}} input,textarea{{width:100%;padding:10px;border:1px solid var(--line);
border-radius:8px;font:inherit}} textarea{{min-height:90px}} @media(max-width:800px){{
main{{grid-template-columns:1fr}}aside{{position:static}}}}
</style></head><body>
<header><h1>Safety Judge Human Labels</h1><div id="counter"></div></header>
<main><section class="card" id="card"></section>
<aside class="card"><progress id="progress" value="0" max="{payload["case_count"]}"></progress>
<label>Reviewer ID</label><input id="reviewer" placeholder="e.g. liuzewen">
<label>Notes</label><textarea id="notes"></textarea>
<button class="unsafe" id="export" style="width:100%;margin-top:12px">
Export labeled packet</button>
<p>Keys: S safe, U unsafe, J next, K previous. Labels require a reviewer ID.</p>
</aside></main>
<script>
const queue={data};
let state=JSON.parse(localStorage.getItem("judge-review-v1")||"null");
if(!state||state.items.length!==queue.items.length)state={{index:0,items:queue.items}};
const card=document.getElementById("card"),counter=document.getElementById("counter"),
progress=document.getElementById("progress"),reviewer=document.getElementById("reviewer"),
notes=document.getElementById("notes");
function done(){{return state.items.filter(x=>x.decision!=="pending").length}}
function save(){{localStorage.setItem("judge-review-v1",JSON.stringify(state));progress.value=done();
counter.textContent=`${{done()}} / ${{state.items.length}} reviewed`;
reviewer.value=state.items[state.index].reviewer||reviewer.value||"";
notes.value=state.items[state.index].notes||"";}}
function esc(v){{return String(v).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",
">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]))}}
function render(){{const x=state.items[state.index];card.innerHTML=`
<div class="eyebrow">${{state.index+1}} / ${{state.items.length}} · ${{esc(x.slice)}}</div>
<h2>${{esc(x.case_id)}}</h2><div class="request">${{esc(x.request)}}</div>
<div class="suggest"><b>Draft suggestion:</b> ${{esc(x.suggested_label)}}<br>
${{esc(x.rationale)}}</div><div class="actions">
<button class="safe" onclick="label('safe')">Safe [S]</button>
<button class="unsafe" onclick="label('unsafe')">Unsafe [U]</button>
<button onclick="move(-1)">Previous [K]</button><button onclick="move(1)">Next [J]</button>
</div>`;save();}}
function label(value){{if(!reviewer.value.trim()){{alert("Reviewer ID required");return}}
const x=state.items[state.index];x.gold_label=value;x.decision=value;x.reviewer=reviewer.value.trim();
x.reviewed_at_utc=new Date().toISOString();x.notes=notes.value.trim();save();
if(state.index<state.items.length-1){{state.index++;render();}}}}
function move(d){{const x=state.items[state.index];x.reviewer=reviewer.value.trim();
x.notes=notes.value.trim();state.index=Math.max(0,Math.min(state.items.length-1,
state.index+d));render();}}
document.getElementById("export").onclick=()=>{{const payload={{schema_version:
"safety-judge.review-export.v1",case_set_hash:queue.case_set_hash,
exported_at_utc:new Date().toISOString(),items:state.items}};
const blob=new Blob([JSON.stringify(payload,null,2)],{{type:"application/json"}});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);
a.download="judge-reviews-export.json";a.click();}};
document.addEventListener("keydown",e=>{{if(["INPUT","TEXTAREA"].includes(e.target.tagName))
return;const k=e.key.toLowerCase();if(k==="s")label("safe");if(k==="u")label("unsafe");
if(k==="j")move(1);if(k==="k")move(-1);}});render();
</script></body></html>"""
