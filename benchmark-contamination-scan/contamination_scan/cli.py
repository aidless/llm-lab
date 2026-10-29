"""CLI for the contamination preflight."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
from llm_lab.evalops import write_json, write_jsonl

from .data import build_fixture, load_ceval_tasks, read_jsonl
from .embeddings import Embedder, save_vectors
from .reporting import build_report, write_reports
from .scanner import Match, best_match, is_suspect

ROOT = Path(__file__).resolve().parent.parent
LLM_LAB_ROOT = ROOT.parent
DEFAULT_TASKS = LLM_LAB_ROOT / "evalops" / "ceval-201" / "tasks-v1.jsonl"
DEFAULT_KNOWLEDGE = LLM_LAB_ROOT / "knowledge"
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_EVIDENCE = ROOT / "evidence"
DEFAULT_REVIEW = ROOT / "review"


def _task_text(task: dict[str, Any]) -> str:
    choices = task.get("choices") or {}
    return f"{task['question']}\n" + "\n".join(f"{key}. {choices[key]}" for key in sorted(choices))


def command_build(args: argparse.Namespace) -> int:
    manifest = build_fixture(
        tasks_path=Path(args.tasks),
        knowledge_root=Path(args.knowledge),
        output_dir=Path(args.output),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _human_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _tool_provenance(args: argparse.Namespace, *, task_count: int, document_count: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": args.embedding_model,
        "base_url": args.base_url,
        "ngram_n": args.ngram_n,
        "task_count": task_count,
        "document_count": document_count,
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            version = client.get(f"{args.base_url.rstrip('/')}/api/version")
            version.raise_for_status()
            tags = client.get(f"{args.base_url.rstrip('/')}/api/tags")
            tags.raise_for_status()
        payload["ollama_version"] = str(version.json().get("version", "unknown"))
        models = tags.json().get("models", [])
        selected = next(
            (
                item
                for item in models
                if isinstance(item, dict) and str(item.get("name")) == args.embedding_model
            ),
            {},
        )
        payload["embedding_model_digest"] = selected.get("digest")
        payload["embedding_model_details"] = selected.get("details")
    except (httpx.HTTPError, ValueError) as exc:
        payload["ollama_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def command_scan(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    documents = read_jsonl(dataset / manifest["corpus"]["path"])
    labels = {str(row["task_id"]): row for row in read_jsonl(dataset / manifest["labels"]["path"])}
    tasks = load_ceval_tasks(Path(args.tasks))
    texts = [_task_text(task) for task in tasks]
    embedder = Embedder(model=args.embedding_model, base_url=args.base_url)
    try:
        task_vectors = embedder.encode(texts)
        document_vectors = embedder.encode([str(document["text"]) for document in documents])
    finally:
        embedder.close()

    evidence = Path(args.output)
    evidence.mkdir(parents=True, exist_ok=True)
    save_vectors(evidence / "task-embeddings.json", task_vectors)
    save_vectors(evidence / "document-embeddings.json", document_vectors)
    write_json(
        evidence / "tool-versions.json",
        _tool_provenance(args, task_count=len(tasks), document_count=len(documents)),
    )

    rows: list[dict[str, Any]] = []
    for task, text, vector in zip(tasks, texts, task_vectors, strict=True):
        match = best_match(
            task_text=text,
            task_vector=vector,
            documents=documents,
            document_vectors=document_vectors,
            ngram_n=args.ngram_n,
        )
        gold = labels[str(task["id"])]
        rows.append(
            {
                "task_id": str(task["id"]),
                "subject": str(task.get("subject")),
                "split": gold["split"],
                "gold_contaminated": bool(gold["gold_contaminated"]),
                "seed_type": gold.get("seed_type"),
                "match": match.to_dict(),
            }
        )
    write_jsonl(evidence / "matches.jsonl", rows)
    report = build_report(
        rows=rows,
        corpus_manifest=manifest,
        human_reviews=_human_reviews(Path(args.human_reviews)),
    )
    write_reports(evidence, report)
    write_review_queue(
        rows,
        report["thresholds"],
        DEFAULT_REVIEW,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "metrics": report["metrics"],
                "thresholds": report["thresholds"],
                "report": str(evidence / "report.html"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "PASS" else 2


def write_review_queue(
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    output: Path,
) -> None:
    task_lookup = {
        str(task["id"]): _task_text(task)
        for task in load_ceval_tasks(DEFAULT_TASKS)
    }
    document_lookup = {
        str(document["id"]): str(document["text"])
        for document in read_jsonl(DEFAULT_DATASET / "corpus-v1.jsonl")
    }
    def suspect(row: dict[str, Any]) -> bool:
        return is_suspect(
            Match(**row["match"]),
            ngram_threshold=thresholds["ngram_threshold"],
            semantic_threshold=thresholds["semantic_threshold"],
        )

    suspects = [
        row
        for row in rows
        if suspect(row)
    ]
    controls = [row for row in rows if row not in suspects][:10]
    selected = (suspects + controls)[:40]
    items = []
    for row in selected:
        match = row["match"]
        items.append(
            {
                "review_id": f"contamination:{row['task_id']}",
                "task_id": row["task_id"],
                "subject": row["subject"],
                "question_text": task_lookup.get(str(row["task_id"]), ""),
                "source_text": document_lookup.get(str(match.get("document_id")), ""),
                "suggested_contaminated": suspect(row),
                "match": match,
                "decision": "pending",
                "reviewer": "",
                "reviewed_at_utc": "",
                "contaminated": None,
                "notes": "",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    queue = {
        "schema_version": "benchmark-contamination.review-queue.v1",
        "items": items,
    }
    write_json(output / "review-queue.json", queue)
    (output / "review.html").write_text(_review_html(queue), encoding="utf-8")


def _review_html(queue: dict[str, Any]) -> str:
    data = json.dumps(queue, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>评测题污染白话复核</title><style>
:root{{--ink:#17201c;--paper:#edf1e7;--card:#fffdf7;--line:#c9d0c5;--yes:#a23b2d;--no:#176247;--amber:#9a6517}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#f7f0df,var(--paper));color:var(--ink);font:17px/1.6 "Noto Sans CJK SC",sans-serif}}header{{position:sticky;top:0;z-index:2;padding:18px 24px;background:#f8f4e9ef;border-bottom:2px solid var(--ink);display:flex;justify-content:space-between}}h1{{margin:0;font:800 29px/1.1 serif}}main{{max-width:1180px;margin:28px auto 80px;padding:0 20px;display:grid;grid-template-columns:1fr 290px;gap:20px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 14px 35px #34443814}}.compare{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.text{{white-space:pre-wrap;background:#f5eddd;border-top:5px solid var(--amber);padding:15px;border-radius:9px;min-height:180px}}button{{padding:11px 15px;border:1px solid var(--ink);border-radius:999px;background:white;font-weight:800;cursor:pointer}}button.yes{{background:var(--yes);color:white}}button.no{{background:var(--no);color:white}}.actions{{display:flex;gap:9px;flex-wrap:wrap}}aside{{position:sticky;top:100px;align-self:start}}label{{display:block;font-weight:800;margin-top:12px}}input,textarea{{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;font:inherit}}textarea{{min-height:90px}}progress{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#17201c;color:#fff;padding:14px;border-radius:9px}}.muted{{color:#665f54}}@media(max-width:850px){{main{{grid-template-columns:1fr}}aside{{position:static}}.compare{{grid-template-columns:1fr}}}}</style></head><body>
<header><h1>评测题污染白话复核</h1><span>仅在本机运行 · 不上传数据</span></header><main><section class="card" id="card"></section><aside class="card"><progress id="progress" value="0" max="{len(queue['items'])}"></progress><div id="counter"></div><label>复核人标识</label><input id="reviewer" placeholder="例如：liuzewen"><label>备注</label><textarea id="notes" placeholder="可选：指出相同或改写的位置"></textarea><button class="yes" id="export" style="width:100%;margin-top:12px">导出人工复核结果</button><p class="muted">证据不足不会算作有效标签。</p></aside></main><script>
const queue={data};const practice=[{{q:"两段文字逐字相同，只改变了换行。算污染吗？",a:true,e:"算。直接复制或仅改变格式属于明显污染。"}},{{q:"两段都谈数学，但题目、数字和问法完全不同。算污染吗？",a:false,e:"通常不算。主题相同不等于样本被复制。"}},{{q:"只看到相似度分数，没有来源原文。能确定吗？",a:null,e:"不能。没有可比较原文时应选证据不足。"}}];let phase=localStorage.getItem("contamination-guide-practice-done")==="yes"?"review":"intro",practiceIndex=0;let state=JSON.parse(localStorage.getItem("contamination-review-v2")||"null");if(!state||state.items.length!==queue.items.length)state={{index:0,items:queue.items}};const card=document.getElementById("card"),counter=document.getElementById("counter"),progress=document.getElementById("progress"),reviewer=document.getElementById("reviewer"),notes=document.getElementById("notes");function esc(v){{return String(v??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]))}}function done(){{return state.items.filter(x=>typeof x.contaminated==="boolean"&&x.decision!=="pending").length}}function save(){{localStorage.setItem("contamination-review-v2",JSON.stringify(state));progress.value=done();counter.textContent=`已复核 ${{done()}} / ${{state.items.length}}`;reviewer.value=state.items[state.index]?.reviewer||reviewer.value||"";notes.value=state.items[state.index]?.notes||""}}
function intro(){{card.innerHTML=`<h2>先做 3 道练习</h2><p>污染不是“题目很常见”，而是评测题可能被直接复制或轻微改写进了训练资料。</p><div class="text"><b>判断重点：</b>比较具体文字、选项、数字和结构。不要只看相似度分数。</div><p><b>本地安全评测，不是生产安全系统。</b></p><button class="no" onclick="startPractice()">开始练习</button>`}}function startPractice(){{phase="practice";practiceIndex=0;renderPractice()}}function renderPractice(msg=""){{const x=practice[practiceIndex];card.innerHTML=`<div class="muted">练习 ${{practiceIndex+1}} / 3</div><h2>${{x.q}}</h2><div class="actions"><button class="yes" onclick="answerPractice(true)">算污染</button><button class="no" onclick="answerPractice(false)">不算污染</button><button onclick="answerPractice(null)">证据不足</button></div><p>${{msg}}</p>`}}function answerPractice(v){{const x=practice[practiceIndex],ok=v===x.a;renderPractice(`<b>${{ok?"答对了":"再想一下"}}：</b>${{x.e}} <button onclick="nextPractice()">继续</button>`)}}function nextPractice(){{practiceIndex++;if(practiceIndex>=3){{localStorage.setItem("contamination-guide-practice-done","yes");phase="review";render()}}else renderPractice()}}
function render(){{if(phase==="intro")return intro();if(phase==="practice")return renderPractice();const x=state.items[state.index],missing=!x.question_text||!x.source_text;card.innerHTML=`<div class="muted">第 ${{state.index+1}} / ${{state.items.length}} 条 · ${{esc(x.task_id)}}</div><h2>请比较下面两段内容</h2><div class="compare"><div><h3>评测题</h3><div class="text">${{esc(x.question_text||"未提供评测题原文")}}</div></div><div><h3>疑似来源</h3><div class="text">${{esc(x.source_text||"未提供疑似来源原文")}}</div></div></div><h2>两段内容是否像是直接复制或轻微改写？</h2>${{missing?'<p class="text"><b>证据不足：</b>缺少一侧原文，请不要猜。</p>':''}}<div class="actions"><button class="yes" onclick="label(true)">是，疑似污染</button><button class="no" onclick="label(false)">否，不像污染</button><button onclick="markUncertain()">证据不足 / 我不确定</button><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button></div><details><summary>查看技术细节</summary><pre>${{esc(JSON.stringify(x.match,null,2))}}</pre></details>`;save()}}function label(v){{if(!reviewer.value.trim()){{alert("请先填写复核人标识");return}}const x=state.items[state.index];x.contaminated=v;x.decision=v?"contaminated":"clean";x.reviewer=reviewer.value.trim();x.reviewed_at_utc=new Date().toISOString();x.notes=notes.value.trim();save();if(state.index<state.items.length-1){{state.index++;render()}}}}function markUncertain(){{const x=state.items[state.index];x.contaminated=null;x.decision="pending";x.reviewer=reviewer.value.trim();x.notes=(notes.value.trim()+" [证据不足]").trim();save();if(state.index<state.items.length-1){{state.index++;render()}}}}function move(d){{const x=state.items[state.index];x.reviewer=reviewer.value.trim();x.notes=notes.value.trim();state.index=Math.max(0,Math.min(state.items.length-1,state.index+d));render()}}
document.getElementById("export").onclick=()=>{{const payload={{schema_version:"benchmark-contamination.review-export.v1",exported_at_utc:new Date().toISOString(),items:state.items}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="review-export.json";a.click()}};render();</script></body></html>"""


def command_report(args: argparse.Namespace) -> int:
    evidence = Path(args.output)
    rows = read_jsonl(evidence / "matches.jsonl")
    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    report = build_report(
        rows=rows,
        corpus_manifest=manifest,
        human_reviews=_human_reviews(Path(args.human_reviews)),
    )
    write_reports(evidence, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


def command_gate(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    decision = str(report["decision"])
    preflight = {
        "schema_version": "evalops.preflight.v1",
        "source": "benchmark-contamination-scan",
        "decision": decision,
        "reasons": report.get("gate_reasons", []),
        "metrics": report.get("metrics", {}),
        "task_manifest": report.get("manifest", {}).get("tasks", {}),
        "human_review": report.get("human_review", {}),
    }
    output = Path(args.output)
    write_json(output, preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision == "PASS" else 3 if decision == "BLOCK" else 2


def command_ingest_review(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.export).read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise TypeError("review export items must be a list")
    complete = []
    for item in items:
        if item.get("decision") == "pending":
            continue
        if not item.get("reviewer") or not item.get("reviewed_at_utc"):
            raise ValueError("each reviewed item requires reviewer and timestamp")
        if not isinstance(item.get("contaminated"), bool):
            raise ValueError("each reviewed item requires a boolean contamination label")
        complete.append(item)
    write_jsonl(Path(args.output), complete)
    print(json.dumps({"reviewed_records": len(complete)}, indent=2))
    return 0 if len(complete) >= 40 else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="contamination-scan",
        description="Benchmark contamination preflight",
    )
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-fixture")
    build.add_argument("--tasks", default=str(DEFAULT_TASKS))
    build.add_argument("--knowledge", default=str(DEFAULT_KNOWLEDGE))
    build.add_argument("--output", default=str(DEFAULT_DATASET))
    build.set_defaults(func=command_build)
    scan = sub.add_parser("scan")
    scan.add_argument("--tasks", default=str(DEFAULT_TASKS))
    scan.add_argument("--dataset", default=str(DEFAULT_DATASET))
    scan.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    scan.add_argument("--base-url", default="http://127.0.0.1:11434")
    scan.add_argument("--embedding-model", default="nomic-embed-text:latest")
    scan.add_argument("--ngram-n", type=int, default=13)
    scan.add_argument("--human-reviews", default=str(DEFAULT_REVIEW / "reviewed.jsonl"))
    scan.set_defaults(func=command_scan)
    report = sub.add_parser("report")
    report.add_argument("--dataset", default=str(DEFAULT_DATASET))
    report.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    report.add_argument("--human-reviews", default=str(DEFAULT_REVIEW / "reviewed.jsonl"))
    report.set_defaults(func=command_report)
    gate = sub.add_parser("gate")
    gate.add_argument("--report", default=str(DEFAULT_EVIDENCE / "report.json"))
    gate.add_argument("--output", default=str(DEFAULT_EVIDENCE / "evalops-preflight.json"))
    gate.set_defaults(func=command_gate)
    ingest = sub.add_parser("ingest-review")
    ingest.add_argument("--export", required=True)
    ingest.add_argument("--output", default=str(DEFAULT_REVIEW / "reviewed.jsonl"))
    ingest.set_defaults(func=command_ingest_review)
    return root


def entry() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    entry()
