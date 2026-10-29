"""Build and validate the human-review packet.

The queue is deliberately separate from execution.  Assistant-authored draft
labels and judge outputs are suggestions, never retroactively described as
human annotations.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_lab.safety.models import SafetyTask, canonical_json, sha256_text

from .output_guard import claim_empty_output

TASK_REVIEW_SCHEMA = "agent-safety.review-item.v1"
EXPORT_SCHEMA = "agent-safety.review-export.v1"


def _task_review_item(task: SafetyTask) -> dict[str, Any]:
    label = task.draft_label
    return {
        "schema_version": TASK_REVIEW_SCHEMA,
        "review_id": f"task:{task.id}",
        "kind": "task_label",
        "task_id": task.id,
        "category": task.category,
        "title": task.title,
        "goal": task.user_goal,
        "suggested": {
            "expected_terminals": list(label.expected_terminals),
            "must_halt": label.must_halt,
            "benign": label.benign,
            "required_tools": list(label.required_tools),
            "rationale": label.rationale,
        },
        "decision": "pending",
        "reviewer": "",
        "reviewed_at_utc": "",
        "notes": "",
    }


def _prepare_trajectory_items(
    trajectories: list[dict[str, Any]],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    def priority(item: dict[str, Any]) -> tuple[int, str]:
        score = 0
        score += 4 if item.get("unsafe_executed") else 0
        score += 2 if item.get("policy_denials") else 0
        score += 2 if item.get("tool_errors") else 0
        score += 1 if item.get("parse_errors") else 0
        score += 1 if item.get("model_errors") else 0
        return (-score, str(item.get("task_id")))

    required = [item for item in trajectories if trajectory_review_required(item)]
    required_ids = {trajectory_review_id(item) for item in required}
    optional = [item for item in trajectories if trajectory_review_id(item) not in required_ids]
    extra_limit = max(0, limit - len(required))
    selected = sorted(required, key=priority) + sorted(optional, key=priority)[:extra_limit]
    items: list[dict[str, Any]] = []
    for trajectory in selected:
        trajectory_hash = sha256_text(canonical_json(trajectory))
        items.append(
            {
                "schema_version": TASK_REVIEW_SCHEMA,
                "review_id": trajectory_review_id(trajectory),
                "kind": "trajectory_review",
                "task_id": trajectory["task_id"],
                "category": trajectory["category"],
                "title": (f"{trajectory['model']} / policy-{trajectory['policy_mode']} / {trajectory['task_id']}"),
                "goal": trajectory.get("terminal_reason", ""),
                "suggested": {
                    "safe_outcome": not bool(trajectory.get("unsafe_executed")),
                    "terminal": trajectory.get("terminal"),
                    "unsafe_executed": bool(trajectory.get("unsafe_executed")),
                    "unsafe_attempt": bool(trajectory.get("unsafe_attempt")),
                    "policy_denials": int(trajectory.get("policy_denials") or 0),
                    "tool_calls": trajectory.get("tool_calls", []),
                    "stop_reason": trajectory.get("terminal_reason", ""),
                    "trajectory_hash": trajectory_hash,
                },
                "review_required": trajectory_review_required(trajectory),
                "decision": "pending",
                "reviewer": "",
                "reviewed_at_utc": "",
                "notes": "",
            }
        )
    return items


def write_review_packet(
    output_dir: Path,
    tasks: list[SafetyTask],
    trajectories: list[dict[str, Any]],
    *,
    trajectory_limit: int = 24,
) -> dict[str, Any]:
    items = [_task_review_item(task) for task in tasks]
    items.extend(_prepare_trajectory_items(trajectories, limit=trajectory_limit))
    required_trajectories = [trajectory_review_id(item) for item in trajectories if trajectory_review_required(item)]
    task_hash = sha256_text(canonical_json([task.to_dict() for task in tasks]))
    trajectory_hash = trajectory_set_hash(trajectories)
    claim_empty_output(output_dir, "review")
    queue = {
        "schema_version": "agent-safety.review-queue.v1",
        "instructions": (
            "Review every task-label record and every trajectory whose unsafe "
            "attempt is marked required. A complete packet has all task labels, "
            "100% required-trajectory coverage, and no flagged required records."
        ),
        "task_set_hash": task_hash,
        "trajectory_set_hash": trajectory_hash,
        "task_label_count": len(tasks),
        "trajectory_review_count": len(items) - len(tasks),
        "required_trajectory_count": len(required_trajectories),
        "required_trajectory_ids": sorted(required_trajectories),
        "items": items,
    }
    (output_dir / "review-queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.html").write_text(_review_html(queue), encoding="utf-8")
    (output_dir / "README.md").write_text(
        _review_readme(len(items), len(tasks), len(required_trajectories)),
        encoding="utf-8",
    )
    return {
        "items": len(items),
        "task_label_items": len(tasks),
        "trajectory_items": len(items) - len(tasks),
        "required_trajectory_items": len(required_trajectories),
    }


def _review_readme(
    item_count: int,
    task_count: int,
    required_trajectory_count: int,
) -> str:
    return f"""# Human review packet

Open `review.html` in a browser. It is a local static page: no data is uploaded.

The packet contains **{item_count}** records. For the project acceptance gate:

1. Review all {task_count} task-label records.
2. Review all {required_trajectory_count} required unsafe-attempt trajectories.
3. Enter a reviewer identifier and export `reviews-export.json`.
4. Run:

```bash
../.venv/bin/python -m agent_safety_eval ingest-review \
  --export review/reviews-export.json \
  --output review/reviews-human.jsonl
```

Only validated records with a reviewer identifier, UTC timestamp and explicit
decision are counted as human review. The assistant-authored draft labels never
count as human labels.
"""


def validate_export(
    path: Path,
    *,
    expected_task_ids: set[str],
    required_trajectory_ids: set[str],
    expected_task_set_hash: str,
    expected_trajectory_set_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("review export has an unsupported schema_version")
    if payload.get("task_set_hash") != expected_task_set_hash:
        raise ValueError("review export task-set hash does not match the frozen tasks")
    if payload.get("trajectory_set_hash") != expected_trajectory_set_hash:
        raise ValueError("review export trajectory-set hash does not match the run")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review export items must be a list")
    complete: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"review item {index} is not an object")
        review_id = str(raw.get("review_id", ""))
        if not review_id or review_id in seen:
            raise ValueError(f"review item {index} has a missing or duplicate id")
        seen.add(review_id)
        if str(raw.get("decision", "pending")) == "pending":
            continue
        reviewer = str(raw.get("reviewer", "")).strip()
        reviewed_at = str(raw.get("reviewed_at_utc", "")).strip()
        if not reviewer:
            raise ValueError(f"review {review_id} has no reviewer id")
        try:
            reviewed_dt = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"review {review_id} has an invalid timestamp") from exc
        if reviewed_dt.tzinfo is None:
            raise ValueError(f"review {review_id} timestamp must include a timezone")
        decision = str(raw.get("decision"))
        if decision not in {"accept", "revise", "flag"}:
            raise ValueError(f"review {review_id} has an invalid decision")
        complete.append(dict(raw))
    unknown_task_reviews = {
        str(item.get("task_id"))
        for item in complete
        if item.get("kind") == "task_label" and str(item.get("task_id")) not in expected_task_ids
    }
    if unknown_task_reviews:
        raise ValueError(f"review export contains unknown task labels: {sorted(unknown_task_reviews)}")
    summary = summarize_review_records(
        complete,
        expected_task_ids=expected_task_ids,
        required_trajectory_ids=required_trajectory_ids,
        task_set_hash=expected_task_set_hash,
        trajectory_set_hash=expected_trajectory_set_hash,
    )
    return complete, summary


def summarize_review_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_task_ids: set[str],
    required_trajectory_ids: set[str],
    task_set_hash: str,
    trajectory_set_hash: str,
) -> dict[str, Any]:
    complete = list(records)
    task_reviews = {str(item.get("task_id")) for item in complete if item.get("kind") == "task_label"}
    trajectory_reviews = [item for item in complete if item.get("kind") == "trajectory_review"]
    accepted_task_reviews = {
        str(item.get("task_id"))
        for item in complete
        if item.get("kind") == "task_label" and item.get("decision") in {"accept", "revise"}
    }
    reviewed_required = {
        str(item["review_id"])
        for item in trajectory_reviews
        if str(item.get("review_id")) in required_trajectory_ids and item.get("decision") in {"accept", "revise"}
    }
    flagged_required = {
        str(item["review_id"])
        for item in trajectory_reviews
        if str(item.get("review_id")) in required_trajectory_ids and item.get("decision") == "flag"
    }
    all_task_labels = expected_task_ids.issubset(accepted_task_reviews)
    unsafe_coverage = len(reviewed_required) / len(required_trajectory_ids) if required_trajectory_ids else 1.0
    minimum_review_records = 100
    enough_review_records = len(complete) >= minimum_review_records
    human_gate_complete = (
        all_task_labels
        and not flagged_required
        and unsafe_coverage == 1.0
        and enough_review_records
    )
    return {
        "schema_version": "agent-safety.review-summary.v1",
        "reviewed_records": len(complete),
        "task_label_records": len(task_reviews),
        "accepted_task_label_records": len(expected_task_ids & accepted_task_reviews),
        "trajectory_review_records": len(trajectory_reviews),
        "minimum_review_records": minimum_review_records,
        "enough_review_records": enough_review_records,
        "human_gate_100_records": human_gate_complete,
        "human_gate_complete": human_gate_complete,
        "all_task_labels_reviewed": all_task_labels,
        "unsafe_trajectories_total": len(required_trajectory_ids),
        "unsafe_trajectories_reviewed": len(reviewed_required),
        "unsafe_trajectories_flagged": len(flagged_required),
        "unsafe_review_coverage": unsafe_coverage,
        "task_set_hash": task_set_hash,
        "trajectory_set_hash": trajectory_set_hash,
        "reviewer_ids": sorted({str(item.get("reviewer")) for item in complete}),
    }


def validate_review_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_task_ids: set[str],
    known_trajectory_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate completed JSONL review rows before using them as human labels."""
    complete: list[dict[str, Any]] = []
    seen: set[str] = set()
    required_label_fields = {"expected_terminals", "must_halt", "benign"}
    for index, row in enumerate(records, 1):
        if not isinstance(row, dict):
            raise ValueError(f"review record {index} is not an object")
        review_id = str(row.get("review_id", "")).strip()
        if not review_id or review_id in seen:
            raise ValueError(f"review record {index} has a missing or duplicate review_id")
        seen.add(review_id)
        decision = str(row.get("decision", "pending"))
        if decision == "pending":
            continue
        if decision not in {"accept", "revise", "flag"}:
            raise ValueError(f"review {review_id} has an invalid decision")
        task_id = str(row.get("task_id", ""))
        if task_id not in expected_task_ids:
            raise ValueError(f"review {review_id} has an unknown task id")
        kind = str(row.get("kind", ""))
        if kind == "task_label":
            if review_id != f"task:{task_id}":
                raise ValueError(f"task review {review_id} has a mismatched review id")
            if decision == "revise":
                reviewed = row.get("reviewed_label")
                if not isinstance(reviewed, dict) or not required_label_fields.issubset(reviewed):
                    raise ValueError(f"revised task label {task_id} is incomplete")
        elif kind == "trajectory_review":
            if review_id not in known_trajectory_ids:
                raise ValueError(f"trajectory review {review_id} is not present in the run")
            if not isinstance(row.get("suggested"), dict):
                raise ValueError(f"trajectory review {review_id} has no evidence payload")
        else:
            raise ValueError(f"review {review_id} has an invalid kind")
        reviewer = str(row.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError(f"review {review_id} has no reviewer id")
        reviewed_at = str(row.get("reviewed_at_utc", "")).strip()
        try:
            reviewed_dt = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"review {review_id} has an invalid timestamp") from exc
        if reviewed_dt.tzinfo is None:
            raise ValueError(f"review {review_id} timestamp must include a timezone")
        complete.append(row)
    return complete


def trajectory_review_required(trajectory: dict[str, Any]) -> bool:
    return bool(
        trajectory.get("unsafe_executed") or trajectory.get("unsafe_attempt") or trajectory.get("permission_violation")
    )


def trajectory_review_id(trajectory: dict[str, Any]) -> str:
    return f"trajectory:{trajectory['model']}:{trajectory['policy_mode']}:{trajectory['task_id']}"


def required_trajectory_review_ids(
    trajectories: Iterable[dict[str, Any]],
) -> set[str]:
    return {trajectory_review_id(item) for item in trajectories if trajectory_review_required(item)}


def trajectory_set_hash(trajectories: Iterable[dict[str, Any]]) -> str:
    rows = list(trajectories)

    def immutable_run_record(item: dict[str, Any]) -> dict[str, Any]:
        # Human relabeling may change task_success and label-dependent stop
        # fields. Evidence identity must remain stable across that rescore.
        keys = (
            "task_id",
            "category",
            "model",
            "policy_mode",
            "provider",
            "task_hash",
            "config_hash",
            "terminal",
            "terminal_reason",
            "events",
            "tool_calls",
            "unsafe_executed",
            "unsafe_attempt",
            "permission_violation",
            "policy_denials",
            "model_errors",
            "parse_errors",
            "tool_errors",
            "latency_ms",
            "token_usage",
            "environment",
        )
        return {key: item.get(key) for key in keys}

    return sha256_text(
        canonical_json(
            [
                {
                    "task_id": item.get("task_id"),
                    "trajectory_hash": sha256_text(canonical_json(immutable_run_record(item))),
                }
                for item in sorted(
                    rows,
                    key=lambda row: (
                        str(row.get("model")),
                        str(row.get("policy_mode")),
                        str(row.get("task_id")),
                    ),
                )
            ]
        )
    )


def task_labels_from_reviews(
    reviews: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if review.get("kind") != "task_label":
            continue
        if review.get("decision") not in {"accept", "revise"}:
            continue
        task_id = str(review["task_id"])
        reviewed = review.get("reviewed_label")
        if isinstance(reviewed, dict):
            payload = reviewed
        elif review.get("decision") == "accept":
            payload = review.get("suggested", {})
        else:
            raise ValueError(f"revised task label {task_id} must include a reviewed_label object")
        labels[task_id] = {
            "id": task_id,
            "expected_terminals": [str(item) for item in payload.get("expected_terminals", [])],
            "must_halt": bool(payload.get("must_halt")),
            "benign": bool(payload.get("benign")),
            "required_tools": [str(item) for item in payload.get("required_tools", [])],
            "reviewer": str(review.get("reviewer", "")),
            "reviewed_at_utc": str(review.get("reviewed_at_utc", "")),
            "review_status": "human-reviewed",
        }
    return labels


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _review_html(queue: dict[str, Any]) -> str:
    data = json.dumps(queue, ensure_ascii=False).replace("</", "<\\/")
    escaped_instructions = html.escape(str(queue["instructions"]))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent 安全人工复核</title>
  <style>
    :root {{
      --ink: #101418; --paper: #f4efe6; --panel: #fffdf8; --line: #d7cbbc;
      --accent: #b23a2b; --safe: #196b52; --warn: #9a5b13;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background:
      radial-gradient(circle at 8% 10%, #e7d5bd 0, transparent 30%),
      linear-gradient(135deg, #f8f3e9, var(--paper));
      color: var(--ink); font: 16px/1.5 "IBM Plex Sans", "Aptos", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 5; background: rgba(244,239,230,.94);
      backdrop-filter: blur(10px); border-bottom: 2px solid var(--ink); padding: 18px 24px; }}
    header h1 {{ margin: 0; font: 700 30px/1.1 Georgia, serif; letter-spacing: -.02em; }}
    .meta {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; color: #514a42; }}
    .layout {{ max-width: 1180px; margin: 26px auto 80px; padding: 0 20px; display: grid;
      grid-template-columns: minmax(0, 1fr) 310px; gap: 22px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
      box-shadow: 0 14px 36px rgba(50,35,20,.08); padding: 24px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: 12px;
      color: var(--accent); font-weight: 800; }}
    h2 {{ margin: 8px 0 12px; font: 700 28px/1.2 Georgia, serif; }}
    .goal {{ white-space: pre-wrap; background: #f7f1e8; border-left: 4px solid var(--warn);
      padding: 14px; border-radius: 8px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #191d20; color: #f7f2e9;
      border-radius: 12px; padding: 16px; max-height: 350px; overflow: auto; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0; }}
    button {{ border: 1px solid var(--ink); background: white; padding: 10px 14px;
      border-radius: 999px; cursor: pointer; font-weight: 700; }}
    button:hover {{ transform: translateY(-1px); }}
    button.accept {{ background: var(--safe); color: white; }}
    button.flag {{ background: var(--accent); color: white; }}
    textarea {{ width: 100%; min-height: 90px; padding: 12px; border: 1px solid var(--line);
      border-radius: 10px; font: inherit; background: white; }}
    .side {{ position: sticky; top: 130px; align-self: start; }}
    .progress {{ height: 12px; background: #e5dacb; border-radius: 999px; overflow: hidden; }}
    .progress > div {{ height: 100%; width: 0; background: var(--safe); transition: width .2s; }}
    label {{ display: block; font-weight: 700; margin: 12px 0 6px; }}
    input {{ width: 100%; padding: 10px; border: 1px solid var(--line); border-radius: 8px; }}
    .download {{ width: 100%; margin-top: 12px; background: var(--ink); color: white; }}
    .small {{ font-size: 13px; color: #655b50; }}
    .pill {{ display: inline-block; padding: 3px 9px; border-radius: 999px;
      background: #eee5d8; font-size: 13px; font-weight: 700; }}
    @media (max-width: 850px) {{ .layout {{ grid-template-columns: 1fr; }} .side {{ position: static; }} }}
  </style>
</head>
<body>
<header>
  <h1>Agent 安全人工复核</h1>
  <div class="meta">
    <span>{escaped_instructions}</span>
    <span>仅在本机运行 · 不上传数据</span>
  </div>
</header>
<main class="layout">
  <section class="card" id="review-card"></section>
  <aside class="card side">
    <div class="eyebrow">复核进度</div>
    <h2 id="counter">0 / {len(queue["items"])}</h2>
    <div class="progress"><div id="bar"></div></div>
    <label for="reviewer">复核人标识</label>
    <input id="reviewer" placeholder="例如：liuzewen">
    <label for="notes">复核备注</label>
    <textarea id="notes" placeholder="可选：记录本条判断依据或疑问"></textarea>
    <button class="download" id="download">导出人工复核结果</button>
    <p class="small">快捷键：A 接受，F 标记问题，J 下一条，K 上一条。导出记录包含
    UTC 时间戳。页面不会自动证明标签正确，最终判断由复核人负责。</p>
  </aside>
</main>
<script>
const queue = {data};
const storageKey = "agent-safety-review-v1";
let state = JSON.parse(localStorage.getItem(storageKey) || "null");
if (!state || state.items.length !== queue.items.length) state = {{index: 0, items: queue.items}};
const card = document.getElementById("review-card");
const counter = document.getElementById("counter");
const bar = document.getElementById("bar");
const notes = document.getElementById("notes");
const reviewer = document.getElementById("reviewer");

function reviewed() {{ return state.items.filter(x => x.decision !== "pending").length; }}
function save() {{
  localStorage.setItem(storageKey, JSON.stringify(state));
  notes.value = state.items[state.index].notes || "";
  reviewer.value = state.items[state.index].reviewer || reviewer.value || "";
  counter.textContent = `${{reviewed()}} / ${{state.items.length}}`;
  bar.style.width = `${{100 * reviewed() / state.items.length}}%`;
}}
function render() {{
  const item = state.items[state.index];
  const suggested = JSON.stringify(item.suggested, null, 2);
  card.innerHTML = `
    <div class="eyebrow">第 ${{state.index + 1}} / ${{state.items.length}} 条 · ${{item.kind}}</div>
    <h2>${{escapeHtml(item.title)}}</h2>
    <span class="pill">${{escapeHtml(item.category)}} · ${{escapeHtml(item.task_id)}}</span>
    <p class="goal">${{escapeHtml(item.goal)}}</p>
    <h3>系统建议与证据</h3>
    <pre>${{escapeHtml(suggested)}}</pre>
    <div class="controls">
      <button class="accept" onclick="decide('accept')">接受建议 [A]</button>
      <button class="flag" onclick="decide('flag')">标记问题 [F]</button>
      <button onclick="move(-1)">上一条 [K]</button>
      <button onclick="move(1)">下一条 [J]</button>
    </div>`;
  save();
}}
function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
}}
function decide(kind) {{
  const item = state.items[state.index];
  if (!reviewer.value.trim()) {{ alert("请先填写复核人标识。"); return; }}
  item.decision = kind;
  item.reviewer = reviewer.value.trim();
  item.reviewed_at_utc = new Date().toISOString();
  item.notes = notes.value.trim();
  if (kind === "accept") item.reviewed_label = item.suggested;
  save();
  if (state.index < state.items.length - 1) {{ state.index += 1; render(); }}
}}
function move(delta) {{
  const item = state.items[state.index];
  item.notes = notes.value.trim();
  item.reviewer = reviewer.value.trim();
  state.index = Math.max(0, Math.min(state.items.length - 1, state.index + delta));
  render();
}}
document.getElementById("download").onclick = () => {{
  const item = state.items[state.index];
  item.notes = notes.value.trim();
  item.reviewer = reviewer.value.trim();
  const payload = {{
    schema_version: "agent-safety.review-export.v1",
    task_set_hash: queue.task_set_hash,
    trajectory_set_hash: queue.trajectory_set_hash,
    exported_at_utc: new Date().toISOString(),
    items: state.items
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "reviews-export.json"; a.click();
}};
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key.toLowerCase() === "a") decide("accept");
  if (e.key.toLowerCase() === "f") decide("flag");
  if (e.key.toLowerCase() === "j") move(1);
  if (e.key.toLowerCase() === "k") move(-1);
}});
render();
</script>
</body>
</html>
"""
