"""Combined reports for the Agent Safety evaluation suite."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

from llm_lab.evalops import write_json

from .output_guard import claim_output

NON_PRODUCTION_EN = "Local safety evaluation, not a production security system."
NON_PRODUCTION_ZH = "本地安全评测，不是生产安全系统。"

METRIC_KEYS = (
    "unsafe_action_rate",
    "permission_violation_rate",
    "stop_precision",
    "stop_recall",
    "over_refusal_rate",
    "task_success_rate",
    "recovery_success_rate",
)

ARM_DESCRIPTIONS = {
    "off": "no policy prompt; runtime gate disabled",
    "prompt": "policy prompt only; runtime gate disabled",
    "gate": "no policy prompt; runtime gate enabled",
    "on": "policy prompt and runtime gate enabled",
}


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}"


def build_combined_report(
    *,
    runs: list[dict[str, Any]],
    task_count: int,
    task_hash: str,
    category_counts: dict[str, int],
    review_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        grouped[(str(run["model"]), str(run["policy_mode"]))] = run

    comparisons: list[dict[str, Any]] = []
    for model in sorted({str(run["model"]) for run in runs}):
        off = grouped.get((model, "off"))
        if not off:
            continue
        for treatment_arm in ("prompt", "gate", "on"):
            treatment = grouped.get((model, treatment_arm))
            if not treatment:
                continue
            deltas: dict[str, float | None] = {}
            for key in METRIC_KEYS:
                before = off["metrics"].get(key)
                after = treatment["metrics"].get(key)
                deltas[key] = float(after) - float(before) if before is not None and after is not None else None
            comparisons.append(
                {
                    "model": model,
                    "baseline_arm": "off",
                    "treatment_arm": treatment_arm,
                    "baseline_config_hash": off["config_hash"],
                    "treatment_config_hash": treatment["config_hash"],
                    "metric_deltas": deltas,
                }
            )

    return {
        "schema_version": "agent-safety.suite-report.v2",
        "dataset": {
            "task_count": task_count,
            "task_set_hash": task_hash,
            "category_counts": category_counts,
        },
        "runs": runs,
        "policy_comparisons": comparisons,
        "arms": ARM_DESCRIPTIONS,
        "human_review": review_summary
        or {
            "reviewed_records": 0,
            "minimum_review_records": 100,
            "enough_review_records": False,
            "human_gate_100_records": False,
            "human_gate_complete": False,
            "all_task_labels_reviewed": False,
            "unsafe_trajectories_total": 0,
            "unsafe_trajectories_reviewed": 0,
            "unsafe_trajectories_flagged": 0,
            "unsafe_review_coverage": 0.0,
            "reviewer_ids": [],
        },
        "overall_limits": [
            NON_PRODUCTION_EN,
            NON_PRODUCTION_ZH,
            "All tools are mock-only; no real filesystem, shell, email or network action occurs.",
            "Assistant-authored draft labels are not human labels and must not be counted as such.",
            "Small local CPU models are not representative of frontier or production agents.",
            "No claim is made about benchmark leadership, regulatory compliance or field safety.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        "# Agent Safety Runtime Monitor",
        "",
        f"> **{NON_PRODUCTION_EN}**",
        f"> **{NON_PRODUCTION_ZH}**",
        "",
        f"- Frozen tasks: **{dataset['task_count']}**",
        f"- Task-set SHA-256: `{dataset['task_set_hash']}`",
        f"- Models: **{len({run['model'] for run in report['runs']})}**",
        f"- Trajectories: **{sum(run['counts']['tasks'] for run in report['runs'])}**",
        "",
        "## Results",
        "",
        "| Model | Policy | Unsafe action | Permission violation | Stop P | Stop R | "
        "Over-refusal | Task success | Recovery |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        lines.append(
            f"| `{run['model']}` | `{run['policy_mode']}` | "
            f"{_fmt_rate(metrics['unsafe_action_rate'])} | "
            f"{_fmt_rate(metrics['permission_violation_rate'])} | "
            f"{_fmt_rate(metrics['stop_precision'])} | "
            f"{_fmt_rate(metrics['stop_recall'])} | "
            f"{_fmt_rate(metrics['over_refusal_rate'])} | "
            f"{_fmt_rate(metrics['task_success_rate'])} | "
            f"{_fmt_rate(metrics['recovery_success_rate'])} |"
        )
    lines += [
        "",
        "## Intervention deltas versus no intervention",
        "",
        "| Model | Treatment | Unsafe action | Stop recall | Over-refusal | Task success |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for comparison in report["policy_comparisons"]:
        delta = comparison["metric_deltas"]
        lines.append(
            f"| `{comparison['model']}` | `{comparison['treatment_arm']}` | "
            f"{_fmt_delta(delta.get('unsafe_action_rate'))} | "
            f"{_fmt_delta(delta.get('stop_recall'))} | "
            f"{_fmt_delta(delta.get('over_refusal_rate'))} | "
            f"{_fmt_delta(delta.get('task_success_rate'))} |"
        )
    review = report["human_review"]
    lines += [
        "",
        "## Human-review gate",
        "",
        f"- Reviewed records: **{review.get('reviewed_records', 0)}**",
        f"- Minimum acceptance records: **{review.get('minimum_review_records', 100)}**",
        f"- Minimum met: **{review.get('enough_review_records', False)}**",
        f"- Complete human gate: **{review.get('human_gate_complete', False)}**",
        f"- All {dataset['task_count']} task labels reviewed: **{review.get('all_task_labels_reviewed', False)}**",
        f"- Unsafe trajectory coverage: **{review.get('unsafe_trajectories_reviewed', 0)} / "
        f"{review.get('unsafe_trajectories_total', 0)}**",
        f"- Flagged required trajectories: **{review.get('unsafe_trajectories_flagged', 0)}**",
        "",
    ]
    if not review.get("human_gate_complete", False):
        lines += [
            "The human-review acceptance criterion is **not met yet**. Open "
            "`review/review.html`, review the packet, export it, and ingest the export.",
            "",
        ]
    lines += ["## Category breakdown", ""]
    for run in report["runs"]:
        lines += [
            f"### `{run['model']}` · policy-{run['policy_mode']}",
            "",
            "| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for category, row in sorted(run["category_breakdown"].items()):
            lines.append(
                f"| `{category}` | {row['total']} | {row['task_success']} | "
                f"{row['unsafe_executed']} | {row['unsafe_attempted']} | {row['tool_errors']} |"
            )
        lines.append("")
    lines += ["## Failure taxonomy", ""]
    for run in report["runs"]:
        lines.append(
            f"- `{run['model']}` / policy-{run['policy_mode']}: "
            f"model={json.dumps(run.get('failure_counts', {}), sort_keys=True)}, "
            f"tool={json.dumps(run.get('tool_failure_counts', {}), sort_keys=True)}"
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "./scripts/rerun.sh",
        "python3 verify_evidence.py",
        "```",
        "",
        "## Non-claims",
        "",
    ]
    lines.extend(f"- {item}" for item in report["overall_limits"])
    lines.append("")
    return "\n".join(lines)


def _rate_bar(value: Any, *, tone: str = "risk") -> str:
    if value is None:
        return '<div class="bar empty"><span>n/a</span></div>'
    numeric = max(0.0, min(1.0, float(value)))
    return f'<div class="bar {tone}"><span style="width:{numeric * 100:.1f}%"></span><b>{numeric:.1%}</b></div>'


def html_report(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    models = sorted({run["model"] for run in report["runs"]})
    cards: list[str] = []
    for run in report["runs"]:
        metrics = run["metrics"]
        rows = []
        for category, row in sorted(run["category_breakdown"].items()):
            rows.append(
                "<tr>"
                f"<td>{html.escape(category)}</td>"
                f"<td>{row['total']}</td>"
                f"<td>{row['task_success']}</td>"
                f"<td>{row['unsafe_executed']}</td>"
                f"<td>{row['unsafe_attempted']}</td>"
                f"<td>{row['tool_errors']}</td>"
                "</tr>"
            )
        cards.append(
            f"""<article class="run-card">
              <div class="run-head">
                <div>
                  <div class="eyebrow">trajectory batch</div>
                  <h3>{html.escape(run["model"])}</h3>
                </div>
                <span class="policy policy-{run["policy_mode"]}">policy-{run["policy_mode"]}</span>
              </div>
              <div class="metric-grid">
                <div><span>Unsafe action</span>{_rate_bar(metrics["unsafe_action_rate"])}</div>
                <div><span>Permission violation</span>{_rate_bar(metrics["permission_violation_rate"])}</div>
                <div><span>Stop precision</span>{_rate_bar(metrics["stop_precision"], tone="safe")}</div>
                <div><span>Stop recall</span>{_rate_bar(metrics["stop_recall"], tone="safe")}</div>
                <div><span>Over-refusal</span>{_rate_bar(metrics["over_refusal_rate"])}</div>
                <div><span>Task success</span>{_rate_bar(metrics["task_success_rate"], tone="safe")}</div>
                <div><span>Recovery success</span>{_rate_bar(metrics["recovery_success_rate"], tone="safe")}</div>
              </div>
              <table>
                <thead><tr><th>Category</th><th>n</th><th>ok</th>
                <th>unsafe</th><th>attempt</th><th>tool err</th></tr></thead>
                <tbody>{"".join(rows)}</tbody>
              </table>
            </article>"""
        )

    comparison_rows = []
    for item in report["policy_comparisons"]:
        delta = item["metric_deltas"]
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(item['model'])}</td>"
            f"<td><code>{html.escape(item['treatment_arm'])}</code></td>"
            f"<td>{_fmt_delta(delta.get('unsafe_action_rate'))}</td>"
            f"<td>{_fmt_delta(delta.get('stop_recall'))}</td>"
            f"<td>{_fmt_delta(delta.get('over_refusal_rate'))}</td>"
            f"<td>{_fmt_delta(delta.get('task_success_rate'))}</td>"
            "</tr>"
        )

    review = report["human_review"]
    review_gate = bool(review.get("human_gate_complete"))
    review_class = "gate-pass" if review_gate else "gate-hold"
    review_text = "PASS" if review_gate else "HOLD"
    category_chips = "".join(
        f"<span>{html.escape(key)} <b>{value}</b></span>" for key, value in sorted(dataset["category_counts"].items())
    )
    failures = []
    for run in report["runs"]:
        failures.append(
            f"<li><code>{html.escape(run['model'])}</code> · policy-{run['policy_mode']}"
            f" — model <code>{html.escape(json.dumps(run.get('failure_counts', {}), sort_keys=True))}</code>,"
            f" tool <code>{html.escape(json.dumps(run.get('tool_failure_counts', {}), sort_keys=True))}</code></li>"
        )
    limit_items = "".join(f"<li>{html.escape(item)}</li>" for item in report["overall_limits"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Safety Runtime Monitor</title>
  <style>
    :root {{
      --ink:#10161a; --paper:#ede6da; --card:#fffdf7; --line:#d1c2ad;
      --red:#b43b2f; --green:#14674f; --amber:#a15f14;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:
      radial-gradient(circle at 85% 5%, #e9d2b4 0, transparent 25%),
      linear-gradient(160deg,#f5efe5,var(--paper) 55%,#e6dccd);
      font:16px/1.5 "Aptos","IBM Plex Sans",sans-serif; }}
    .hero {{ padding:56px max(24px,7vw) 30px; border-bottom:2px solid var(--ink); }}
    .eyebrow {{ color:var(--red); font-size:12px; font-weight:900; letter-spacing:.16em;
      text-transform:uppercase; }}
    h1 {{ margin:12px 0 10px; max-width:900px; font:800 clamp(40px,7vw,84px)/.95 Georgia,serif;
      letter-spacing:-.05em; }}
    .lede {{ max-width:900px; font-size:20px; color:#3f4649; }}
    .warning {{ display:inline-block; margin-top:10px; padding:8px 12px; border:2px solid var(--red);
      background:#fff7f1; font-weight:800; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:24px; }}
    .stats span,.category-chips span {{ border:1px solid var(--line); background:var(--card);
      border-radius:999px; padding:7px 12px; }}
    main {{ max-width:1380px; margin:0 auto; padding:36px 24px 80px; }}
    .top-grid {{ display:grid; grid-template-columns:1.4fr .6fr; gap:22px; }}
    .panel {{ background:var(--card); border:1px solid var(--line); border-radius:22px;
      padding:24px; box-shadow:0 18px 50px rgba(67,46,22,.08); }}
    .panel h2 {{ font:700 30px/1.1 Georgia,serif; margin:0 0 14px; }}
    .run-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; margin-top:24px; }}
    .run-card {{ background:var(--card); border:1px solid var(--line); border-radius:18px; padding:22px; }}
    .run-head {{ display:flex; justify-content:space-between; gap:16px; align-items:start; }}
    .run-head h3 {{ margin:5px 0 0; font:700 24px/1.15 Georgia,serif; overflow-wrap:anywhere; }}
    .policy {{ border-radius:999px; padding:5px 11px; font-weight:800; font-size:13px; }}
    .policy-on {{ background:#d8f0e6; color:var(--green); }}
    .policy-prompt {{ background:#f4e7c7; color:#76520d; }}
    .policy-gate {{ background:#d9e8f5; color:#174d78; }}
    .policy-off {{ background:#f4dcd7; color:var(--red); }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin:18px 0; }}
    .metric-grid > div {{ border-top:1px solid var(--line); padding-top:9px; }}
    .metric-grid > div > span {{ font-size:12px; font-weight:800; text-transform:uppercase;
      letter-spacing:.08em; color:#5c574f; }}
    .bar {{ position:relative; height:23px; margin-top:5px; background:#e6ddd0; border-radius:999px;
      overflow:hidden; }}
    .bar > span {{ position:absolute; inset:0 auto 0 0; background:var(--red); opacity:.78; }}
    .bar.safe > span {{ background:var(--green); }}
    .bar b {{ position:relative; z-index:1; padding-left:9px; font-size:13px; line-height:23px; }}
    .bar.empty {{ padding-left:9px; font-size:13px; line-height:23px; color:#786e62; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 5px; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    .gate {{ display:flex; align-items:center; justify-content:space-between; gap:18px;
      border-radius:16px; padding:18px; margin-top:18px; }}
    .gate-pass {{ background:#d8f0e6; }}
    .gate-hold {{ background:#f6ddd7; }}
    .gate strong {{ font:800 42px/.9 Georgia,serif; }}
    .delta-table {{ margin-top:20px; }}
    .note {{ border-left:5px solid var(--amber); background:#fff7e7; padding:14px; border-radius:0 10px 10px 0; }}
    ul {{ padding-left:22px; }}
    code {{ background:#eee6da; border-radius:4px; padding:1px 5px; overflow-wrap:anywhere; }}
    .footer {{ margin-top:26px; color:#5c574f; font-size:14px; }}
    @media (max-width:950px) {{
      .top-grid,.run-grid {{ grid-template-columns:1fr; }}
      h1 {{ font-size:52px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="eyebrow">EvalOps extension · mock-only runtime</div>
    <h1>Agent Safety Runtime Monitor</h1>
    <p class="lede">A frozen {dataset["task_count"]}-task safety suite separating prompt-only,
    runtime-gate-only and combined intervention effects across local Ollama models.</p>
    <div class="warning">{html.escape(NON_PRODUCTION_EN)}<br>{html.escape(NON_PRODUCTION_ZH)}</div>
    <div class="stats">
      <span><b>{dataset["task_count"]}</b> tasks</span>
      <span><b>{len(models)}</b> models</span>
      <span><b>{sum(run["counts"]["tasks"] for run in report["runs"])}</b> trajectories</span>
      <span><b>{len(report["runs"])}</b> policy runs</span>
      <span>SHA-256 <code>{html.escape(dataset["task_set_hash"][:16])}…</code></span>
    </div>
  </header>
  <main>
    <section class="top-grid">
      <div class="panel">
        <div class="eyebrow">Intervention ablation</div>
        <h2>Prompt and gate effects, separately</h2>
        <table class="delta-table">
          <thead><tr><th>Model</th><th>Treatment</th><th>Unsafe action</th>
          <th>Stop recall</th><th>Over-refusal</th><th>Task success</th></tr></thead>
          <tbody>{"".join(comparison_rows)}</tbody>
        </table>
        <p class="note">A negative unsafe-action delta is protective. Any drop in task success or
        stop recall is a monitor cost that must be inspected, not hidden.</p>
      </div>
      <div class="panel">
        <div class="eyebrow">Human review</div>
        <h2>Acceptance gate</h2>
        <div class="gate {review_class}">
          <div>
            <strong>{review_text}</strong>
            <div>{review.get("reviewed_records", 0)} reviewed records</div>
            <div>minimum {review.get("minimum_review_records", 100)}</div>
          </div>
          <div>{"All task labels and required unsafe trajectories reviewed" if review_gate else "Open review/review.html"}</div>
        </div>
        <p><b>All task labels reviewed:</b> {review.get("all_task_labels_reviewed", False)}</p>
        <p><b>Unsafe trajectory coverage:</b>
        {review.get("unsafe_trajectories_reviewed", 0)} /
        {review.get("unsafe_trajectories_total", 0)}
        ({review.get("unsafe_review_coverage", 0.0):.1%})</p>
        <p>The assistant-authored task labels are draft evidence. They are never relabeled as
        human review without a reviewer ID and UTC timestamp.</p>
      </div>
    </section>
    <div class="category-chips stats">{category_chips}</div>
    <section class="run-grid">{"".join(cards)}</section>
    <section class="panel" style="margin-top:24px">
      <div class="eyebrow">Failure taxonomy</div>
      <h2>Operational failures</h2>
      <ul>{"".join(failures)}</ul>
    </section>
    <section class="panel" style="margin-top:24px">
      <div class="eyebrow">Evidence boundary</div>
      <h2>What this report does not claim</h2>
      <ul>{limit_items}</ul>
      <p class="footer">Generated from frozen tasks, raw trajectory records and deterministic
      aggregation. Verify with <code>python3 verify_evidence.py</code>.</p>
    </section>
  </main>
</body>
</html>
"""


def write_reports(
    output_dir: str,
    report: dict[str, Any],
    *,
    markdown_name: str = "report.md",
    html_name: str = "report.html",
) -> None:
    path = claim_output(output_dir, "evidence", allow_existing_owned=True)
    write_json(path / "report.json", report)
    (path / markdown_name).write_text(markdown_report(report), encoding="utf-8")
    (path / html_name).write_text(html_report(report), encoding="utf-8")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval used by companion calibration reports."""
    if total <= 0:
        return {"low": 0.0, "high": 1.0}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def most_common(values: Iterable[str]) -> list[tuple[str, int]]:
    return Counter(values).most_common()
