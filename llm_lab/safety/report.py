"""JSON, Markdown, and standalone HTML reports for Agent Safety runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from llm_lab.safety.models import canonical_json, sha256_text
from llm_lab.safety.runner import (
    SafetyRunConfig,
    Trajectory,
    aggregate_trajectories,
    answer_hash,
)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if percent:
            return f"{float(value):.2%}"
        return f"{float(value):.2f}"
    return str(value)


def run_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["metrics"]
    label = report["label_source"]
    gate = (
        "human-reviewed labels" if report["human_review_complete"] else "DRAFT labels only; human review is incomplete"
    )
    lines = [
        "# Agent Safety Local Evaluation",
        "",
        "> **Local synthetic experiment. This is not a production security assessment.**",
        "> **本地安全评测，不是生产安全系统。**",
        "",
        f"- Model: `{report['model']}`",
        f"- Provider: `{report['provider']}`",
        f"- Policy mode: **{report['policy_mode']}**",
        f"- Task set SHA-256: `{report['task_set_hash']}`",
        f"- Config SHA-256: `{report['config_hash']}`",
        f"- Trajectory set SHA-256: `{report['trajectory_set_hash']}`",
        f"- Label source: `{label}` ({gate})",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Numerator / denominator |",
        "|---|---:|---:|",
        (
            f"| Unsafe executed-action rate | {metrics['unsafe_action_rate']:.2%} "
            f"| {counts['unsafe_executed']} / {counts['tasks']} |"
        ),
        (
            f"| Permission-violation rate | {metrics['permission_violation_rate']:.2%} "
            f"| {counts['permission_violation']} / {counts['tasks']} |"
        ),
        (f"| Task success rate | {metrics['task_success_rate']:.2%} | {counts['task_success']} / {counts['tasks']} |"),
        (
            f"| Over-refusal rate | {_metric(metrics['over_refusal_rate'], percent=True)} "
            f"| {counts['over_refusal']} / {counts['benign_controls']} |"
        ),
        (
            f"| Stop precision | {_metric(metrics['stop_precision'], percent=True)} "
            f"| {counts['stop_correct']} / {counts['predicted_halt']} |"
        ),
        (
            f"| Stop recall | {_metric(metrics['stop_recall'], percent=True)} "
            f"| {counts['stop_correct']} / {counts['must_halt']} |"
        ),
        (
            f"| Average latency | {metrics['average_latency_ms']:.2f} ms "
            f"| p50={metrics['p50_latency_ms']:.2f}, p95={metrics['p95_latency_ms']:.2f} |"
        ),
        (f"| Throughput | {metrics['tasks_per_second']:.3f} task/s | wall={metrics['wall_seconds']:.3f}s |"),
        "",
        "## Failure Surface",
        "",
        f"- Unsafe actions attempted but blocked: {counts['unsafe_attempted']}",
        f"- Policy denials: {counts['policy_denials']}",
        f"- Model failures: {counts['model_errors']}",
        f"- Action-format failures: {counts['parse_errors']}",
        f"- Mock tool failures: {counts['tool_errors']}",
        "",
        "## Category Breakdown",
        "",
        "| Category | Tasks | Success | Unsafe executed | Unsafe attempted |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in report["category_breakdown"].items():
        lines.append(
            f"| `{category}` | {values['total']} | {values['task_success']} | "
            f"{values['unsafe_executed']} | {values['unsafe_attempted']} |"
        )
    lines.extend(["", "## Boundaries", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "- Metrics describe this frozen synthetic task set and selected local model only.",
            "- The report does not establish production SLOs, multi-GPU serving, or real-world safety.",
            "",
        ]
    )
    return "\n".join(lines)


_HTML_STYLE = """
:root{--ink:#17201d;--paper:#f5f1e7;--card:#fffdf7;--rule:#c9c0ad;--red:#9d2f23;--green:#176142;--ochre:#a86f12}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#eee7d8,#f8f4e9 55%,#e6ede8);color:var(--ink);font:15px/1.55 Georgia,"Noto Serif SC",serif}
main{max-width:1120px;margin:32px auto;padding:0 22px 48px}.hero{background:var(--ink);color:var(--paper);padding:32px;border-radius:18px 18px 4px 4px;position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;right:-40px;top:-60px;width:260px;height:260px;border:34px solid #6b8d7b;border-radius:50%;opacity:.55}
h1{margin:0 0 8px;font-size:clamp(28px,5vw,54px);line-height:1.02;letter-spacing:-.03em}h2{margin-top:32px}
.warn{margin:20px 0;padding:14px 18px;border-left:6px solid var(--red);background:#fff1e8;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:20px}
.stat{background:var(--card);border:1px solid var(--rule);padding:16px;border-radius:8px;box-shadow:3px 3px 0 #d8d0be}.stat b{display:block;font-size:28px;color:var(--red)}
.stat.ok b{color:var(--green)}.stat.caution b{color:var(--ochre)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--rule)}th,td{padding:9px 11px;border-bottom:1px solid #ddd4c3;text-align:left}th{background:#ece5d7}
code{font:13px/1.45 "IBM Plex Mono","Noto Sans Mono",monospace;background:#eee8db;padding:2px 5px;border-radius:3px}.meta{background:var(--card);padding:18px;border:1px solid var(--rule);border-radius:8px}
ul{padding-left:22px}.foot{margin-top:36px;color:#5d655f;font-size:13px}@media(max-width:600px){main{margin:12px auto}.hero{padding:22px}th,td{font-size:13px;padding:7px}}
"""


def run_html(report: dict[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["metrics"]
    category_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(category)}</code></td>"
        f"<td>{values['total']}</td><td>{values['task_success']}</td>"
        f"<td>{values['unsafe_executed']}</td><td>{values['unsafe_attempted']}</td>"
        "</tr>"
        for category, values in report["category_breakdown"].items()
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report["limitations"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Safety Local Evaluation</title><style>{_HTML_STYLE}</style></head>
<body><main>
<section class="hero"><h1>Agent Safety<br>Local Evaluation</h1>
<p>{html.escape(report["model"])} · policy <strong>{html.escape(report["policy_mode"])}</strong></p></section>
<div class="warn">Local synthetic experiment. This is not a production security assessment.<br>本地安全评测，不是生产安全系统。</div>
<div class="meta">
<p><strong>Task set:</strong> <code>{report["task_set_hash"]}</code></p>
<p><strong>Config:</strong> <code>{report["config_hash"]}</code></p>
<p><strong>Trajectory set:</strong> <code>{report["trajectory_set_hash"]}</code></p>
<p><strong>Label source:</strong> {html.escape(str(report["label_source"]))} · human review complete: <strong>{_metric(report["human_review_complete"])}</strong></p>
</div>
<h2>Primary Metrics</h2><div class="grid">
<div class="stat"><b>{metrics["unsafe_action_rate"]:.2%}</b>unsafe action rate<br>{counts["unsafe_executed"]}/{counts["tasks"]}</div>
<div class="stat"><b>{metrics["permission_violation_rate"]:.2%}</b>permission violation<br>{counts["permission_violation"]}/{counts["tasks"]}</div>
<div class="stat ok"><b>{metrics["task_success_rate"]:.2%}</b>task success<br>{counts["task_success"]}/{counts["tasks"]}</div>
<div class="stat caution"><b>{_metric(metrics["over_refusal_rate"], percent=True)}</b>over-refusal<br>{counts["over_refusal"]}/{counts["benign_controls"]}</div>
<div class="stat"><b>{_metric(metrics["stop_precision"], percent=True)}</b>stop precision</div>
<div class="stat"><b>{_metric(metrics["stop_recall"], percent=True)}</b>stop recall</div>
<div class="stat"><b>{metrics["p50_latency_ms"]:.0f} ms</b>p50 latency</div>
<div class="stat"><b>{metrics["p95_latency_ms"]:.0f} ms</b>p95 latency</div>
</div>
<h2>Failure Surface</h2>
<table><tr><th>Signal</th><th>Count</th></tr>
<tr><td>Unsafe actions attempted but blocked</td><td>{counts["unsafe_attempted"]}</td></tr>
<tr><td>Policy denials</td><td>{counts["policy_denials"]}</td></tr>
<tr><td>Model failures</td><td>{counts["model_errors"]}</td></tr>
<tr><td>Action-format failures</td><td>{counts["parse_errors"]}</td></tr>
<tr><td>Mock tool failures</td><td>{counts["tool_errors"]}</td></tr></table>
<h2>Category Breakdown</h2>
<table><tr><th>Category</th><th>Tasks</th><th>Success</th><th>Unsafe executed</th><th>Unsafe attempted</th></tr>{category_rows}</table>
<h2>Boundaries</h2><ul>{limitations}</ul>
<p class="foot">All tools are mock-only. Metrics describe this frozen local task set and selected model only; they are not production evidence.</p>
</main></body></html>
"""


def write_run_outputs(
    output_dir: Path,
    *,
    tasks: list[Any],
    trajectories: list[Trajectory],
    config: SafetyRunConfig,
    wall_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = aggregate_trajectories(
        trajectories,
        tasks=tasks,
        config=config,
        wall_seconds=wall_seconds,
    )
    report["answer_hash"] = answer_hash(trajectories)
    meta = {
        "schema_version": "agent-safety.run.v1",
        "config": config.to_dict(),
        "task_count": len(tasks),
        "task_set_hash": report["task_set_hash"],
        "trajectory_set_hash": report["trajectory_set_hash"],
        "answer_hash": report["answer_hash"],
        "label_source": report["label_source"],
        "human_review_complete": report["human_review_complete"],
        "mock_only": True,
    }
    trajectory_payloads = [trajectory.to_dict() for trajectory in sorted(trajectories, key=lambda item: item.task_id)]
    (output_dir / "run-meta.json").write_text(_json_text(meta), encoding="utf-8")
    (output_dir / "trajectories.jsonl").write_text(
        "".join(canonical_json(payload) + "\n" for payload in trajectory_payloads),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(_json_text(report), encoding="utf-8")
    (output_dir / "report.md").write_text(run_markdown(report), encoding="utf-8")
    (output_dir / "report.html").write_text(run_html(report), encoding="utf-8")
    report["output_dir"] = str(output_dir)
    return report


def matrix_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agent Safety Policy Matrix",
        "",
        "> **Local synthetic experiment. This is not a production security assessment.**",
        "> **本地安全评测，不是生产安全系统。**",
        "",
        f"- Task set SHA-256: `{report['task_set_hash']}`",
        f"- Trajectories: {report['counts']['runs']} runs · {report['counts']['trajectories']} trajectories",
        f"- Human review complete: **{_metric(report['human_review_complete'])}**",
        "",
        "| Model | Policy | Unsafe rate | Permission violation | Task success | Over-refusal | Stop P/R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['model']}` | `{row['policy_mode']}` | "
            f"{metrics['unsafe_action_rate']:.2%} | "
            f"{metrics['permission_violation_rate']:.2%} | "
            f"{metrics['task_success_rate']:.2%} | "
            f"{_metric(metrics['over_refusal_rate'], percent=True)} | "
            f"{_metric(metrics['stop_precision'], percent=True)} / "
            f"{_metric(metrics['stop_recall'], percent=True)} |"
        )
    lines.extend(["", "## Policy Effect", ""])
    if report["policy_effects"]:
        lines.append("| Model | Unsafe-rate Δ (on − off) | Success Δ (on − off) | Over-refusal Δ (on − off) |")
        lines.append("|---|---:|---:|---:|")
        for effect in report["policy_effects"]:
            lines.append(
                f"| `{effect['model']}` | {effect['unsafe_rate_delta']:+.2%} | "
                f"{effect['success_rate_delta']:+.2%} | "
                f"{effect['over_refusal_rate_delta']:+.2%} |"
            )
    else:
        lines.append("Policy-effect deltas require paired off/on runs for the same model.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- All dangerous actions ran only against an in-memory mock environment.",
            "- Draft labels are not human-reviewed evidence.",
            "- CPU latency and single-process concurrency are not GPU serving SLOs.",
            "",
        ]
    )
    return "\n".join(lines)


def matrix_html(report: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(row['model'])}</code></td>"
        f"<td><strong>{html.escape(row['policy_mode'])}</strong></td>"
        f"<td>{row['metrics']['unsafe_action_rate']:.2%}</td>"
        f"<td>{row['metrics']['permission_violation_rate']:.2%}</td>"
        f"<td>{row['metrics']['task_success_rate']:.2%}</td>"
        f"<td>{_metric(row['metrics']['over_refusal_rate'], percent=True)}</td>"
        f"<td>{_metric(row['metrics']['stop_precision'], percent=True)} / "
        f"{_metric(row['metrics']['stop_recall'], percent=True)}</td>"
        "</tr>"
        for row in report["runs"]
    )
    effects = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['model'])}</code></td>"
        f"<td>{item['unsafe_rate_delta']:+.2%}</td>"
        f"<td>{item['success_rate_delta']:+.2%}</td>"
        f"<td>{item['over_refusal_rate_delta']:+.2%}</td>"
        "</tr>"
        for item in report["policy_effects"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Safety Policy Matrix</title><style>{_HTML_STYLE}</style></head><body><main>
<section class="hero"><h1>Policy Matrix<br>Agent Safety</h1><p>Policy-off versus policy-on in a mock-only sandbox</p></section>
<div class="warn">Local synthetic experiment. This is not a production security assessment.<br>本地安全评测，不是生产安全系统。</div>
<div class="meta"><p><strong>Task set:</strong> <code>{report["task_set_hash"]}</code></p>
<p><strong>Human review complete:</strong> {_metric(report["human_review_complete"])}</p></div>
<h2>Runs</h2><table><tr><th>Model</th><th>Policy</th><th>Unsafe rate</th><th>Permission violation</th><th>Task success</th><th>Over-refusal</th><th>Stop P / R</th></tr>{rows}</table>
<h2>Policy Effect</h2><table><tr><th>Model</th><th>Unsafe-rate Δ</th><th>Success Δ</th><th>Over-refusal Δ</th></tr>{effects or '<tr><td colspan="4">Paired off/on runs are required.</td></tr>'}</table>
<p class="foot">All dangerous actions were mock-only. Draft labels are not human-reviewed evidence. CPU timings are not production SLOs.</p>
</main></body></html>
"""


def write_matrix_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix.json").write_text(_json_text(report), encoding="utf-8")
    (output_dir / "matrix.md").write_text(matrix_markdown(report), encoding="utf-8")
    (output_dir / "matrix.html").write_text(matrix_html(report), encoding="utf-8")
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_text((output_dir / name).read_text(encoding='utf-8'))}  {name}\n"
            for name in ("matrix.json", "matrix.md", "matrix.html")
        ),
        encoding="utf-8",
    )
