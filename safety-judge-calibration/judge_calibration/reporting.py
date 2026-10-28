"""Render safety-judge calibration reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from llm_lab.evalops import write_json


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _ci(interval: dict[str, float]) -> str:
    return f"[{interval['low']:.1%}, {interval['high']:.1%}]"


def _fmt_float(value: Any, digits: int) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    rows = [
        "| Judge | Accuracy | 95% Wilson | FPR | FNR | Over-refusal | Under-refusal | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for judge, metrics in report["judges"].items():
        rows.append(
            f"| `{judge}` | {_pct(metrics['accuracy'])} | "
            f"{_ci(metrics['accuracy_wilson_95'])} | {_pct(metrics['fpr'])} | "
            f"{_pct(metrics['fnr'])} | {_pct(metrics['over_refusal_rate'])} | "
            f"{_pct(metrics['under_refusal_rate'])} | {_pct(metrics['error_rate'])} |"
        )
    calibration_rows = [
        "| Judge | Brier | ECE (10-bin) | Latency p50 | Latency p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for judge, metrics in report["judges"].items():
        calibration_rows.append(
            f"| `{judge}` | {_fmt_float(metrics['brier_score'], 4)} | "
            f"{_fmt_float(metrics['ece_10_bin'], 4)} | "
            f"{metrics['latency_ms']['p50']:.1f} ms | "
            f"{metrics['latency_ms']['p95']:.1f} ms |"
        )
    pair_rows = [
        "| Judge A | Judge B | n | Agreement | Cohen κ |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in report["pairwise"]:
        kappa = "n/a" if pair["cohen_kappa"] is None else f"{pair['cohen_kappa']:.3f}"
        pair_rows.append(
            f"| `{pair['left']}` | `{pair['right']}` | {pair['n']} | {_pct(pair['agreement'])} | {kappa} |"
        )
    hold_rows = ["| Condition | Pass | Evidence |", "|---|---|---|"]
    for condition in report["hold_conditions"]:
        hold_rows.append(
            f"| `{condition['key']}` | {condition['passed']} | `{html.escape(str(condition['evidence']))}` |"
        )
    return "\n".join(
        [
            "# Safety Judge Calibration",
            "",
            f"> Decision: **{report['decision']}**",
            "> **本地安全评测，不是生产安全系统。 / Local safety evaluation, not a production security system.**",
            "",
            f"- Cases: **{report['cases']}**",
            f"- Judge records: **{report['records']}**",
            f"- Ground truth: **{report['ground_truth_source']}**",
            "",
            "## Judge metrics",
            "",
            *rows,
            "",
            "## Confidence and latency",
            "",
            *calibration_rows,
            "",
            "## Pairwise agreement",
            "",
            *pair_rows,
            "",
            "## HOLD conditions",
            "",
            *hold_rows,
            "",
            "## Slice metrics",
            "",
        ]
        + [
            f"- `{slice_name}`: "
            + "; ".join(
                f"{judge} acc={_pct(metrics['accuracy'])} FPR={_pct(metrics['fpr'])} FNR={_pct(metrics['fnr'])}"
                for judge, metrics in judges.items()
            )
            for slice_name, judges in report["slices"].items()
        ]
        + [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def html_report(report: dict[str, Any]) -> str:
    cards = []
    for judge, metrics in report["judges"].items():
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">judge model</div>
              <h3>{html.escape(judge)}</h3>
              <div class="score">{_pct(metrics["accuracy"])}</div>
              <p>accuracy · Wilson {_ci(metrics["accuracy_wilson_95"])}</p>
              <div class="grid">
                <div><b>{_pct(metrics["fpr"])}</b><span>FPR / over-refusal</span></div>
                <div><b>{_pct(metrics["fnr"])}</b><span>FNR / dangerous miss</span></div>
                <div><b>{_pct(metrics["error_rate"])}</b><span>judge errors</span></div>
                <div><b>{_fmt_float(metrics["brier_score"], 4)}</b><span>Brier score</span></div>
                <div><b>{_fmt_float(metrics["ece_10_bin"], 4)}</b><span>ECE · 10 bins</span></div>
                <div><b>{metrics["latency_ms"]["p50"]:.0f} ms</b><span>latency p50</span></div>
                <div><b>{metrics["latency_ms"]["p95"]:.0f} ms</b><span>latency p95</span></div>
              </div>
            </article>
            """
        )
    hold_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['key'])}</code></td>"
        f"<td>{'PASS' if item['passed'] else 'FAIL'}</td>"
        f"<td><code>{html.escape(str(item['evidence']))}</code></td>"
        "</tr>"
        for item in report["hold_conditions"]
    )
    pair_html_rows = []
    for item in report["pairwise"]:
        kappa = "n/a" if item["cohen_kappa"] is None else f"{item['cohen_kappa']:.3f}"
        pair_html_rows.append(
            "<tr>"
            f"<td>{html.escape(item['left'])}</td>"
            f"<td>{html.escape(item['right'])}</td>"
            f"<td>{item['n']}</td><td>{_pct(item['agreement'])}</td>"
            f"<td>{kappa}</td></tr>"
        )
    pair_rows = "".join(pair_html_rows)
    decision_class = "pass" if report["decision"] == "PASS" else "hold"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety Judge Calibration</title>
<style>
:root {{ --ink:#161a1d; --paper:#edf1ef; --card:#fffefa; --line:#cbd3ce;
  --green:#12614b; --red:#ad382e; --amber:#a66412; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:
  linear-gradient(90deg,rgba(20,70,55,.06) 1px,transparent 1px),
  linear-gradient(rgba(20,70,55,.06) 1px,transparent 1px),var(--paper);
  background-size:28px 28px; font:16px/1.5 "Aptos","IBM Plex Sans",sans-serif; }}
header {{ padding:54px max(24px,7vw); background:
  radial-gradient(circle at 80% 20%, #d5e6dd, transparent 35%), #f7f2e8;
  border-bottom:2px solid var(--ink); }}
.eyebrow {{ color:var(--green); font-size:12px; font-weight:900;
  letter-spacing:.16em; text-transform:uppercase; }}
h1 {{ margin:12px 0; max-width:900px; font:800 clamp(42px,7vw,82px)/.95 Georgia,serif;
  letter-spacing:-.05em; }}
.lede {{ max-width:900px; font-size:20px; }}
.decision {{ display:inline-flex; gap:20px; margin-top:18px; padding:14px 18px;
  border-radius:14px; font:800 26px/1 Georgia,serif; }}
.decision.pass {{ background:#d8eee5; color:var(--green); }}
.decision.hold {{ background:#f5ded8; color:var(--red); }}
main {{ max-width:1280px; margin:auto; padding:34px 24px 80px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:20px; }}
.card,.panel {{ background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:24px; box-shadow:0 16px 40px rgba(35,50,43,.07); }}
.card h3 {{ margin:6px 0 12px; font:700 24px/1.1 Georgia,serif; overflow-wrap:anywhere; }}
.score {{ font:800 64px/.9 Georgia,serif; color:var(--green); }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:18px; }}
.grid div {{ border-top:1px solid var(--line); padding-top:8px; }}
.grid b {{ display:block; font-size:20px; }} .grid span {{ font-size:12px; color:#5c625e; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:9px 6px; border-bottom:1px solid var(--line); text-align:left;
  vertical-align:top; }}
.panel {{ margin-top:22px; }}
.panel h2 {{ font:700 28px/1.1 Georgia,serif; margin:6px 0 16px; }}
code {{ background:#e9eee9; padding:2px 5px; border-radius:4px; overflow-wrap:anywhere; }}
@media(max-width:720px) {{ h1 {{ font-size:50px; }} .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">LLM judge measurement</div>
  <h1>Safety Judge Calibration</h1>
  <p class="lede">FPR, FNR, agreement, Wilson intervals and explicit HOLD
  conditions across prompt injection, unauthorized requests and normal requests.</p>
  <p><strong>本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.</strong></p>
  <div class="decision {decision_class}">{report["decision"]} · {report["cases"]} cases</div>
</header>
<main>
  <section class="cards">{"".join(cards)}</section>
  <section class="panel">
    <div class="eyebrow">pairwise consistency</div>
    <h2>Judge-to-judge agreement</h2>
    <table><thead><tr><th>Judge A</th><th>Judge B</th><th>n</th>
    <th>Agreement</th><th>Cohen κ</th></tr></thead><tbody>{pair_rows}</tbody></table>
  </section>
  <section class="panel">
    <div class="eyebrow">deployment gate</div>
    <h2>HOLD conditions</h2>
    <table><thead><tr><th>Condition</th><th>Result</th><th>Evidence</th></tr></thead>
    <tbody>{hold_rows}</tbody></table>
  </section>
  <section class="panel">
    <div class="eyebrow">scope</div>
    <h2>What this does not prove</h2>
    <ul>{"".join(f"<li>{html.escape(item)}</li>" for item in report["limitations"])}</ul>
  </section>
</main>
</body>
</html>
"""


def write_reports(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    (output / "report.html").write_text(html_report(report), encoding="utf-8")
