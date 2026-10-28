"""Metrics, threshold tuning, gate decisions and report rendering."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from llm_lab.evalops import write_json

from .scanner import Match, is_suspect


def wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 1.0}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _predictions(
    rows: list[dict[str, Any]],
    *,
    split: str,
    ngram_threshold: float,
    semantic_threshold: float,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if split != "all" and row["split"] != split:
            continue
        match = Match(**row["match"])
        output.append(
            {
                **row,
                "suspect": is_suspect(
                    match,
                    ngram_threshold=ngram_threshold,
                    semantic_threshold=semantic_threshold,
                ),
            }
        )
    return output


def _score(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    positives = [row for row in rows if row["gold_contaminated"]]
    negatives = [row for row in rows if not row["gold_contaminated"]]
    tp = sum(row["suspect"] for row in positives)
    fp = sum(row["suspect"] for row in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": split,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tune_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] | None = None
    for ngram_threshold in [value / 20 for value in range(8, 20)]:
        for semantic_threshold in [value / 20 for value in range(10, 20)]:
            predicted = _predictions(
                rows,
                split="dev",
                ngram_threshold=ngram_threshold,
                semantic_threshold=semantic_threshold,
            )
            score = _score(predicted, "dev")
            candidate = {
                "ngram_threshold": ngram_threshold,
                "semantic_threshold": semantic_threshold,
                "dev_f1": score["f1"],
                "dev_precision": score["precision"],
                "dev_recall": score["recall"],
            }
            if best is None or (
                candidate["dev_f1"],
                candidate["dev_precision"],
                -candidate["ngram_threshold"],
                -candidate["semantic_threshold"],
            ) > (
                best["dev_f1"],
                best["dev_precision"],
                -best["ngram_threshold"],
                -best["semantic_threshold"],
            ):
                best = candidate
    if best is None:
        raise RuntimeError("threshold tuning produced no candidate")
    return best


def build_report(
    *,
    rows: list[dict[str, Any]],
    corpus_manifest: dict[str, Any],
    human_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = tune_thresholds(rows)
    prediction_thresholds = {
        "ngram_threshold": thresholds["ngram_threshold"],
        "semantic_threshold": thresholds["semantic_threshold"],
    }
    test_rows = _predictions(rows, split="test", **prediction_thresholds)
    dev_rows = _predictions(rows, split="dev", **prediction_thresholds)
    test_score = _score(test_rows, "test")
    positives = [row for row in test_rows if row["gold_contaminated"]]
    negatives = [row for row in test_rows if not row["gold_contaminated"]]
    false_negatives = [row for row in positives if not row["suspect"]]
    false_positives = [row for row in negatives if row["suspect"]]
    reviews_complete = len(human_reviews) >= 40
    fn_upper = wilson(len(false_negatives), len(positives))["high"]
    fp_upper = wilson(len(false_positives), len(negatives))["high"]
    gate_reasons = []
    if not reviews_complete:
        gate_reasons.append("human_review_lt_40")
    if test_score["recall"] < 0.85:
        gate_reasons.append("test_recall_lt_0.85")
    if test_score["precision"] < 0.85:
        gate_reasons.append("test_precision_lt_0.85")
    if fn_upper > 0.30:
        gate_reasons.append("false_negative_wilson_upper_gt_0.30")
    if fp_upper > 0.12:
        gate_reasons.append("false_positive_wilson_upper_gt_0.12")
    decision = (
        "BLOCK"
        if (test_score["recall"] < 0.70 or fn_upper > 0.35)
        else "HOLD"
        if gate_reasons
        else "PASS"
    )
    metrics = {
        "contamination_rate": test_score["tp"] / len(test_rows) if test_rows else None,
        "false_positive_rate": test_score["fp"] / len(negatives) if negatives else None,
        "false_negative_rate": test_score["fn"] / len(positives) if positives else None,
        "precision": test_score["precision"],
        "recall": test_score["recall"],
        "f1": test_score["f1"],
        "false_positive_rate_wilson_95": wilson(len(false_positives), len(negatives)),
        "false_negative_rate_wilson_95": wilson(len(false_negatives), len(positives)),
    }
    suspicious_samples = sorted(
        [row for row in test_rows + dev_rows if row["suspect"]],
        key=lambda item: (
            not item["match"]["exact"],
            -item["match"]["ngram_score"],
            -item["match"]["semantic_score"],
        ),
    )[:60]
    return {
        "schema_version": "benchmark-contamination.report.v1",
        "manifest": corpus_manifest,
        "thresholds": thresholds,
        "metrics": metrics,
        "test_confusion": test_score,
        "dev_confusion": _score(dev_rows, "dev"),
        "false_positive_samples": [row["task_id"] for row in false_positives[:20]],
        "false_negative_samples": [row["task_id"] for row in false_negatives[:20]],
        "suspicious_samples": suspicious_samples,
        "human_review": {
            "reviewed_records": len(human_reviews),
            "gate_40_reviews": reviews_complete,
            "reviewer_ids": sorted(
                {
                    str(row.get("reviewer"))
                    for row in human_reviews
                    if str(row.get("reviewer", "")).strip()
                }
            ),
        },
        "decision": decision,
        "gate_reasons": gate_reasons,
        "limitations": corpus_manifest["limitations"]
        + [
            "Threshold tuning uses the dev split only, but the fixture is synthetic.",
            "Embedding similarity is model-dependent and not a proof of memorization.",
            "A clean result cannot prove a model was not exposed to the benchmark.",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    rows = [
        "# Benchmark Contamination Preflight",
        "",
        f"> Decision: **{report['decision']}**",
        "> **本地安全评测，不是生产安全系统。 / Local safety evaluation, not a production security system.**",
        "",
        f"- Frozen tasks: **{report['manifest']['tasks']['count']}**",
        f"- Corpus documents: **{report['manifest']['corpus']['count']}**",
        f"- N-gram threshold: `{report['thresholds']['ngram_threshold']:.2f}`",
        f"- Semantic threshold: `{report['thresholds']['semantic_threshold']:.2f}`",
        "",
        "## Test metrics",
        "",
        "| Metric | Value | 95% Wilson |",
        "|---|---:|---:|",
        f"| Contamination rate | {_pct(metrics['contamination_rate'])} | n/a |",
        f"| False positive rate | {_pct(metrics['false_positive_rate'])} | "
        f"{_ci(metrics['false_positive_rate_wilson_95'])} |",
        f"| False negative rate | {_pct(metrics['false_negative_rate'])} | "
        f"{_ci(metrics['false_negative_rate_wilson_95'])} |",
        f"| Precision | {_pct(metrics['precision'])} | n/a |",
        f"| Recall | {_pct(metrics['recall'])} | n/a |",
        f"| F1 | {metrics['f1']:.3f} | n/a |",
        "",
        "## Gate",
        "",
    ]
    if report["gate_reasons"]:
        rows.extend(f"- `{reason}`" for reason in report["gate_reasons"])
    else:
        rows.append("- All declared preflight conditions passed.")
    rows += [
        "",
        "## EvalOps integration",
        "",
        "```bash",
        "python3 -m contamination_scan report",
        "python3 -m contamination_scan gate --report evidence/report.json",
        "```",
        "",
        "A non-`PASS` decision exits non-zero and must stop an EvalOps run.",
        "",
        "## Limitations",
        "",
    ]
    rows.extend(f"- {item}" for item in report["limitations"])
    rows.append("")
    return "\n".join(rows)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _ci(interval: dict[str, float]) -> str:
    return f"[{interval['low']:.1%}, {interval['high']:.1%}]"


def html_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    decision_class = report["decision"].lower()
    reason_items = (
        "".join(f"<li><code>{html.escape(reason)}</code></li>" for reason in report["gate_reasons"])
        or "<li>All declared conditions passed.</li>"
    )
    suspects = "".join(
        "<tr>"
        f"<td><code>{html.escape(row['task_id'])}</code></td>"
        f"<td>{'yes' if row['match']['exact'] else 'no'}</td>"
        f"<td>{row['match']['ngram_score']:.3f}</td>"
        f"<td>{row['match']['semantic_score']:.3f}</td>"
        f"<td><code>{html.escape(row['match']['document_id'])}</code></td>"
        "</tr>"
        for row in report["suspicious_samples"][:30]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark Contamination Preflight</title>
<style>
:root{{--ink:#172019;--paper:#eff0e7;--card:#fffef7;--line:#c9c9b8;
--green:#1e6345;--red:#a73a2f;--amber:#a66a12}}
*{{box-sizing:border-box}}body{{margin:0;background:
repeating-linear-gradient(135deg,#eceee4 0,#eceee4 1px,transparent 1px,transparent 14px);
color:var(--ink);font:16px/1.5 "Aptos",sans-serif}}
header{{padding:54px max(24px,7vw);background:linear-gradient(110deg,#f8f4e8,#e1e9dd);
border-bottom:2px solid var(--ink)}} .eyebrow{{font-size:12px;font-weight:900;
letter-spacing:.16em;color:var(--green);text-transform:uppercase}}
h1{{font:800 clamp(42px,7vw,84px)/.95 Georgia,serif;letter-spacing:-.05em;
max-width:900px;margin:12px 0}} .lede{{max-width:900px;font-size:20px}}
.decision{{display:inline-block;padding:12px 18px;border-radius:12px;margin-top:16px;
font:800 25px/1 Georgia,serif}} .decision.pass{{background:#d8eee4;color:var(--green)}}
.decision.hold{{background:#faecd5;color:var(--amber)}} .decision.block{{
background:#f4dcd7;color:var(--red)}}
main{{max-width:1280px;margin:auto;padding:34px 24px 80px;display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;
box-shadow:0 16px 40px rgba(39,48,35,.07)}} .wide{{grid-column:1/-1}}
.panel h2{{font:700 28px/1.1 Georgia,serif;margin:6px 0 14px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
.stats div{{border-top:1px solid var(--line);padding-top:10px}}
.stats b{{font:800 38px/.9 Georgia,serif;display:block}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:9px 6px;border-bottom:1px solid var(--line);text-align:left}}
code{{background:#e9ece3;padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}}
ul{{padding-left:22px}}@media(max-width:850px){{main{{grid-template-columns:1fr}}
.wide{{grid-column:auto}}.stats{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="eyebrow">EvalOps preflight · seeded audit fixture</div>
<h1>Benchmark Contamination Scanner</h1>
<p class="lede">Exact, 13-character n-gram and local semantic matching against a
frozen candidate corpus, with dev-only threshold tuning and explicit review gates.</p>
<p><strong>本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.</strong></p>
<div class="decision {decision_class}">{report["decision"]}</div></header>
<main>
<section class="panel"><div class="eyebrow">test split</div><h2>Detection quality</h2>
<div class="stats">
<div><b>{_pct(metrics["precision"])}</b><span>precision</span></div>
<div><b>{_pct(metrics["recall"])}</b><span>recall</span></div>
<div><b>{metrics["f1"]:.3f}</b><span>F1</span></div>
<div><b>{_pct(metrics["false_positive_rate"])}</b><span>false positive</span></div>
<div><b>{_pct(metrics["false_negative_rate"])}</b><span>false negative</span></div>
<div><b>{report["human_review"]["reviewed_records"]}</b><span>human reviews</span></div>
</div></section>
<section class="panel"><div class="eyebrow">gate status</div><h2>Preflight reasons</h2>
<ul>{reason_items}</ul>
<p>N-gram: <code>{report["thresholds"]["ngram_threshold"]:.2f}</code> ·
Semantic: <code>{report["thresholds"]["semantic_threshold"]:.2f}</code></p></section>
<section class="panel wide"><div class="eyebrow">review queue</div>
<h2>Suspicious samples</h2><table><thead><tr><th>Task</th><th>Exact</th>
<th>N-gram</th><th>Semantic</th><th>Matched document</th></tr></thead>
<tbody>{suspects}</tbody></table></section>
<section class="panel wide"><div class="eyebrow">evidence boundary</div>
<h2>What this does not prove</h2>
<ul>{"".join(f"<li>{html.escape(item)}</li>" for item in report["limitations"])}</ul>
</section></main></body></html>"""


def write_reports(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    (output / "report.html").write_text(html_report(report), encoding="utf-8")
