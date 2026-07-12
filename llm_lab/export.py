"""Export utilities — JSON, CSV, XLSX, and HTML report."""

import csv
import io
import json
from typing import Any

CSV_COLUMNS = ["id", "intent_id", "seq", "timestamp", "action", "model", "detail", "cost_usd"]


def export_json(intent_id: str, events: list[dict[str, Any]]) -> str:
    return json.dumps({"intent_id": intent_id, "events": events}, indent=2, ensure_ascii=False)


def export_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(CSV_COLUMNS)
    for r in rows:
        writer.writerow([r.get(c) for c in CSV_COLUMNS])
    return output.getvalue()


def export_xlsx(rows: list[dict[str, Any]]) -> bytes:
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl is required for XLSX export; pip install openpyxl") from None

    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is None:
        ws = wb.create_sheet("Events")
    ws.title = "Events"
    ws.append(CSV_COLUMNS)
    for r in rows:
        ws.append([r.get(c) for c in CSV_COLUMNS])
    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def export_html(result: dict[str, Any]) -> str:
    steps = result.get("steps_detail", [])
    rows_html = ""
    for i, s in enumerate(steps):
        v = s.get("verdict", {})
        label = v.get("label", "—")
        badge = {"pass": "pass", "fail": "fail", "partial": "partial"}.get(label, "—")
        rows_html += f"""<tr>
            <td>{i + 1}</td>
            <td>{s.get('action', '—')}</td>
            <td><span class="badge badge-{badge}">{label.upper()}</span></td>
            <td>{s.get('tokens', '—')}</td>
            <td>${s.get('cost', 0):.6f}</td>
        </tr>"""

    passed = result.get("all_passed", False)
    status = "PASS" if passed else "FAIL"
    status_class = "pass" if passed else "fail"

    steps_json = json.dumps(steps, indent=2, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm-lab Report — {result.get('intent_id', 'unknown')}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #1a1a1a; padding: 2rem; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #666; font-size: 0.875rem; margin-bottom: 1.5rem; }}
  .summary {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1rem 1.25rem; flex: 1; min-width: 140px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; color: #888; }}
  .card .value {{ font-size: 1.25rem; font-weight: 600; margin-top: 0.25rem; }}
  .card .value.pass {{ color: #16a34a; }}
  .card .value.fail {{ color: #dc2626; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th, td {{ text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; color: #888; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 0.125rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .badge-pass {{ background: #dcfce7; color: #16a34a; }}
  .badge-fail {{ background: #fee2e2; color: #dc2626; }}
  .badge-partial {{ background: #fef9c3; color: #ca8a04; }}
  pre {{ background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; margin-top: 1.5rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>llm-lab Report</h1>
  <div class="subtitle">Run {result.get('intent_id', '—')} &middot; {result.get('model', '—')}</div>

  <div class="summary">
    <div class="card">
      <label>Status</label>
      <div class="value {status_class}">{status}</div>
    </div>
    <div class="card">
      <label>Model</label>
      <div class="value">{result.get('model', '—')}</div>
    </div>
    <div class="card">
      <label>Goal</label>
      <div class="value" style="font-size:1rem">{result.get('goal', '—')}</div>
    </div>
    <div class="card">
      <label>Steps</label>
      <div class="value">{result.get('steps', 0)}</div>
    </div>
    <div class="card">
      <label>Tokens</label>
      <div class="value">{result.get('total_tokens', 0)}</div>
    </div>
    <div class="card">
      <label>Cost</label>
      <div class="value">${result.get('total_cost_usd', 0):.6f}</div>
    </div>
  </div>

  <table>
    <thead><tr><th>#</th><th>Action</th><th>Verdict</th><th>Tokens</th><th>Cost</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <pre>{steps_json}</pre>
</div>
</body>
</html>"""
