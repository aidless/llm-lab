"""llm-lab CLI — run evals and comparisons from the terminal."""

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_lab import export as export_mod

app = typer.Typer(name="llm-lab", help="Local-first LLM eval pipeline", add_completion=False)
console = Console()


def _row_tokens(raw: Any) -> int:
    """Parse token usage as stored in event_log (JSON string, int, or None)."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, dict):
        return int(raw.get("total_tokens", 0) or 0)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return 0
        if isinstance(data, dict):
            return int(data.get("total_tokens", 0) or 0)
    return 0


def _row_verdict(row: dict[str, Any]) -> str:
    """Derive a verdict label for an event row.

    event_log has no dedicated verdict column; verdicts live inside the run
    result stored in ``detail`` (API-driven runs) or are absent (CLI runs).
    """
    v = row.get("verdict")
    if v:
        return v
    detail = row.get("detail")
    if isinstance(detail, str):
        try:
            data = json.loads(detail)
        except (json.JSONDecodeError, ValueError):
            return "pass"
        if isinstance(data, dict) and "steps_detail" in data:
            details = data["steps_detail"]
            return "fail" if any(
                (d.get("verdict") or {}).get("label") == "fail" for d in details
            ) else "pass"
    return "pass"


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit", is_eager=True),
) -> None:
    if version:
        console.print("llm-lab 0.1.0")
        raise typer.Exit()
    if ctx.invoked_subcommand is None and not version:
        console.print("[yellow]llm-lab[/yellow] – use [bold]llm-lab --help[/bold] for commands")
        raise typer.Exit()


# ── run ──────────────────────────────────────────────────────────────────────


@app.command()
def run(
    goal: str = typer.Argument(..., help="Task description"),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
    verifier: str = typer.Option("deepeval", "--verifier", "-v", help="Verifier name"),
    preset: str = typer.Option(None, "--preset", "-p", help="Model preset (cheap, balanced, best, quick)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be done without running"),
):
    """Run a single evaluation plan."""
    if preset:
        from llm_lab.settings import resolve_preset

        cfg = resolve_preset(preset)
        if cfg:
            model = model or cfg["model"]
            verifier = cfg["verifier"]

    if dry_run:
        console.print("[yellow]DRY-RUN:[/yellow] would call")
        console.print(
            f"  goal=[cyan]{goal}[/cyan]  model=[cyan]{model or 'default'}[/cyan]  verifier=[cyan]{verifier}[/cyan]"
        )
        return
    from llm_lab.runner import run_plan
    from llm_lab.tracer import trace_call

    with console.status("[bold green]Running plan..."):
        result = run_plan(goal, model, verifier)

    # Persist the run so `report`/`diff`/`history` can read an accurate verdict.
    with contextlib.suppress(Exception):
        asyncio.run(
            trace_call(
                intent_id=result["run_id"],
                seq=1,
                model=result["model"],
                prompt=result["goal"],
                output=json.dumps(result, ensure_ascii=False),
                token_usage={"total_tokens": result["total_tokens"]},
                cost_usd=result["total_cost_usd"],
                verdict="pass" if result["all_passed"] else "fail",
            )
        )

    if json_output:
        console.print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    _print_run_result(result)


def _print_run_result(result: dict) -> None:
    status = "✅ PASS" if result["all_passed"] else "❌ FAIL"
    console.print(
        Panel(
            f"[bold]{status}[/bold]  |  model: [cyan]{result['model']}[/cyan]  "
            f"|  template: [yellow]{result['plan_template'] or 'llm-fallback'}[/yellow]",
            title=f"Run [bold]{result['intent_id']}[/bold]",
        )
    )
    table = Table("Step", "Action", "Verdict", "Tokens", "Cost")
    for s in result["steps_detail"]:
        v = s["verdict"]
        label = v["label"]
        badge = {"pass": "[green]PASS[/green]", "fail": "[red]FAIL[/red]", "partial": "[yellow]PARTIAL[/yellow]"}.get(
            label, label
        )
        table.add_row(
            str(len(table.rows) + 1), s["action"], badge, str(s.get("tokens", "—")), f"${s.get('cost', 0):.6f}"
        )
    console.print(table)
    console.print(f"[dim]Total: {result['total_tokens']} tokens  |  ${result['total_cost_usd']:.6f}[/dim]")


# ── compare ──────────────────────────────────────────────────────────────────


@app.command()
def compare(
    goal: str = typer.Argument(..., help="Task description"),
    model_a: str = typer.Option(None, "--model-a", "-a", help="First model"),
    model_b: str = typer.Option(None, "--model-b", "-b", help="Second model"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
):
    """Compare two models on the same task."""
    from llm_lab.runner import compare as run_compare

    with console.status("[bold green]Running A/B comparison..."):
        result = run_compare(goal, model_a, model_b)

    if json_output:
        console.print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    winner = result["summary"]["winner"]
    w_str = {"a": "Model A wins 🏆", "b": "Model B wins 🏆", "tie": "Tie 🤝"}.get(winner, winner)
    delta = result["summary"]["cost_delta"]
    tdelta = result["summary"]["token_delta"]

    console.print(
        Panel(
            f"[bold]{w_str}[/bold]\n"
            f"Cost Δ: [cyan]{'+' if delta >= 0 else ''}{delta:.6f}[/cyan]  |  "
            f"Token Δ: [cyan]{'+' if tdelta >= 0 else ''}{tdelta}[/cyan]",
            title="A/B Comparison",
        )
    )

    for label, side in [("Model A", "model_a"), ("Model B", "model_b")]:
        m = result[side]
        status = "✅" if m["all_passed"] else "❌"
        console.print(f"\n[bold]{label}[/bold] ({m['model']}) {status}")
        console.print(f"  Tokens: {m['total_tokens']}  |  Cost: ${m['total_cost_usd']:.6f}")
        for s in m["steps"]:
            badge = {"pass": "🟢", "fail": "🔴", "partial": "🟡"}.get(s["verdict"]["label"], "⚪")
            console.print(f"  {badge} {s['action']}: {s['output'][:120]}{'...' if len(s['output']) > 120 else ''}")


# ── serve ────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    port: int = typer.Option(8123, "--port", "-p", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Bind address"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Hot reload"),
) -> None:
    """Start the web UI server."""
    import uvicorn

    console.print(f"[green]Starting llm-lab server at http://{host}:{port}[/green]")
    uvicorn.run("main:app", host=host, port=port, reload=reload)


# ── history ──────────────────────────────────────────────────────────────────


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
):
    """List recent evaluation runs."""

    async def _fetch():
        from llm_lab import db as database

        await database.init_db()
        return await database.list_intents(limit)

    rows = asyncio.run(_fetch())

    if json_output:
        console.print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    if not rows:
        console.print("[dim]No runs yet.[/dim]")
        return

    table = Table("intent_id", "Action", "Model", "Timestamp")
    for r in rows:
        table.add_row(r["intent_id"], r["action"], r["model"] or "—", r["timestamp"][:19])
    console.print(table)


# ── export ───────────────────────────────────────────────────────────────────


@app.command()
def export(
    intent_id: str = typer.Argument(..., help="Run ID to export"),
    output: str = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, xlsx)"),
):
    """Export a run as JSON, CSV, or XLSX."""

    async def _fetch():
        from llm_lab import db as database

        await database.init_db()
        rows = await database.get_events(intent_id)
        return [dict(r) for r in rows]

    rows = asyncio.run(_fetch())
    if not rows:
        console.print("[yellow]No data found for this run.[/yellow]")
        raise typer.Exit(code=1)

    if fmt == "json":
        blob = export_mod.export_json(intent_id, rows)
    elif fmt == "csv":
        blob = export_mod.export_csv(rows)
    elif fmt == "xlsx":
        try:
            content = export_mod.export_xlsx(rows)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None
        if output:
            Path(output).write_bytes(content)
            console.print(f"[green]Exported to {output}[/green]")
        else:
            console.print("[yellow]XLSX output requires --output/-o[/yellow]")
        return
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")
        raise typer.Exit(code=1)

    if output:
        Path(output).write_text(blob, encoding="utf-8")
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(blob)


# ── report ───────────────────────────────────────────────────────────────────


@app.command()
def report(
    intent_id: str = typer.Argument(..., help="Run ID to generate report for"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file"),
):
    """Generate an HTML report for a completed run."""

    async def _fetch():
        from llm_lab import db as database

        await database.init_db()
        rows = await database.get_events(intent_id)
        run_info = await database.get_run_info(intent_id)
        return rows, run_info

    rows, run_info = asyncio.run(_fetch())
    if not rows:
        console.print("[yellow]No data found for this run.[/yellow]")
        raise typer.Exit(code=1)

    result = {
        "intent_id": intent_id,
        "goal": run_info.get("goal", "") if run_info else "",
        "model": rows[0].get("model", "—") if rows else "—",
        "steps": len(rows),
        "total_tokens": sum(_row_tokens(r.get("token_usage")) for r in rows),
        "total_cost_usd": sum(r.get("cost_usd", 0) or 0 for r in rows),
        "all_passed": all(_row_verdict(r) != "fail" for r in rows),
        "steps_detail": [
            {
                "action": r.get("action", ""),
                "prompt": "",
                "output": r.get("detail", ""),
                "model": r.get("model", ""),
                "verdict": {"label": _row_verdict(r), "reason": ""},
                "tokens": _row_tokens(r.get("token_usage")),
                "cost": r.get("cost_usd", 0) or 0,
            }
            for r in rows
        ],
    }

    html = export_mod.export_html(result)
    if output:
        Path(output).write_text(html, encoding="utf-8")
        console.print(f"[green]Report written to {output}[/green]")
    else:
        console.print(html)


# ── watch ────────────────────────────────────────────────────────────────────


@app.command()
def watch(
    goal: str = typer.Argument(..., help="Task description"),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
    verifier: str = typer.Option("deepeval", "--verifier", "-v", help="Verifier name"),
    watch_dir: str = typer.Option(".", "--watch-dir", "-d", help="Directory to watch"),
    interval: int = typer.Option(5, "--interval", "-i", help="Poll interval in seconds"),
):
    """Watch a directory and re-run the plan on file changes."""
    from llm_lab.runner import run_plan

    last_hashes: dict[str, float] = {}

    def _check() -> bool:
        changed = False
        for f in Path(watch_dir).rglob("*"):
            if f.suffix in (".py", ".yaml", ".yml", ".md", ".txt") and f.is_file():
                mtime = f.stat().st_mtime
                key = str(f.resolve())
                if key not in last_hashes:
                    last_hashes[key] = mtime
                elif mtime > last_hashes[key]:
                    last_hashes[key] = mtime
                    changed = True
        return changed

    console.print(f"[green]Watching [bold]{watch_dir}[/bold] for changes (poll {interval}s)...[/green]")
    while True:
        if _check():
            console.print("[yellow]Change detected, re-running...[/yellow]")
            with console.status("[bold green]Running plan..."):
                result = run_plan(goal, model, verifier)
            _print_run_result(result)
        time.sleep(interval)


# ── diff ─────────────────────────────────────────────────────────────────────


@app.command()
def diff(
    intent_a: str = typer.Argument(..., help="First run ID"),
    intent_b: str = typer.Argument(..., help="Second run ID"),
):
    """Compare two runs side by side."""

    async def _fetch(iid: str):
        from llm_lab import db as database

        await database.init_db()
        rows = await database.get_events(iid)
        return [dict(r) for r in rows]

    rows_a, rows_b = asyncio.run(_fetch(intent_a)), asyncio.run(_fetch(intent_b))

    if not rows_a:
        console.print(f"[yellow]No data for {intent_a}[/yellow]")
        raise typer.Exit(code=1)
    if not rows_b:
        console.print(f"[yellow]No data for {intent_b}[/yellow]")
        raise typer.Exit(code=1)

    cost_a = sum(r.get("cost_usd", 0) or 0 for r in rows_a)
    cost_b = sum(r.get("cost_usd", 0) or 0 for r in rows_b)
    tokens_a = sum(_row_tokens(r.get("token_usage")) for r in rows_a)
    tokens_b = sum(_row_tokens(r.get("token_usage")) for r in rows_b)

    table = Table("", intent_a, intent_b)
    table.add_row("Model", rows_a[0].get("model", "—"), rows_b[0].get("model", "—"))
    table.add_row("Steps", str(len(rows_a)), str(len(rows_b)))
    table.add_row("Total Tokens", str(tokens_a), str(tokens_b))
    table.add_row("Total Cost", f"${cost_a:.6f}", f"${cost_b:.6f}")
    table.add_row("Cost Δ", "", f"${cost_b - cost_a:+.6f}")
    table.add_row("Token Δ", "", f"{tokens_b - tokens_a:+d}")
    console.print(table)


# ── entry point ──────────────────────────────────────────────────────────────


def entry() -> None:
    app()


if __name__ == "__main__":
    entry()
