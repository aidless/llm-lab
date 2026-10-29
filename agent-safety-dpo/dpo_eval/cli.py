import typer, json, pathlib
from . import runner
app = typer.Typer()

@app.command("run")
def cmd_run(
    model: str = typer.Option(help="Path to trained DPO model"),
    benchmark: str = typer.Option(help="Path to benchmark tasks JSONL"),
    output: str = typer.Option(default="evidence/report.md", help="Output path"),
):
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    metrics = runner.evaluate_hybrid_model(model, benchmark, output)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": app(prog_name="dpo_eval")
