import typer, json, pathlib
from . import trainer
app = typer.Typer()

@app.command("train")
def cmd_train(
    model: str = typer.Option(default="Qwen/Qwen3-0.6B-Instruct", help="Base model"),
    train_data: str = typer.Option(help="Path to DPO triples JSONL"),
    output_dir: str = typer.Option(default="./checkpoints", help="Output directory"),
    lr: float = typer.Option(default=1e-5, help="Learning rate"),
    epochs: int = typer.Option(default=3, help="Training epochs"),
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = trainer.train_dpo(model, train_data, output_dir, lr, epochs)
    print(json.dumps({"status": "completed", "output_dir": output_dir}, indent=2))

if __name__ == "__main__": app(prog_name="dpo_trainer")
