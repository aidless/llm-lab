import typer, sys
from . import diagnose, command_gen, probe, simulate, summarize
app = typer.Typer()

@app.command("diagnose")
def cmd_diagnose(): return diagnose.run_diagnose()

@app.command("command")
def cmd_command(backend: str = typer.Option(help="vllm or lmdeploy"), model: str = typer.Option(), port: int = typer.Option(8000), tp: int = typer.Option(1)):
    if backend == "vllm": return command_gen.vllm_command(model, port, tp)
    if backend == "lmdeploy": return command_gen.lmdeploy_command(model, port, tp)
    raise ValueError("backend must be vllm or lmdeploy")

@app.command("probe")
def cmd_probe(backend: str = typer.Option(help="ollama, vllm or lmdeploy"), base_url: str = typer.Option(), model: str = typer.Option()):
    return probe.probe_openai_compat(base_url, model)

@app.command("simulate-ceval201")
def cmd_simulate(backend: str = typer.Option(), output: str = typer.Option()):
    result = simulate.simulate_ceval201(backend, "placeholder"); print(json.dumps(result, indent=2))
    return result

@app.command("summarize")
def cmd_summarize(ollama_run: str, sim_run: str):
    print(json.dumps(summarize.summarize(ollama_run, sim_run), indent=2))

if __name__ == "__main__": app(prog_name="evalops-gpu-sim")
