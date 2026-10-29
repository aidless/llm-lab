import json
def summarize(ollama_run: str, sim_run: str) -> dict:
    with open(ollama_run) as f: odata = json.load(f)
    with open(sim_run) as f: sdata = json.load(f)
    return {"ollama_tasks": odata.get("task_count"), "ollama_correct": odata.get("correct_count"), "simulated_throughput": sdata.get("throughput_tasks_per_sec"), "note": "GPU simulation placeholder; actual throughput requires real vLLM/LMDeploy on GPU"}
