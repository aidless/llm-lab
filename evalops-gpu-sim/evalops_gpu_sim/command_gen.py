import json
from pathlib import Path

def vllm_command(model: str, port: int = 8000, tp_size: int = 1) -> dict:
    return {"backend": "vllm", "command": f"vllm serve {model} --port {port} --tensor-parallel-size {tp_size}", "args": {"model": model, "port": port, "tp": tp_size}}

def lmdeploy_command(model: str, port: int = 8001, tp_size: int = 1) -> dict:
    return {"backend": "lmdeploy", "command": f"lmdeploy serve api_server {model} --server-port {port} --tensor-parallel-size {tp_size}", "args": {"model": model, "port": port, "tp": tp_size}}
