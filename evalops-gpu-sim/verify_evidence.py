#!/usr/bin/env python3
import json, pathlib, sys
from hashlib import sha256

ROOT = pathlib.Path(__file__).parent
EVIDENCE = ROOT/"evidence"

def verify():
    # Check checksums
    with open(EVIDENCE/"SHA256SUMS") as f:
        lines = [l.strip() for l in f if l.strip()]
    known = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2: known[parts[1]] = parts[0]
    for name, h in known.items():
        path = EVIDENCE/name
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != h: print(f"MISMATCH: {name}"); return False
    # Decision gate
    report_path = EVIDENCE/"report-vllm-simulation.json"
    with open(report_path) as f: rdata = json.load(f)
    gpu_executed = (rdata.get("gpu_count",0)>0)
    if gpu_executed: decision = "PASS"
    else: decision = "HOLD"
    print(json.dumps({"decision": decision, "real_gpu_run_executed": gpu_executed}, indent=2))
    return True

if __name__ == "__main__": sys.exit(0 if verify() else 1)
