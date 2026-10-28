# Eval Infra Contract Smoke

- Overall: **PASS**
- Backends exercised: **vllm, lmdeploy**
- Claim boundary: localhost fixture + managed subprocess only.
- 本地安全评测，不是生产安全系统。 Local contract smoke, not a GPU serving benchmark.

## Backend results

| Backend | Process ready | Probe | Generated command |
|---|---:|---:|---|
| `vllm` | True | True | `vllm serve fixture-model --host 127.0.0.1 --port 41243 --tensor-parallel-size 1` |
| `lmdeploy` | True | True | `lmdeploy serve api_server fixture-model --server-name 127.0.0.1 --server-port 60409 --tp 1` |

## Non-claims

- This is a localhost OpenAI-compatible fixture, not a real vLLM or LMDeploy server.
- The host has no usable NVIDIA GPU and neither serving package was executed.
- Passing this smoke proves the adapter and process lifecycle only.
