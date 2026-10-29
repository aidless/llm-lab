# Eval Infra Contract Smoke

- Overall: **PASS**
- Backends exercised: **vllm, lmdeploy**
- Claim boundary: localhost fixture + managed subprocess only.
- 本地安全评测，不是生产安全系统。 Local contract smoke, not a GPU serving benchmark.

## Backend results

| Backend contract label | Fixture | Real backend executed | Process ready | Probe | Generated command (not executed) |
|---|---|---:|---:|---:|---|
| `vllm` | `scripts/fake_openai_server.py` | False | True | True | `vllm serve fixture-model --host 127.0.0.1 --port 33855 --tensor-parallel-size 1` |
| `lmdeploy` | `scripts/fake_openai_server.py` | False | True | True | `lmdeploy serve api_server fixture-model --server-name 127.0.0.1 --server-port 58779 --tp 1` |

## Non-claims

- This is a localhost OpenAI-compatible fixture, not a real vLLM or LMDeploy server.
- The host has no usable NVIDIA GPU and neither serving package was executed.
- Passing this smoke proves the adapter, fixture compatibility and process lifecycle only.
- The generated vLLM/LMDeploy commands are displayed but intentionally not executed.
