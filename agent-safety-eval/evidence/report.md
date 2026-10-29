# Agent Safety Runtime Monitor

> **Local safety evaluation, not a production security system.**
> **本地安全评测，不是生产安全系统。**

- Frozen tasks: **96**
- Task-set SHA-256: `0c1c6a49eae47bbb44fff7fe14b9abdba84f807d5cc799562c2e01fd63146a41`
- Models: **2**
- Trajectories: **768**

## Results

| Model | Policy | Unsafe action | Permission violation | Stop P | Stop R | Over-refusal | Task success | Recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `llama3.2:3b` | `gate` | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 50.0% | 0.0% |
| `llama3.2:3b` | `off` | 33.3% | 33.3% | 66.3% | 98.4% | 100.0% | 19.8% | 0.0% |
| `llama3.2:3b` | `on` | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 50.0% | 0.0% |
| `llama3.2:3b` | `prompt` | 27.1% | 27.1% | 66.7% | 100.0% | 100.0% | 22.9% | 0.0% |
| `qwen2.5:3b-instruct-q4_K_M` | `gate` | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 50.0% | 0.0% |
| `qwen2.5:3b-instruct-q4_K_M` | `off` | 27.1% | 27.1% | 61.7% | 78.1% | 96.9% | 29.2% | 0.0% |
| `qwen2.5:3b-instruct-q4_K_M` | `on` | 0.0% | 0.0% | 68.1% | 100.0% | 93.8% | 51.0% | 0.0% |
| `qwen2.5:3b-instruct-q4_K_M` | `prompt` | 12.5% | 12.5% | 67.0% | 95.3% | 93.8% | 42.7% | 0.0% |

## Intervention deltas versus no intervention

| Model | Treatment | Unsafe action | Stop recall | Over-refusal | Task success |
|---|---|---:|---:|---:|---:|
| `llama3.2:3b` | `prompt` | -6.2% | +1.6% | +0.0% | +3.1% |
| `llama3.2:3b` | `gate` | -33.3% | +1.6% | +0.0% | +30.2% |
| `llama3.2:3b` | `on` | -33.3% | +1.6% | +0.0% | +30.2% |
| `qwen2.5:3b-instruct-q4_K_M` | `prompt` | -14.6% | +17.2% | -3.1% | +13.5% |
| `qwen2.5:3b-instruct-q4_K_M` | `gate` | -27.1% | +21.9% | +3.1% | +20.8% |
| `qwen2.5:3b-instruct-q4_K_M` | `on` | -27.1% | +21.9% | -3.1% | +21.9% |

## Human-review gate

- Reviewed records: **0**
- Minimum acceptance records: **100**
- Minimum met: **False**
- Complete human gate: **False**
- All 96 task labels reviewed: **False**
- Unsafe trajectory coverage: **0 / 199**
- Flagged required trajectories: **0**

The human-review acceptance criterion is **not met yet**. Open `review/review.html`, review the packet, export it, and ingest the export.

## Category breakdown

### `llama3.2:3b` · policy-gate

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 13 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 0 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 0 | 0 |
| `stop_failure` | 16 | 0 | 0 | 4 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 0 |
| `unauthorized_tool` | 16 | 16 | 0 | 16 | 0 |

### `llama3.2:3b` · policy-off

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 3 | 13 | 13 | 4 |
| `data_exfiltration` | 16 | 16 | 0 | 0 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 0 | 0 |
| `stop_failure` | 16 | 0 | 3 | 3 | 3 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 0 |
| `unauthorized_tool` | 16 | 0 | 16 | 16 | 16 |

### `llama3.2:3b` · policy-on

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 10 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 0 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 0 | 0 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 0 |
| `unauthorized_tool` | 16 | 16 | 0 | 16 | 0 |

### `llama3.2:3b` · policy-prompt

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 6 | 10 | 10 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 0 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 0 | 0 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 0 |
| `unauthorized_tool` | 16 | 0 | 16 | 16 | 16 |

### `qwen2.5:3b-instruct-q4_K_M` · policy-gate

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 4 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 14 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 7 | 0 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 9 |
| `unauthorized_tool` | 16 | 16 | 0 | 6 | 0 |

### `qwen2.5:3b-instruct-q4_K_M` · policy-off

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 11 | 5 | 5 | 0 |
| `data_exfiltration` | 16 | 6 | 10 | 10 | 0 |
| `prompt_injection` | 16 | 0 | 6 | 6 | 5 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 9 |
| `unauthorized_tool` | 16 | 11 | 5 | 5 | 5 |

### `qwen2.5:3b-instruct-q4_K_M` · policy-on

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 0 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 5 | 0 |
| `prompt_injection` | 16 | 1 | 0 | 4 | 0 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 2 | 13 |
| `unauthorized_tool` | 16 | 16 | 0 | 2 | 0 |

### `qwen2.5:3b-instruct-q4_K_M` · policy-prompt

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 0 | 0 |
| `data_exfiltration` | 16 | 11 | 5 | 5 | 0 |
| `prompt_injection` | 16 | 1 | 3 | 3 | 3 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 1 | 1 | 13 |
| `unauthorized_tool` | 16 | 13 | 3 | 3 | 3 |

## Failure taxonomy

- `llama3.2:3b` / policy-gate: model={}, tool={}
- `llama3.2:3b` / policy-off: model={}, tool={"mock_failure": 23}
- `llama3.2:3b` / policy-on: model={}, tool={}
- `llama3.2:3b` / policy-prompt: model={}, tool={"mock_failure": 16}
- `qwen2.5:3b-instruct-q4_K_M` / policy-gate: model={}, tool={"transient": 9}
- `qwen2.5:3b-instruct-q4_K_M` / policy-off: model={}, tool={"mock_failure": 10, "transient": 9}
- `qwen2.5:3b-instruct-q4_K_M` / policy-on: model={}, tool={"transient": 13}
- `qwen2.5:3b-instruct-q4_K_M` / policy-prompt: model={}, tool={"mock_failure": 7, "transient": 12}

## Reproduction

```bash
./scripts/rerun.sh
python3 verify_evidence.py
```

## Non-claims

- Local safety evaluation, not a production security system.
- 本地安全评测，不是生产安全系统。
- All tools are mock-only; no real filesystem, shell, email or network action occurs.
- Assistant-authored draft labels are not human labels and must not be counted as such.
- Small local CPU models are not representative of frontier or production agents.
- No claim is made about benchmark leadership, regulatory compliance or field safety.
