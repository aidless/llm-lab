# Agent Safety Runtime Monitor

> **Local safety evaluation, not a production security system.**
> **本地安全评测，不是生产安全系统。**

- Frozen tasks: **96**
- Task-set SHA-256: `0c1c6a49eae47bbb44fff7fe14b9abdba84f807d5cc799562c2e01fd63146a41`
- Models: **2**
- Trajectories: **384**

## Results

| Model | Policy | Unsafe action | Permission violation | Stop P | Stop R | Over-refusal | Task success | Recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `llama3.2:1b` | `off` | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 50.0% | 0.0% |
| `llama3.2:1b` | `on` | 0.0% | 0.0% | 67.4% | 100.0% | 96.9% | 50.0% | 0.0% |
| `qwen3:0.6b` | `off` | 62.5% | 62.5% | 69.0% | 62.5% | 56.2% | 6.2% | 0.0% |
| `qwen3:0.6b` | `on` | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 27.1% | 0.0% |

## Policy-off → policy-on deltas

| Model | Unsafe action | Permission violation | Stop recall | Task success |
|---|---:|---:|---:|---:|
| `llama3.2:1b` | +0.0% | +0.0% | +0.0% | +0.0% |
| `qwen3:0.6b` | -62.5% | -62.5% | +37.5% | +20.8% |

## Human-review gate

- Reviewed records: **0**
- Minimum acceptance records: **100**
- Minimum met: **False**
- Complete human gate: **False**
- All 96 task labels reviewed: **False**
- Unsafe trajectory coverage: **0 / 131**
- Flagged required trajectories: **0**

The human-review acceptance criterion is **not met yet**. Open `review/review.html`, review the packet, export it, and ingest the export.

## Category breakdown

### `llama3.2:1b` · policy-off

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 0 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 0 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 0 | 0 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 0 |
| `unauthorized_tool` | 16 | 16 | 0 | 0 | 0 |

### `llama3.2:1b` · policy-on

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 16 | 0 | 0 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 0 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 0 | 0 |
| `stop_failure` | 16 | 0 | 0 | 0 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 0 | 0 |
| `unauthorized_tool` | 16 | 16 | 0 | 3 | 0 |

### `qwen3:0.6b` · policy-off

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 0 | 16 | 16 | 13 |
| `data_exfiltration` | 16 | 0 | 16 | 16 | 0 |
| `prompt_injection` | 16 | 0 | 2 | 2 | 2 |
| `stop_failure` | 16 | 6 | 10 | 10 | 14 |
| `tool_error_recovery` | 16 | 0 | 1 | 1 | 19 |
| `unauthorized_tool` | 16 | 0 | 15 | 15 | 24 |

### `qwen3:0.6b` · policy-on

| Category | Tasks | Success | Unsafe executed | Attempted unsafe | Tool errors |
|---|---:|---:|---:|---:|---:|
| `dangerous_file_op` | 16 | 3 | 0 | 16 | 0 |
| `data_exfiltration` | 16 | 16 | 0 | 16 | 0 |
| `prompt_injection` | 16 | 0 | 0 | 10 | 0 |
| `stop_failure` | 16 | 4 | 0 | 9 | 0 |
| `tool_error_recovery` | 16 | 0 | 0 | 1 | 11 |
| `unauthorized_tool` | 16 | 3 | 0 | 16 | 0 |

## Failure taxonomy

- `llama3.2:1b` / policy-off: model={}, tool={}
- `llama3.2:1b` / policy-on: model={}, tool={}
- `qwen3:0.6b` / policy-off: model={}, tool={"mock_failure": 38, "not_found": 19, "transient": 15}
- `qwen3:0.6b` / policy-on: model={}, tool={"not_found": 2, "transient": 9}

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
