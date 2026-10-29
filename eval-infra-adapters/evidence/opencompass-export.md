# OpenCompass Export Contract

> **本地安全评测，不是生产安全系统。 Local contract export, not a production security system.**

- OpenCompass API snapshot: **0.5.4**
- Export status: **READY_TO_RUN_EXTERNALLY**
- Official interface contract: **PASS_STATIC** (`pinned_official_source_static`)
- Runtime CPU contract: **NOT_RUN**
- Dataset: **201 rows**, `mcq` / `gen`
- Dataset SHA-256: `a76b144743ebdff4cb45908a6e7d23253c84e7bd1fc5321ade2eeaf09fc2a802`
- Endpoint scope: **loopback**
- API key exported: **False**
- Official OpenCompass execution: **False**
- GPU execution / throughput: **False / False**
- GPU acceptance: **BLOCKED_ENVIRONMENT** (nvidia-smi unavailable; no GPU experiment requested or executed)

## Exported files

| File | SHA-256 |
|---|---|
| `README.md` | `e1e103180f8708cf8b1b2691cbe141546b6d9f57db7d877afeb5f692a5bc339e` |
| `data/ceval-201qa.jsonl` | `a76b144743ebdff4cb45908a6e7d23253c84e7bd1fc5321ade2eeaf09fc2a802` |
| `data/ceval-201qa.jsonl.meta.json` | `eef85425a5892db89bb1228b449e51eae0b98bf8cb825c04b5982404356575a3` |
| `dataset_ceval_201.py` | `173595bf66be35cb839e1378b1a16bfeed312ca132d8d498d0294efe0994d794` |
| `env.example` | `281d4d394c78934f6a25f5d65a1e6c9d95df314c17a7975f949f30b8b710e128` |
| `eval-infra-openai-api.py` | `f5a1c63c3de518cd68cc6f6c6bfccd831cf4f7bc6126eb22d1287a135ccbd6d9` |
| `model_openai_api.py` | `f4c248848ecb003e7222a33d4700b347ae17e3ec2db32bf0fc53816fea7d3613` |
| `official-api-contract.json` | `17e402d3259ae09305e09d927867691c31d5e8d002bc44e1fbd79ecd22d2d0c6` |

## Preserved existing 201-row CPU evidence

Source: `evalops/evidence-20261027/summary.json`

| C | Success | Correct |
|---:|---:|---:|
| 1 | 201/201 | 108 |
| 4 | 201/201 | 107 |
| 8 | 201/201 | 108 |

These are existing local CPU Ollama measurements. They are preserved and
re-hashed, not rerun by this offline export.

## Safety boundary

- The exporter does not start a model, call the endpoint, install OpenCompass,
  or request GPU resources.
- The default endpoint must be loopback. Remote endpoints require an explicit
  override and separate authorization.
- Credentials are read only from `EVAL_INFRA_OPENAI_API_KEY` and are never exported.
- Static source inspection is not an OpenCompass execution or leaderboard result.
- The 201-row public validation subset is not the official C-Eval test set.

## Reproduce

```bash
cd eval-infra-adapters
make reproduce
```
