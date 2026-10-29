# OpenCompass 0.5.4 Export

> 本地安全评测，不是生产安全系统。 Local contract export, not a GPU run or an
> official C-Eval leaderboard result.

This directory contains configs pinned to the public OpenCompass 0.5.4
`OpenAISDK` model interface and its `CustomDataset` quick-evaluation protocol.
The dataset is the frozen 201-row public C-Eval validation subset.

## Files

- `eval-infra-openai-api.py`: combined model and dataset config.
- `model_openai_api.py`: standalone model config.
- `dataset_ceval_201.py`: standalone dataset config.
- `data/ceval-201qa.jsonl`: 201 rows, SHA-256
  `a76b144743ebdff4cb45908a6e7d23253c84e7bd1fc5321ade2eeaf09fc2a802`.
- `data/ceval-201qa.jsonl.meta.json`: `mcq` + `gen` metadata.
- `env.example`: non-secret runtime parameters.
- `official-api-contract.json`: pinned official source/interface snapshot.

## Run

The OpenAI-compatible endpoint must already be running. The default endpoint is
`http://127.0.0.1:8000/v1` and the default served model name is `Qwen/Qwen2.5-1.5B-Instruct`.

```bash
export EVAL_INFRA_OPENAI_API_KEY=EMPTY
opencompass opencompass-export/eval-infra-openai-api.py \
  --work-dir opencompass-export/runs/ceval-201 \
  --debug
```

For a config-only check:

```bash
opencompass opencompass-export/eval-infra-openai-api.py --dry-run --debug
```

## Safety boundary

- Exporting and config checks do not start a model, call an endpoint, or use a GPU.
- The default endpoint is loopback-only. A non-loopback endpoint requires the
  explicit `--allow-non-loopback-endpoint` exporter flag and separate authorization.
- API credentials are read only from `EVAL_INFRA_OPENAI_API_KEY`; no secret is exported.
- This is a 201-row validation subset, not the official C-Eval test set.
- No external OpenCompass execution, GPU execution, throughput, production SLO,
  or leaderboard result is claimed by this export.
