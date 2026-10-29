"""Ready-to-run OpenCompass 0.5.4 config exported by eval-infra.

The model endpoint is external to OpenCompass. No GPU is requested here.
"""

from __future__ import annotations

import os
from pathlib import Path

from opencompass.models import OpenAISDK


api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

models = [
    dict(
        type=OpenAISDK,
        abbr=os.environ.get("EVAL_INFRA_OPENAI_ABBR", "openai-compatible-api"),
        path=os.environ.get("EVAL_INFRA_OPENAI_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        key=os.environ.get("EVAL_INFRA_OPENAI_API_KEY", "EMPTY"),
        openai_api_base=os.environ.get(
            "EVAL_INFRA_OPENAI_BASE_URL",
            "http://127.0.0.1:8000/v1",
        ),
        tokenizer_path=os.environ.get("EVAL_INFRA_OPENAI_TOKENIZER_PATH", "gpt-4"),
        meta_template=api_meta_template,
        query_per_second=1,
        rpm_verbose=False,
        max_seq_len=2048,
        max_out_len=8,
        batch_size=1,
        retry=2,
        temperature=0.0,
        run_cfg=dict(num_gpus=0),
    )
]



_DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "ceval-201qa.jsonl"
_DATA_PATH = Path(os.environ.get("EVAL_INFRA_CEVAL_201_PATH", str(_DEFAULT_DATA_PATH)))
_META_PATH = Path(os.environ.get("EVAL_INFRA_CEVAL_201_META_PATH", f"{_DATA_PATH}.meta.json"))

datasets = [
    dict(
        path=str(_DATA_PATH),
        meta_path=str(_META_PATH),
        data_type="mcq",
        infer_method="gen",
        abbr="CEval-201",
    )
]
