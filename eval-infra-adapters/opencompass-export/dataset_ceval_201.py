"""Frozen 201-row C-Eval quick-evaluation config for OpenCompass."""

from __future__ import annotations

import os
from pathlib import Path

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
