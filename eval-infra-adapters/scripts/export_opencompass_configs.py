#!/usr/bin/env python3
"""Export OpenCompass configs and write JSON, Markdown and HTML evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_infra_adapters.opencompass import (  # noqa: E402
    DEFAULT_CEVAL_META_SOURCE,
    DEFAULT_CEVAL_SOURCE,
    DEFAULT_EVALOPS_SUMMARY,
    DEFAULT_TASKS_SOURCE,
    export_opencompass_bundle,
    render_opencompass_html,
    render_opencompass_markdown,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--output-dir", default=str(ROOT / "opencompass-export"))
    root.add_argument("--evidence-dir", default=str(ROOT / "evidence"))
    root.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    root.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    root.add_argument("--api-key-env", default="EVAL_INFRA_OPENAI_API_KEY")
    root.add_argument("--allow-non-loopback-endpoint", action="store_true")
    root.add_argument("--data-source", default=str(DEFAULT_CEVAL_SOURCE))
    root.add_argument("--meta-source", default=str(DEFAULT_CEVAL_META_SOURCE))
    root.add_argument("--tasks-source", default=str(DEFAULT_TASKS_SOURCE))
    root.add_argument("--evalops-summary", default=str(DEFAULT_EVALOPS_SUMMARY))
    return root


def main() -> int:
    args = parser().parse_args()
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report = export_opencompass_bundle(
        Path(args.output_dir),
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        allow_non_loopback_endpoint=args.allow_non_loopback_endpoint,
        data_source=Path(args.data_source),
        meta_source=Path(args.meta_source),
        tasks_source=Path(args.tasks_source),
        evalops_summary=Path(args.evalops_summary),
    )
    (evidence_dir / "opencompass-export.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "opencompass-export.md").write_text(
        render_opencompass_markdown(report),
        encoding="utf-8",
    )
    (evidence_dir / "opencompass-export.html").write_text(
        render_opencompass_html(report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
