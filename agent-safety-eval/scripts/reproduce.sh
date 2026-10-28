#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=../.venv/bin/python

"$PY" -m pytest -q tests
./scripts/rerun.sh
