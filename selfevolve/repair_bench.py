"""Execution-based, fixed code-repair benchmark for SelfEvolve.

The benchmark is deliberately separate from ``RecursiveSelfImprover``.  Its
fixtures are immutable in source control and hidden tests are never supplied
to a patch-generating model.  A passing public test alone never counts as a
resolved task.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Callable

from selfevolve.self_modification import OllamaBackend


@dataclasses.dataclass(frozen=True)
class RepairTask:
    """One frozen, one-file repair task with separate public/hidden checks."""

    id: str
    category: str
    instruction: str
    source: str
    public_test: str
    hidden_test: str


def _case(
    task_id: str, category: str, instruction: str, source: str,
    public: str, hidden: str,
) -> RepairTask:
    return RepairTask(task_id, category, instruction, source, public, hidden)


# These fixtures intentionally stay compact: each task isolates one repair
# decision and uses no dependency beyond the standard library.
TASKS: tuple[RepairTask, ...] = (
    _case("exception_01", "exception", "Return an empty string when value is None.",
          "def normalize(value):\n    return value.strip().lower()\n",
          "from target import normalize\nassert normalize(' Hi ') == 'hi'\n",
          "from target import normalize\nassert normalize(None) == ''\n"),
    _case("exception_02", "exception", "Read port from config, defaulting to 8000 for a missing key.",
          "def port(config):\n    return int(config['port'])\n",
          "from target import port\nassert port({'port': '9000'}) == 9000\n",
          "from target import port\nassert port({}) == 8000\n"),
    _case("exception_03", "exception", "Return None instead of raising for a zero divisor.",
          "def ratio(total, count):\n    return total / count\n",
          "from target import ratio\nassert ratio(9, 3) == 3\n",
          "from target import ratio\nassert ratio(1, 0) is None\n"),
    _case("exception_04", "exception", "Parse an optional timeout; malformed or absent values default to 30.",
          "def timeout(config):\n    return int(config['timeout'])\n",
          "from target import timeout\nassert timeout({'timeout': '15'}) == 15\n",
          "from target import timeout\nassert timeout({}) == 30\nassert timeout({'timeout': 'bad'}) == 30\n"),
    _case("exception_05", "exception", "Return an empty path for None, otherwise remove one trailing slash.",
          "def clean_path(path):\n    return path.rstrip('/')\n",
          "from target import clean_path\nassert clean_path('/api/') == '/api'\n",
          "from target import clean_path\nassert clean_path(None) == ''\n"),
    _case("boundary_01", "boundary", "Clamp value inclusively between low and high, even when the bounds are reversed.",
          "def clamp(value, low, high):\n    return max(low, min(value, high))\n",
          "from target import clamp\nassert clamp(5, 0, 10) == 5\n",
          "from target import clamp\nassert clamp(5, 10, 0) == 5\n"),
    _case("boundary_02", "boundary", "Split items into chunks; size must be positive and invalid sizes return an empty list.",
          "def chunks(items, size):\n    return [items[i:i + size] for i in range(0, len(items), size)]\n",
          "from target import chunks\nassert chunks([1, 2, 3], 2) == [[1, 2], [3]]\n",
          "from target import chunks\nassert chunks([1], 0) == []\nassert chunks([1], -1) == []\n"),
    _case("boundary_03", "boundary", "Return up to n largest values; negative n means no values.",
          "def top_n(values, n):\n    return sorted(values, reverse=True)[:n]\n",
          "from target import top_n\nassert top_n([2, 7, 1], 2) == [7, 2]\n",
          "from target import top_n\nassert top_n([2, 7, 1], -1) == []\n"),
    _case("boundary_04", "boundary", "Return the statistical median; empty input returns None.",
          "def median(values):\n    values = sorted(values)\n    return values[len(values) // 2]\n",
          "from target import median\nassert median([3, 1, 2]) == 2\n",
          "from target import median\nassert median([1, 4]) == 2.5\nassert median([]) is None\n"),
    _case("boundary_05", "boundary", "Generate retry delays starting at one second; a nonpositive retry count produces no delays.",
          "def delays(retries):\n    return [2 ** i for i in range(abs(retries))]\n",
          "from target import delays\nassert delays(3) == [1, 2, 4]\n",
          "from target import delays\nassert delays(-2) == []\n"),
    _case("state_01", "state", "Only persist a setting when its validator accepts it.",
          "def apply_setting(state, key, value, valid):\n    state[key] = value\n    return valid(value)\n",
          "from target import apply_setting\ns = {}\nassert apply_setting(s, 'x', 2, lambda v: v > 0)\nassert s == {'x': 2}\n",
          "from target import apply_setting\ns = {'x': 1}\nassert not apply_setting(s, 'x', -1, lambda v: v > 0)\nassert s == {'x': 1}\n"),
    _case("state_02", "state", "Append an item only if it is not already present and return whether it was added.",
          "def add_unique(items, item):\n    items.append(item)\n    return item not in items[:-1]\n",
          "from target import add_unique\na = []\nassert add_unique(a, 'x')\nassert a == ['x']\n",
          "from target import add_unique\na = ['x']\nassert not add_unique(a, 'x')\nassert a == ['x']\n"),
    _case("state_03", "state", "Transfer money only when funds are sufficient; otherwise leave both balances unchanged.",
          "def transfer(accounts, source, target, amount):\n    accounts[source] -= amount\n    if accounts[source] < 0:\n        return False\n    accounts[target] += amount\n    return True\n",
          "from target import transfer\na = {'a': 5, 'b': 1}\nassert transfer(a, 'a', 'b', 3)\nassert a == {'a': 2, 'b': 4}\n",
          "from target import transfer\na = {'a': 2, 'b': 1}\nassert not transfer(a, 'a', 'b', 3)\nassert a == {'a': 2, 'b': 1}\n"),
    _case("state_04", "state", "Reserve a positive quantity only when enough inventory is available.",
          "def reserve(stock, sku, quantity):\n    stock[sku] -= quantity\n    return stock[sku] >= 0\n",
          "from target import reserve\ns = {'a': 4}\nassert reserve(s, 'a', 2)\nassert s == {'a': 2}\n",
          "from target import reserve\ns = {'a': 1}\nassert not reserve(s, 'a', 2)\nassert s == {'a': 1}\nassert not reserve(s, 'a', 0)\n"),
    _case("state_05", "state", "Set status only to pending, running, or done; invalid values must not change state.",
          "def set_status(state, status):\n    state['status'] = status\n    return True\n",
          "from target import set_status\ns = {}\nassert set_status(s, 'running')\nassert s['status'] == 'running'\n",
          "from target import set_status\ns = {'status': 'pending'}\nassert not set_status(s, 'bad')\nassert s == {'status': 'pending'}\n"),
    _case("tool_01", "tool_error", "Return None when a client call times out; otherwise return its value.",
          "def fetch(client, request):\n    return client.send(request)\n",
          "from target import fetch\nclass C:\n    def send(self, request): return 'ok'\nassert fetch(C(), 'x') == 'ok'\n",
          "from target import fetch\nclass C:\n    def send(self, request): raise TimeoutError()\nassert fetch(C(), 'x') is None\n"),
    _case("tool_02", "tool_error", "Return None when a client get operation raises OSError.",
          "def safe_get(client, url):\n    return client.get(url)\n",
          "from target import safe_get\nclass C:\n    def get(self, url): return {'ok': True}\nassert safe_get(C(), 'x') == {'ok': True}\n",
          "from target import safe_get\nclass C:\n    def get(self, url): raise OSError('offline')\nassert safe_get(C(), 'x') is None\n"),
    _case("tool_03", "tool_error", "Return a response result when present; missing results return None.",
          "def result(response):\n    return response['result']\n",
          "from target import result\nassert result({'result': 3}) == 3\n",
          "from target import result\nassert result({}) is None\n"),
    _case("tool_04", "tool_error", "Run every job, collecting successful results while skipping failed jobs.",
          "def run_batch(run, jobs):\n    return [run(job) for job in jobs]\n",
          "from target import run_batch\nassert run_batch(lambda x: x * 2, [1, 2]) == [2, 4]\n",
          "from target import run_batch\ndef run(x):\n    if x == 2: raise RuntimeError()\n    return x\nassert run_batch(run, [1, 2, 3]) == [1, 3]\n"),
    _case("tool_05", "tool_error", "Parse a tool response. Non-dict and missing output responses return None.",
          "def output(response):\n    return response.get('output')\n",
          "from target import output\nassert output({'output': 'ok'}) == 'ok'\n",
          "from target import output\nassert output(None) is None\nassert output({}) is None\n"),
)


def _run_test(source: str, test: str) -> dict[str, object]:
    """Run a fixture test in a fresh directory; harness failures are failures."""
    with tempfile.TemporaryDirectory(prefix="selfevolve_repair_") as tmp:
        root = pathlib.Path(tmp)
        (root / "target.py").write_text(source, encoding="utf-8")
        (root / "check.py").write_text(test, encoding="utf-8")
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "check.py"], cwd=root, capture_output=True,
                text=True, timeout=20,
            )
            return {
                "passed": result.returncode == 0,
                "stderr": result.stderr[-1000:],
                "duration_s": round(time.monotonic() - started, 3),
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "stderr": "TIMEOUT", "duration_s": 20.0}
        except OSError as exc:
            return {"passed": False, "stderr": f"HARNESS_ERROR: {exc}", "duration_s": 0.0}


def _ollama_candidate(
    task: RepairTask, backend: OllamaBackend,
    extra_fields: dict[str, object] | None = None,
) -> tuple[str | None, str]:
    prompt = (
        "You are fixing one small Python module. " + task.instruction + "\n"
        "Return only the complete corrected target.py in a Python code fence. "
        "Keep the public function signature and use only the standard library.\n\n"
        "```python\n" + task.source + "\n```"
    )
    response = backend.generate(
        prompt, max_tokens=2048, temperature=0.0, seed=42,
        extra_fields=extra_fields,
    )
    if not response:
        return None, "provider_no_output"
    candidate = OllamaBackend._extract_code(response)
    return candidate, "candidate" if candidate is not None else "invalid_python_output"


def _heuristic_candidate(task: RepairTask) -> str | None:
    """Current static heuristics cannot infer fixture behavior, so no patch.

    This is deliberately an honest baseline for the existing generic
    analyzer: it may propose style/safety edits but has no task-level oracle.
    """
    del task
    return None


def _balanced_smoke_tasks() -> tuple[RepairTask, ...]:
    """Return ten fixed tasks distributed 3/3/2/2 across categories."""
    order = (0, 5, 10, 15, 1, 6, 11, 16, 2, 7)
    return tuple(TASKS[index] for index in order)


def run_benchmark(
    arm: str, model: str = "qwen2.5-coder:7b", smoke: bool = False,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one frozen arm and return evidence suitable for JSON persistence."""
    if arm not in {"no_change", "heuristic", "ollama"}:
        raise ValueError("arm must be no_change, heuristic, or ollama")
    backend = OllamaBackend(model=model) if arm == "ollama" else None
    if backend is not None and not backend.is_available():
        raise RuntimeError("Ollama backend is unavailable; no result was produced")

    selected_tasks = _balanced_smoke_tasks() if smoke else TASKS
    cases: list[dict[str, object]] = []
    for task in selected_tasks:
        started = time.monotonic()
        candidate: str | None = None
        error = ""
        if arm == "heuristic":
            candidate = _heuristic_candidate(task)
            candidate_status = "no_task_oracle"
        elif backend is not None:
            candidate, candidate_status = _ollama_candidate(
                task, backend, extra_fields=extra_fields,
            )
        else:
            candidate_status = "not_requested"
        source = candidate if candidate is not None else task.source
        try:
            ast.parse(source)
            syntax_valid = True
        except SyntaxError as exc:
            syntax_valid = False
            error = f"SYNTAX_ERROR: {exc.msg}"
        public = _run_test(source, task.public_test) if syntax_valid else {"passed": False, "stderr": error}
        hidden = _run_test(source, task.hidden_test) if syntax_valid else {"passed": False, "stderr": error}
        cases.append({
            "id": task.id,
            "category": task.category,
            "candidate_generated": candidate is not None,
            "candidate_status": candidate_status,
            "syntax_valid": syntax_valid,
            "public_passed": public["passed"],
            "hidden_passed": hidden["passed"],
            "resolved": bool(public["passed"] and hidden["passed"] and syntax_valid),
            "public_stderr": public.get("stderr", ""),
            "hidden_stderr": hidden.get("stderr", ""),
            "duration_s": round(time.monotonic() - started, 3),
        })
    resolved = sum(bool(case["resolved"]) for case in cases)
    return {
        "schema_version": 1,
        "arm": arm,
        "model": model if arm == "ollama" else None,
        "settings": {
            "temperature": 0.0, "seed": 42, "max_tokens": 2048,
            "extra_fields": extra_fields,
        } if arm == "ollama" else None,
        "task_count": len(selected_tasks),
        "resolved_count": resolved,
        "resolved_rate": resolved / len(selected_tasks),
        "cases": cases,
        "limitations": "Twenty tasks are an exploratory gate, not a general code-repair claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed SelfEvolve repair benchmark")
    parser.add_argument("--arm", choices=("no_change", "heuristic", "ollama"), required=True)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--smoke", action="store_true", help="Run the fixed balanced ten-task smoke suite")
    parser.add_argument("--field", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Extra top-level field for /api/generate; repeatable, e.g. --field think=false")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    extra_fields: dict[str, object] = {}
    for item in args.field:
        if "=" not in item:
            parser.error(f"--field expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        extra_fields[key] = _parse_field_value(value)
    result = run_benchmark(
        args.arm, args.model, smoke=args.smoke,
        extra_fields=extra_fields or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("arm", "task_count", "resolved_count", "resolved_rate")}))


def _parse_field_value(value: str) -> object:
    """Parse a CLI field value into bool/int/float, falling back to text."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
