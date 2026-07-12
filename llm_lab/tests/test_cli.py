"""Tests for cli.py — Typer CLI app."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from llm_lab import cli

_cli_runner = CliRunner()


def test_version_displays():
    result = _cli_runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_no_command_shows_message():
    result = _cli_runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "llm-lab" in result.stdout


def test_help_shows_commands():
    result = _cli_runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "history", "export", "compare", "serve"):
        assert cmd in result.stdout


def test_run_with_no_args_fails():
    result = _cli_runner.invoke(cli.app, ["run"])
    assert result.exit_code != 0


def test_dry_run_flag_prints_and_returns():
    result = _cli_runner.invoke(cli.app, ["run", "--dry-run", "test goal"])
    assert result.exit_code == 0
    assert "DRY-RUN" in result.stdout


def test_run_json_output(monkeypatch):
    from llm_lab import runner
    fake_result = {
        "intent_id": "mock-run",
        "all_passed": True,
        "model": "gpt-4o",
        "plan_template": None,
        "steps_detail": [],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }
    monkeypatch.setattr(runner, "run_plan", lambda _goal=None, _model=None, _verifier=None: fake_result)

    result = _cli_runner.invoke(cli.app, ["run", "--json", "test message"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["intent_id"] == "mock-run"


def test_run_human_readable(monkeypatch):
    from llm_lab import runner
    fake_result = {
        "intent_id": "mock-run",
        "all_passed": True,
        "model": "gpt-4o",
        "plan_template": None,
        "steps_detail": [],
        "total_tokens": 100,
        "total_cost_usd": 0.005,
    }
    monkeypatch.setattr(runner, "run_plan", lambda _goal=None, _model=None, _verifier=None: fake_result)

    result = _cli_runner.invoke(cli.app, ["run", "test message"])
    assert result.exit_code == 0
    assert "PASS" in result.stdout or "mock-run" in result.stdout


def test_history_empty(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_hist.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    result = _cli_runner.invoke(cli.app, ["history"])
    assert result.exit_code == 0
    assert "No runs" in result.stdout or "no runs" in result.stdout.lower()

    if tmp.exists():
        tmp.unlink()


def test_history_json(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_hist2.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="h-test", seq=1, action="call", model="gpt-4o"))

    result = _cli_runner.invoke(cli.app, ["history", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data) >= 1
    assert data[0]["intent_id"] == "h-test"

    if tmp.exists():
        tmp.unlink()


@pytest.mark.timeout(15)
def test_compare_mocked(monkeypatch):
    from llm_lab import runner
    def fake_compare(goal, model_a=None, model_b=None):
        return {
            "summary": {"winner": "tie", "cost_delta": 0.0, "token_delta": 0},
            "model_a": {"model": "gpt-4o", "all_passed": True, "total_tokens": 10, "total_cost_usd": 0.001, "steps": []},
            "model_b": {"model": "gpt-4o-mini", "all_passed": True, "total_tokens": 5, "total_cost_usd": 0.0005, "steps": []},
        }
    monkeypatch.setattr(runner, "compare", fake_compare)
    result = _cli_runner.invoke(cli.app, ["compare", "test goal"])
    assert result.exit_code == 0


@pytest.mark.timeout(15)
def test_compare_json(monkeypatch):
    from llm_lab import runner
    def fake_compare(goal, model_a=None, model_b=None):
        return {
            "summary": {"winner": "tie", "cost_delta": 0.0, "token_delta": 0},
            "model_a": {"model": "gpt-4o", "all_passed": True, "total_tokens": 10, "total_cost_usd": 0.001, "steps": []},
            "model_b": {"model": "gpt-4o-mini", "all_passed": True, "total_tokens": 5, "total_cost_usd": 0.0005, "steps": []},
        }
    monkeypatch.setattr(runner, "compare", fake_compare)

    result = _cli_runner.invoke(cli.app, ["compare", "--json", "test goal"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["summary"]["winner"] == "tie"


def test_export_requires_intent_id():
    result = _cli_runner.invoke(cli.app, ["export"])
    assert result.exit_code != 0


def test_export_with_data(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_exp.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="exp-test", seq=1, action="call"))

    result = _cli_runner.invoke(cli.app, ["export", "exp-test"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["intent_id"] == "exp-test"
    assert len(data["events"]) == 1

    if tmp.exists():
        tmp.unlink()


def test_export_to_file(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_exp2.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="exp-file", seq=1, action="call"))

    out = Path("_test_export.json")
    if out.exists():
        out.unlink()

    result = _cli_runner.invoke(cli.app, ["export", "exp-file", "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.exists()

    if out.exists():
        out.unlink()
    if tmp.exists():
        tmp.unlink()


def test_serve_accepts_flags():
    result = _cli_runner.invoke(cli.app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.stdout


# ── cli.py uncovered-branch coverage ─────────────────────────────────────


def test_run_with_steps_detail(monkeypatch):
    from llm_lab import runner

    fake_result = {
        "intent_id": "mock-detail",
        "all_passed": True,
        "model": "gpt-4o",
        "plan_template": "eval-model",
        "steps_detail": [
            {
                "action": "test",
                "verdict": {"label": "pass"},
                "tokens": 50,
                "cost": 0.001,
                "output": "ok",
            }
        ],
        "total_tokens": 50,
        "total_cost_usd": 0.001,
    }
    monkeypatch.setattr(runner, "run_plan", lambda _goal=None, _model=None, _verifier=None: fake_result)

    result = _cli_runner.invoke(cli.app, ["run", "test goal"])
    assert result.exit_code == 0
    assert "mock-detail" in result.stdout
    assert "PASS" in result.stdout


def test_compare_with_steps(monkeypatch):
    from llm_lab import runner

    def fake_compare(goal, model_a=None, model_b=None):
        return {
            "summary": {"winner": "a", "cost_delta": 0.001, "token_delta": 5},
            "model_a": {
                "model": "gpt-4o",
                "all_passed": True,
                "total_tokens": 10,
                "total_cost_usd": 0.001,
                "steps": [{"verdict": {"label": "pass"}, "action": "step1", "output": "ok"}],
            },
            "model_b": {
                "model": "gpt-4o-mini",
                "all_passed": False,
                "total_tokens": 5,
                "total_cost_usd": 0.0005,
                "steps": [{"verdict": {"label": "fail"}, "action": "step1", "output": "nope"}],
            },
        }

    monkeypatch.setattr(runner, "compare", fake_compare)
    result = _cli_runner.invoke(cli.app, ["compare", "test goal"])
    assert result.exit_code == 0


def test_history_with_data(monkeypatch):
    from llm_lab import db as _local_db

    tmp = Path("_test_cli_hist3.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio

    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="h-display", seq=1, action="call", model="gpt-4o"))

    result = _cli_runner.invoke(cli.app, ["history"])
    assert result.exit_code == 0
    assert "h-display" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_entry_function(monkeypatch):
    """cli.py line 206: entry() calls app()."""
    monkeypatch.setattr("sys.argv", ["llm-lab", "--help"])
    with pytest.raises(SystemExit):
        cli.entry()


def test_main_guard():
    """cli.py line 210: __main__ guard path is importable."""
    assert hasattr(cli, "entry")
    assert callable(cli.entry)


def test_serve_command(monkeypatch):

    mock_uvicorn = MagicMock()
    monkeypatch.setitem(sys.modules, "uvicorn", mock_uvicorn)
    result = _cli_runner.invoke(cli.app, ["serve", "--port", "9999"])
    assert result.exit_code == 0


def test_main_guard_run(monkeypatch):
    """cli.py line 210: __name__ == '__main__' entry block."""
    import runpy

    monkeypatch.setattr("sys.argv", ["llm-lab", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(Path(__file__).parent.parent / "cli.py"), run_name="__main__")


# ── export subcommand format branches ──────────────────────────────────────


def _make_export_fake_db(monkeypatch, name: str, rows: int = 1):
    from llm_lab import db as _local_db

    tmp = Path(f"_test_cli_{name}.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio

    asyncio.run(_local_db.init_db())
    for i in range(rows):
        asyncio.run(_local_db.append_event(intent_id="exp-fmt", seq=i + 1, action="call"))
    return tmp


def test_export_no_data(monkeypatch):
    """fmt=json, rows empty → "No data found" + exit code 1."""
    tmp = _make_export_fake_db(monkeypatch, "exp_no_data", rows=0)
    result = _cli_runner.invoke(cli.app, ["export", "exp-fmt"])
    assert result.exit_code == 1
    assert "No data found" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_export_csv(monkeypatch):
    """fmt=csv → CSV text output."""
    tmp = _make_export_fake_db(monkeypatch, "exp_csv")
    result = _cli_runner.invoke(cli.app, ["export", "exp-fmt", "--format", "csv"])
    assert result.exit_code == 0
    assert "id,intent_id" in result.stdout
    assert "exp-fmt" in result.stdout

    if tmp.exists():
        tmp.unlink()


@pytest.mark.skipif(not __import__("importlib").util.find_spec("openpyxl"), reason="openpyxl not installed")
def test_export_xlsx_fails_without_output(monkeypatch):
    """fmt=xlsx without --output → "XLSX output requires --output"."""
    tmp = _make_export_fake_db(monkeypatch, "exp_xlsx_noout")
    result = _cli_runner.invoke(cli.app, ["export", "exp-fmt", "--format", "xlsx"])
    assert result.exit_code == 0
    assert "XLSX output requires" in result.stdout

    if tmp.exists():
        tmp.unlink()


@pytest.mark.skipif(not __import__("importlib").util.find_spec("openpyxl"), reason="openpyxl not installed")
def test_export_xlsx_to_file(monkeypatch):
    """fmt=xlsx with --output → writes file."""
    tmp = _make_export_fake_db(monkeypatch, "exp_xlsx_out")
    out = Path("_test_export.xlsx")
    result = _cli_runner.invoke(cli.app, ["export", "exp-fmt", "--format", "xlsx", "--output", str(out)])
    assert result.exit_code == 0
    assert "Exported to" in result.stdout
    assert out.exists()
    assert out.read_bytes()[:2] == b"PK"

    out.unlink(missing_ok=True)
    if tmp.exists():
        tmp.unlink()


def test_export_xlsx_runtime_error(monkeypatch):
    """export_xlsx raises RuntimeError → error message + exit code 1."""
    from unittest.mock import patch as _mock_patch

    tmp = _make_export_fake_db(monkeypatch, "exp_xlsx_err")
    with _mock_patch("llm_lab.cli.export_mod.export_xlsx", side_effect=RuntimeError("mock error")):
        result = _cli_runner.invoke(cli.app, ["export", "exp-fmt", "--format", "xlsx"])
    assert result.exit_code == 1
    assert "mock error" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_export_unknown_format(monkeypatch):
    """fmt=xyz → "Unknown format" + exit code 1."""
    tmp = _make_export_fake_db(monkeypatch, "exp_unknown")
    result = _cli_runner.invoke(cli.app, ["export", "exp-fmt", "--format", "xyz"])
    assert result.exit_code == 1
    assert "Unknown format" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_watch_breaks_on_keyboard_interrupt(monkeypatch):
    """watch — mock sleep to raise KeyboardInterrupt so the loop exits cleanly."""
    import time as _real_time
    call_count = 0
    def _breaking_sleep(_secs):
        nonlocal call_count
        call_count += 1
        raise KeyboardInterrupt()
    monkeypatch.setattr(_real_time, "sleep", _breaking_sleep)

    from llm_lab import runner as _runner
    monkeypatch.setattr(_runner, "run_plan", lambda goal=None, model=None, verifier=None: {
        "intent_id": "watch-test", "all_passed": True, "model": model or "gpt-4o",
        "plan_template": None, "steps_detail": [],
        "total_tokens": 0, "total_cost_usd": 0.0,
    })

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _cli_runner.invoke(cli.app, ["watch", "test goal", "--watch-dir", tmpdir, "--interval", "1"])
    assert result.exit_code == 130


def test_watch_detects_change_and_runs_plan(monkeypatch):
    """watch — create file, modify mtime mid-loop; covers _check() body and run path."""
    import os
    import time as _real_time

    class _Ctrl:
        def __init__(self):
            self.count = 0
            self.test_file = None

    ctrl = _Ctrl()

    def _controlled_sleep(_secs):
        ctrl.count += 1
        if ctrl.count == 1:
            # after first _check() has stored the file mtime, bump it
            if ctrl.test_file and os.path.isfile(ctrl.test_file):
                os.utime(ctrl.test_file, (2000000000, 2000000000))
            return
        raise KeyboardInterrupt()

    monkeypatch.setattr(_real_time, "sleep", _controlled_sleep)

    from llm_lab import runner as _runner
    monkeypatch.setattr(_runner, "run_plan", lambda goal=None, model=None, verifier=None: {
        "intent_id": "watch-change-test", "all_passed": True, "model": model or "gpt-4o",
        "plan_template": None, "steps_detail": [],
        "total_tokens": 0, "total_cost_usd": 0.0,
    })

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.py"
        p.write_text("print('hello')")
        os.utime(p, (1000000000, 1000000000))
        ctrl.test_file = str(p)
        result = _cli_runner.invoke(cli.app, ["watch", "test goal", "--watch-dir", tmpdir, "--interval", "1"])
    assert result.exit_code == 130
    assert "Change detected" in result.stdout


# ── report command ───────────────────────────────────────────────────────────


def test_report_no_data(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_report_nodata.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    result = _cli_runner.invoke(cli.app, ["report", "nonexistent"])
    assert result.exit_code == 1
    assert "No data" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_report_with_data(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_report_data.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(
        intent_id="r-test", seq=1, action="call", model="gpt-4o",
        cost_usd=0.001, detail="output text",
    ))

    result = _cli_runner.invoke(cli.app, ["report", "r-test"])
    assert result.exit_code == 0
    assert "r-test" in result.stdout
    assert "gpt-4o" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_report_to_file(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_report_file.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(
        intent_id="r-file", seq=1, action="call", model="gpt-4o",
        cost_usd=0.001,
    ))

    out = Path("_test_report.html")
    result = _cli_runner.invoke(cli.app, ["report", "r-file", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "r-file" in out.read_text(encoding="utf-8")

    out.unlink(missing_ok=True)
    if tmp.exists():
        tmp.unlink()


# ── watch command ────────────────────────────────────────────────────────────


def test_watch_command_help():
    result = _cli_runner.invoke(cli.app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--watch-dir" in result.stdout
    assert "--interval" in result.stdout


# ── diff command ─────────────────────────────────────────────────────────────


def test_diff_no_data_a(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_diff_a.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="b", seq=1, action="call", model="gpt-4o"))

    result = _cli_runner.invoke(cli.app, ["diff", "a", "b"])
    assert result.exit_code == 1
    assert "No data for a" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_diff_no_data_b(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_diff_b.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="a", seq=1, action="call", model="gpt-4o"))

    result = _cli_runner.invoke(cli.app, ["diff", "a", "b"])
    assert result.exit_code == 1
    assert "No data for b" in result.stdout

    if tmp.exists():
        tmp.unlink()


def test_diff_with_data(monkeypatch):
    from llm_lab import db as _local_db
    tmp = Path("_test_cli_diff_both.db")
    monkeypatch.setattr(_local_db, "DB_PATH", str(tmp))

    import asyncio
    asyncio.run(_local_db.init_db())
    asyncio.run(_local_db.append_event(intent_id="a", seq=1, action="call", model="gpt-4o", cost_usd=0.001))
    asyncio.run(_local_db.append_event(intent_id="b", seq=1, action="call", model="claude-3", cost_usd=0.002))

    result = _cli_runner.invoke(cli.app, ["diff", "a", "b"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.stdout
    assert "claude-3" in result.stdout

    if tmp.exists():
        tmp.unlink()


# ── run --preset option ──────────────────────────────────────────────────────


def test_run_with_preset_overrides_model(monkeypatch):
    from llm_lab import runner as _runner
    called_with = {}

    def capture(goal, model=None, verifier=None):
        called_with.update(goal=goal, model=model, verifier=verifier)
        return {
            "intent_id": "preset-test", "all_passed": True, "model": model,
            "plan_template": None, "steps_detail": [],
            "total_tokens": 0, "total_cost_usd": 0.0,
        }

    monkeypatch.setattr(_runner, "run_plan", capture)

    result = _cli_runner.invoke(cli.app, ["run", "--preset", "cheap", "test preset"])
    assert result.exit_code == 0, result.stdout
    assert "gpt-4o-mini" in result.stdout


def test_run_with_preset_json_output(monkeypatch):
    from llm_lab import runner as _runner
    monkeypatch.setattr(_runner, "run_plan", lambda goal=None, model=None, verifier=None: {
        "intent_id": "preset-json", "all_passed": True, "model": model,
        "plan_template": None, "steps_detail": [],
        "total_tokens": 0, "total_cost_usd": 0.0,
    })

    result = _cli_runner.invoke(cli.app, ["run", "--preset", "balanced", "--json", "test"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["intent_id"] == "preset-json"


def test_run_with_invalid_preset_errors(monkeypatch):
    result = _cli_runner.invoke(cli.app, ["run", "--preset", "nonexistent", "test goal"])
    assert result.exit_code != 0
