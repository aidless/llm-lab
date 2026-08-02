"""RecursiveSelfImprovement — the agent modifies its own source code.

This is the core of recursive self-improvement toward AGI:
  1. Read own source (introspection).
  2. Identify improvement opportunities via meta-cognition.
  3. Generate candidate patches.
  4. Test patches in a sandboxed environment.
  5. Promote successful patches to production.
  6. Loop — the improved agent is now better at improving itself.

Safety: All modifications are applied to a COPY of the source, tested,
and only promoted if they pass validation.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import typing as t
import urllib.request
import urllib.error

from .core import (
    ImprovementHypothesis,
    MetaCognition,
    SelfEvolvingAgent,
    Skill,
    TaskOutcome,
)


# ──────────────────────────────────────────────
#  Patch — a single atomic modification
# ──────────────────────────────────────────────


class Patch:
    """A single atomic change to a source file.

    Format: unified diff, but we store structured fields
    for analysis and rollback.
    """

    def __init__(self, file_path: str, old_code: str, new_code: str,
                 description: str = "", author: str = "selfevolve"):
        self.file_path = file_path
        self.old_code = old_code
        self.new_code = new_code
        self.description = description
        self.author = author
        self.created_at = time.time()
        self.id = hashlib.sha256(
            f"{file_path}:{old_code[:50]}:{time.time()}".encode()
        ).hexdigest()[:12]

    def to_unified_diff(self) -> str:
        return "".join(difflib.unified_diff(
            self.old_code.splitlines(keepends=True),
            self.new_code.splitlines(keepends=True),
            fromfile=self.file_path,
            tofile=self.file_path,
        ))

    def apply(self, root_dir: str = ".") -> bool:
        """Apply this patch to the filesystem. Returns True on success."""
        full_path = os.path.join(root_dir, self.file_path)
        if not os.path.exists(full_path):
            return False
        with open(full_path, encoding="utf-8") as f:
            current = f.read()
        if current != self.old_code:
            return False  # file changed since patch was generated
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(self.new_code)
        return True

    def revert(self, root_dir: str = ".") -> bool:
        """Revert this patch."""
        # DUPLICATE: blocks at L79 and L92 - extract to shared function

        # DUPLICATE: blocks at L79 and L94 - extract to shared function

        # DUPLICATE: blocks at L79 and L96 - extract to shared function

        # DUPLICATE: blocks at L79 and L98 - extract to shared function

        # DUPLICATE: blocks at L79 and L100 - extract to shared function

        # DUPLICATE: blocks at L79 and L102 - extract to shared function

        # DUPLICATE: blocks at L79 and L104 - extract to shared function

        # DUPLICATE: blocks at L79 and L106 - extract to shared function

        # DUPLICATE: blocks at L79 and L108 - extract to shared function

        # DUPLICATE: blocks at L79 and L110 - extract to shared function

        # DUPLICATE: blocks at L79 and L112 - extract to shared function

        # DUPLICATE: blocks at L79 and L114 - extract to shared function

        # DUPLICATE: blocks at L79 and L116 - extract to shared function

        # DUPLICATE: blocks at L79 and L118 - extract to shared function

        # DUPLICATE: blocks at L79 and L120 - extract to shared function

        # DUPLICATE: blocks at L79 and L122 - extract to shared function

        # DUPLICATE: blocks at L79 and L124 - extract to shared function

        # DUPLICATE: blocks at L79 and L126 - extract to shared function

        # DUPLICATE: blocks at L79 and L128 - extract to shared function

        # DUPLICATE: blocks at L79 and L130 - extract to shared function

        # DUPLICATE: blocks at L79 and L132 - extract to shared function

        full_path = os.path.join(root_dir, self.file_path)
        if not os.path.exists(full_path):
            return False
        with open(full_path, encoding="utf-8") as f:
            current = f.read()
        if current != self.new_code:
            return False
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(self.old_code)
        return True


# ──────────────────────────────────────────────
#  Sandbox — safe environment for testing patches
# ──────────────────────────────────────────────


class Sandbox:
    """A filesystem sandbox for testing code modifications.

    Creates a temporary copy of the agent's source files,
    applies the patch, runs tests, and reports results.
    No modifications to the real codebase until explicitly promoted.
    """

    def __init__(self, source_root: str = "."):
        self.source_root = pathlib.Path(source_root).resolve()
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._sandbox_path: pathlib.Path | None = None

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="selfevolve_sandbox_")
        self._sandbox_path = pathlib.Path(self._tmpdir.name)
        return self

    def __exit__(self, *args):
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._sandbox_path = None

    def prepare(self, files: list[str]) -> None:
        """Copy specified files into the sandbox, plus any test files."""
        assert self._sandbox_path is not None, "Sandbox not entered"
        for rel_path in files:
            src = self.source_root / rel_path
            if src.exists():
                dst = self._sandbox_path / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        tests_dir = self.source_root / "tests"
        if tests_dir.is_dir():
            for test_src in sorted(tests_dir.rglob("*.py")):
                rel = test_src.relative_to(self.source_root)
                dst = self._sandbox_path / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(test_src.read_text(encoding="utf-8"), encoding="utf-8")

    def apply_patch(self, patch: Patch) -> bool:
        """Apply a patch inside the sandbox."""
        assert self._sandbox_path is not None
        full = self._sandbox_path / patch.file_path
        if not full.exists():
            return False
        current = full.read_text(encoding="utf-8")
        if current != patch.old_code:
            return False
        full.write_text(patch.new_code, encoding="utf-8")
        return True

    def run_test(self, test_cmd: list[str] | None = None) -> dict:
        """Run tests in the sandbox and return results.

        Returns dict with: success, stdout, stderr, returncode, duration_s.
        """
        assert self._sandbox_path is not None
        if test_cmd is None:
            # Auto-detect test command
            test_cmd = self._detect_test_cmd()

        start = time.time()
        try:
            result = subprocess.run(
                test_cmd,
                cwd=str(self._sandbox_path),
                capture_output=True,
                text=True,
                timeout=60,
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "TIMEOUT (60s)",
                "returncode": -1,
                "duration_s": 60.0,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -2,
                "duration_s": time.time() - start,
            }

        duration = time.time() - start
        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration_s": round(duration, 2),
        }

    def _detect_test_cmd(self) -> list[str]:
        """Detect available test runner."""
        assert self._sandbox_path is not None
        # Check for test files or configs
        test_files = list(self._sandbox_path.rglob("test_*.py"))
        if test_files:
            # Prefer the stdlib standalone runner (pytest may not be installed).
            standalone = self._sandbox_path / "tests" / "test_selfevolve.py"
            if standalone.exists():
                return [sys.executable, str(standalone)]
            return [sys.executable, str(test_files[0])]

        # No tests found — patch passes if files exist (apply already succeeded).
        return [sys.executable, "-c", "pass"]


# ──────────────────────────────────────────────
#  CodeAnalyzer — static analysis of own code
# ──────────────────────────────────────────────


class CodeAnalyzer:
    """Analyzes the agent's own source code for improvement opportunities.

    Uses Python AST to detect:
    - Dead code (unused functions/variables)
    - Complexity hotspots (deep nesting, many branches)
    - Inefficient patterns (repeated computation)
    - Missing error handling
    - Opportunities for generalization
    """

    def __init__(self, source_root: str = "."):
        self.source_root = pathlib.Path(source_root)

    def analyze_module(self, module_path: str) -> dict:
        """Run all analyses on a single module.

        Returns a dict with findings keyed by severity.
        """
        full_path = self.source_root / module_path
        if not full_path.exists():
            return {"error": f"File not found: {module_path}"}

        source = full_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}

        findings = {
            "complexity": self._check_complexity(tree),
            "error_handling": self._check_error_handling(tree),
            "dead_code": self._check_dead_code(tree),
            "duplication": self._check_duplication(source),
            "generalization": self._check_generalization(tree),
            "todo_comments": self._check_todos(source),
        }
        return findings

    def _check_complexity(self, tree: ast.AST) -> list[dict]:
        """Find functions with high cyclomatic complexity."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._cyclomatic_complexity(node)
                if complexity > 8:
                    findings.append({
                        "type": "high_complexity",
                        "name": node.name,
                        "lineno": node.lineno,
                        "complexity": complexity,
                        "suggestion": "Consider splitting into smaller functions",
                    })
        return findings

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """Compute McCabe cyclomatic complexity for a function body."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                  ast.Assert, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _check_error_handling(self, tree: ast.AST) -> list[dict]:
    # NOTE(_check_error_handling): high complexity (16) - consider splitting

        """Find bare excepts, missing error handling, and None idioms."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    findings.append({
                        "type": "bare_except",
                        "lineno": node.lineno,
                        "suggestion": "Specify exception type instead of bare except",
                    })
            if isinstance(node, ast.FunctionDef) and not self._has_try(node):
                # Functions with potentially dangerous operations
                dangerous = ["open", "eval", "exec", "subprocess"]
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        if child.func.id in dangerous:
                            findings.append({
                                "type": "missing_try",
                                "name": node.name,
                                "lineno": node.lineno,
                                "suggestion": f"Wrap {child.func.id}() call in try/except",
                            })
        # Idiom check: == None / != None should be is None / is not None
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Eq, ast.NotEq)) and \
                            isinstance(comparator, ast.Constant) and \
                            comparator.value is None:
                        findings.append({
                            "type": "none_comparison",
                            "lineno": node.lineno,
                            "op": "==" if isinstance(op, ast.Eq) else "!=",
                            "suggestion": "Compare with 'is None' instead of '== None'",
                        })
        return findings

    def _has_try(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False

    def _check_dead_code(self, tree: ast.AST) -> list[dict]:
        """Find unused function definitions (simplified heuristic)."""
        defined = set()
        called = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)

        unused = defined - called - {"__init__", "main"}
        return [
            {
                "type": "possibly_unused",
                "name": name,
                "suggestion": "Remove or export via __all__",
            }
            for name in sorted(unused)
        ]

    def _check_duplication(self, source: str) -> list[dict]:
        """Find repeated code blocks (simplified)."""
        lines = source.splitlines()
        findings = []
        for length in [5, 10]:
            seen = {}
            for i in range(len(lines) - length + 1):
                block = "\n".join(lines[i:i + length])
                h = hashlib.md5(block.encode()).hexdigest()
                if h in seen:
                    findings.append({
                        "type": "duplicated_block",
                        "lineno_a": seen[h],
                        "lineno_b": i,
                        "lines": length,
                        "suggestion": "Extract duplicated block into a shared function",
                    })
                else:
                    seen[h] = i
        return findings[:5]  # limit to first 5 findings

    def _check_generalization(self, tree: ast.AST) -> list[dict]:
        """Find hard-coded values that could be parameters."""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if 0 <= node.value <= 100:
                    # Check if this is a configurable parameter
                    parent = self._find_parent_function(node, tree)
                    if parent:
                        findings.append({
                            "type": "magic_number",
                            "value": node.value,
                            "lineno": node.lineno,
                            "function": parent,
                            "suggestion": f"Replace {node.value} with a named constant or parameter",
                        })
        return findings[:10]

    def _find_parent_function(self, node: ast.AST, tree: ast.AST) -> str | None:
        """Find the enclosing function name for a node."""
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(n):
                    if child is node:
                        return n.name
        return None

    def _check_todos(self, source: str) -> list[dict]:
        """Find TODO/FIXME/HACK comments."""
        findings = []
        for i, line in enumerate(source.splitlines(), 1):
            for marker in ["TODO", "FIXME", "HACK", "XXX"]:
                if marker in line.upper() and "#" in line:
                    findings.append({
                        "type": "todo_comment",
                        "marker": marker,
                        "lineno": i,
                        "text": line.strip(),
                    })
        return findings


# ──────────────────────────────────────────────
#  OllamaBackend — optional LLM patch generation
# ──────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class OllamaBackend:
    """Minimal stdlib-only client for a local Ollama server.

    Used to generate candidate patches from static-analysis findings
    when ``--ollama`` is enabled. Every method fails soft: if Ollama
    is not running, not installed, or returns garbage, we return
    ``None`` and the pipeline falls back to heuristic patches.
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b",
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available: bool | None = None
        self._available_at: float = 0.0

    def _request(self, path: str, payload: dict | None = None) -> dict | None:
        """POST (or GET) a JSON request; returns parsed dict or None."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload else None
        req = urllib.request.Request(url, data=data, method="POST" if payload else "GET")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def is_available(self) -> bool:
        """Check the /api/tags endpoint; result cached for 30s."""
        now = time.time()
        if self._available is not None and now - self._available_at < 30.0:
            return self._available
        tags = self._request("/api/tags")
        self._available = isinstance(tags, dict) and "models" in tags
        self._available_at = now
        return self._available

    def generate(self, prompt: str, max_tokens: int = 256) -> str | None:
        """Run a completion; returns raw response text or None."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        resp = self._request("/api/generate", payload)
        if not resp or not resp.get("response"):
            return None
        return str(resp["response"]).strip()

    @staticmethod
    def _extract_code(response: str) -> str | None:
        """Pull code out of a model response (fenced or raw)."""
        text = response.strip()
        m = _FENCE_RE.search(text)
        candidate = m.group(1) if m else text
        # Strip stray leading/trailing markdown when no fence was found
        if not m:
            candidate = re.sub(r"^```[a-zA-Z]*\n", "", candidate)
            candidate = re.sub(r"\n```$", "", candidate)
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            return None

    @staticmethod
    def _enclosing_function(old_code: str,
                            finding: dict) -> tuple[str, int, int, int] | None:
        """Return (dedented function source, start, end, col_offset) for the
        innermost function containing the finding's line, or None."""
        try:
            tree = ast.parse(old_code)
        except SyntaxError:
            return None
        target = int(finding.get("lineno") or finding.get("lineno_a") or 1)
        best: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= target <= node.end_lineno:
                    if best is None or node.lineno > best.lineno:
                        best = node
        if best is None:
            return None
        lines = old_code.splitlines()
        start, end = best.lineno - 1, best.end_lineno
        pad = best.col_offset
        src = "\n".join(lines[start:end])
        if pad:
            src = "\n".join(
                l[pad:] if l.startswith(" " * pad) else l
                for l in src.splitlines()
            )
        if len(src.splitlines()) > 150:
            return None
        return src, start, end, pad

    @classmethod
    def _splice_function(cls, old_code: str, func_src: str,
                         start: int, end: int, pad: int) -> str | None:
        """Replace lines[start:end] with the rewritten (dedented) function."""
        lines = old_code.splitlines()
        body = [(" " * pad) + l if l else "" for l in func_src.splitlines()]
        new_source = "\n".join(lines[:start] + body + lines[end:])
        if new_source == "\n".join(lines):
            return None
        try:
            ast.parse(new_source)
            return new_source
        except SyntaxError:
            return None

    def suggest_patch(self, module_name: str, old_code: str,
                      finding: dict) -> str | None:
        """Ask the LLM for a minimal safe fix; returns new source or None.

        Large files are handled in region mode: only the function that
        contains the finding is sent to the model (small local models
        cannot fit a whole file), and the rewrite is spliced back at its
        original indentation. Small files use the whole-file mode. Output
        is validated with ``ast.parse`` before it is trusted.
        """
        if not self.is_available():
            return None
        lines_ctx = ""
        if finding.get("lineno"):
            lines_ctx = f" at line {finding['lineno']}"

        def _run(prompt: str) -> str | None:
            response = self.generate(prompt, max_tokens=2048)
            if not response:
                return None
            return self._extract_code(response)

        region = self._enclosing_function(old_code, finding)
        if region is not None:
            func_src, start, end, pad = region
            prompt = (
                f"You are improving Python source code. Module: {module_name}.\n"
                f"Finding: {finding.get('type', 'unknown')}{lines_ctx}: "
                f"{finding.get('suggestion', '')}\n"
                "Below is the function that contains the problem. Apply a "
                "MINIMAL, SAFE fix to it. Keep the function signature, "
                "docstring and every other line unchanged unless the fix "
                "requires it. Use only the standard library. Do NOT refactor "
                "or add features.\n"
                "Return ONLY the complete rewritten function (including the "
                "``def`` line), dedented to column 0, inside a ```python "
                "code fence. It must parse as valid Python.\n\n"
                "```\n" + func_src + "\n```\n"
            )
            new_func = _run(prompt)
            if new_func is not None and new_func != func_src:
                return self._splice_function(old_code, new_func,
                                             start, end, pad)
            return None

        if len(old_code.splitlines()) > 300:
            return None
        prompt = (
            f"You are improving Python source code. Module: {module_name}.\n"
            f"Finding: {finding.get('type', 'unknown')}{lines_ctx}: "
            f"{finding.get('suggestion', '')}\n"
            "Apply a MINIMAL, SAFE fix to the code below. Keep every import, "
            "class, function signature and docstring unchanged unless the fix "
            "requires it. Use only the standard library. Do NOT refactor or "
            "add features.\n"
            "Return ONLY the complete new file content inside a ```python "
            "code fence. The code must parse as valid Python.\n\n"
            "```\n" + old_code + "\n```\n"
        )
        new_code = _run(prompt)
        if new_code is None or new_code == old_code:
            return None
        return new_code


# ──────────────────────────────────────────────
#  RecursiveSelfImprover — the full loop
# ──────────────────────────────────────────────


class RecursiveSelfImprover:
    """The recursive self-improvement loop.

    This is the key AGI mechanism: the agent reads its own code,
    identifies weaknesses, generates patches, tests them, and
    promotes successful improvements — making itself better at
    the next iteration of the same loop.

    The recursion is in the fact that an improved agent is better
    at improving itself, creating a positive feedback loop.
    """

    def __init__(self, agent: SelfEvolvingAgent,
                 source_root: str = ".",
                 auto_promote: bool = False,
                 llm_backend: OllamaBackend | None = None,
                 max_llm_calls_per_cycle: int = 3):
        self.agent = agent
        self.source_root = pathlib.Path(source_root).resolve()
        self.code_analyzer = CodeAnalyzer(str(self.source_root))
        self.auto_promote = auto_promote
        self.llm_backend = llm_backend
        self.max_llm_calls_per_cycle = max_llm_calls_per_cycle

        self.patches_applied: list[Patch] = []
        self.patches_tested: list[dict] = []
        self.patches_promoted: list[Patch] = []
        self.patches_rejected: list[Patch] = []
        self.improvement_cycle: int = 0
        self.llm_patch_count: int = 0
        self._llm_calls_this_cycle: int = 0

    def run_cycle(self, target_files: list[str] | None = None) -> dict:
        """Execute one cycle of self-improvement.

        1. Analyze own code for improvement opportunities.
        2. Generate candidate patches.
        3. Test in sandbox.
        4. Promote or reject.

        Returns results dict.
        """
        self.improvement_cycle += 1
        self._llm_calls_this_cycle = 0

        if target_files is None:
            target_files = self._find_agent_files()

        # Phase 1: Analyze
        all_findings = {}
        for f in target_files:
            findings = self.code_analyzer.analyze_module(f)
            all_findings[f] = findings

        # Phase 2: Generate patches from hypotheses
        hypotheses = self.agent.meta_cognition.top_improvement_opportunities(n=3)
        patches = []

        for hyp in hypotheses:
            # Read the target file
            fpath = self.source_root / hyp.target_module
            if not fpath.exists():
                continue
            old_code = fpath.read_text(encoding="utf-8")

            # Generate a candidate modification
            new_code = self._generate_candidate_code(
                hyp, old_code, all_findings.get(hyp.target_module, {})
            )
            if new_code and new_code != old_code:
                patch = Patch(
                    file_path=hyp.target_module,
                    old_code=old_code,
                    new_code=new_code,
                    description=hyp.description,
                )
                patches.append(patch)

        # Also generate patches from static analysis findings
        for fname, findings in all_findings.items():
            if isinstance(findings, dict) and "error" not in findings:
                for category, items in findings.items():
                    for item in items[:1]:  # top finding per category
                        patch = self._patch_from_finding(fname, item)
                        if patch:
                            patches.append(patch)

        # Phase 3: Test in sandbox
        results = []
        for patch in patches[:5]:  # limit to 5 patches per cycle
            result = self._test_patch(patch, target_files)
            results.append(result)

            if result["passed"]:
                self.patches_tested.append(result)
                if self.auto_promote:
                    self._promote_patch(patch)
                else:
                    self.patches_applied.append(patch)
            else:
                self.patches_rejected.append(patch)

        # Phase 4: Report
        summary = {
            "cycle": self.improvement_cycle,
            "files_analyzed": len(target_files),
            "findings": sum(len(f.get(k, [])) if isinstance(f, dict) else 0
                           for f in all_findings.values()
                           for k in ["complexity", "error_handling", "dead_code",
                                      "duplication", "generalization", "todo_comments"]),
            "patches_generated": len(patches),
            "patches_passed": sum(1 for r in results if r["passed"]),
            "patches_failed": sum(1 for r in results if not r["passed"]),
            "patches_promoted": len(self.patches_promoted),
            "total_promoted": len(self.patches_promoted),
        }
        return summary

    def _find_agent_files(self) -> list[str]:
        """Find Python files belonging to the agent."""
        files = []
        for pattern in ["selfevolve/*.py", "selfevolve/**/*.py"]:
            for f in self.source_root.glob(pattern):
                files.append(str(f.relative_to(self.source_root)))
        return files

    def _generate_candidate_code(self, hypothesis: ImprovementHypothesis,
                                  old_code: str,
                                  findings: dict) -> str | None:
        """Generate a specific code modification based on a hypothesis.

        In production, this delegates to the LLM. For stdlib-only
        operation, we use heuristic transformations.
        """
        new_code = old_code

        # Heuristic 1: Add try/except around bare blocks
        if "tool" in hypothesis.description.lower():
            new_code = self._heuristic_wrap_tool_calls(new_code)

        # Heuristic 2: Add caching to repeated computation
        if "performance" in hypothesis.description.lower():
            new_code = self._heuristic_add_cache(new_code)

        # Heuristic 3: Add parameterization for magic numbers
        if findings and "generalization" in findings:
            for finding in findings["generalization"]:
                if finding["type"] == "magic_number":
                    new_code = self._heuristic_parametrize_constant(
                        new_code, finding
                    )

        return new_code if new_code != old_code else None

    def _heuristic_wrap_tool_calls(self, source: str) -> str:
        """Add retry comment to dense function bodies."""
        lines = source.splitlines()
        result = []
        for line in lines:
            stripped = line.strip()
            result.append(line)
            if stripped.startswith("def ") and stripped.endswith(":"):
                indent = " " * (len(line) - len(line.lstrip()))
                result.append(f"{indent}    # TODO: add retry/deadline logic here when LLM available")
        return "\n".join(result)

    def _heuristic_add_cache(self, source: str) -> str:
        """Add a caching docstring to the first function."""
        lines = source.splitlines()
        result = []
        cache_added = False
        for line in lines:
            stripped = line.strip()
            result.append(line)
            if not cache_added and stripped.startswith("def ") and stripped.endswith(":"):
                indent = " " * (len(line) - len(line.lstrip()))
                result.append(f'{indent}    """Cached variant. TODO: implement LRU cache when LLM available."""')
                cache_added = True
        return "\n".join(result)

    def _heuristic_parametrize_constant(self, source: str,
                                         finding: dict) -> str:
        """Replace a magic number with a named constant."""
        lines = source.splitlines()
        value_str = str(finding["value"])
        lineno = finding["lineno"] - 1  # 0-indexed
        if 0 <= lineno < len(lines):
            old_line = lines[lineno]
            if value_str in old_line:
                new_line = old_line.replace(
                    f" {value_str}",
                    f" {finding['value']}  # TODO: extract as named constant",
                    1
                )
                if new_line != old_line:
                    lines[lineno] = new_line
        return "\n".join(lines)

    def _patch_from_finding(self, fname: str, finding: dict) -> Patch | None:
        """Generate a patch from a static analysis finding.

        If an LLM backend is available (and within the per-cycle call
        budget), the LLM tries first; heuristic transforms are the
        fallback so the pipeline still works stdlib-only.
        """
        full_path = self.source_root / fname
        if not full_path.exists():
            return None

        source = full_path.read_text(encoding="utf-8")
        ftype = finding.get("type", "")
        lines = source.splitlines()

        if self.llm_backend is not None and \
                self._llm_calls_this_cycle < self.max_llm_calls_per_cycle:
            self._llm_calls_this_cycle += 1
            new_source = self.llm_backend.suggest_patch(fname, source, finding)
            if new_source:
                self.llm_patch_count += 1
                return Patch(
                    file_path=fname,
                    old_code=source,
                    new_code=new_source,
                    description=f"LLM fix ({ftype}): "
                                f"{finding.get('suggestion', '')}",
                )

        if ftype == "bare_except":
            new_source = source.replace(
                "except:",
                "except Exception:  # selfevolve: specify exception type",
            )
            if new_source != source:
                return Patch(
                    file_path=fname,
                    old_code=source,
                    new_code=new_source,
                    description=f"Replace bare except at line {finding['lineno']}",
                )

        if ftype == "high_complexity":
            name = finding.get("name", "")
            lineno = finding.get("lineno", 1) - 1
            # Idempotency: never flag the same function twice.
            if name and any(f"NOTE({name})" in l for l in lines):
                return None
            if 0 <= lineno < len(lines):
                indent = " " * (len(lines[lineno]) - len(lines[lineno].lstrip()))
                note = f"{indent}# NOTE({name}): high complexity ({finding.get('complexity', '?')}) - consider splitting\n"
                lines.insert(lineno + 1, note)
                new_source = "\n".join(lines)
                if new_source != source:
                    return Patch(
                        file_path=fname,
                        old_code=source,
                        new_code=new_source,
                        description=f"Flag high-complexity function {name} for future refactoring",
                    )

        if ftype == "none_comparison":
            lineno = finding.get("lineno", 1) - 1
            op = finding.get("op", "==")
            if 0 <= lineno < len(lines):
                old_line = lines[lineno]
                new_line = old_line.replace(f"== None", "is None", 1) if op == "==" \
                    else old_line.replace(f"!= None", "is not None", 1)
                if new_line != old_line:
                    lines[lineno] = new_line
                    new_source = "\n".join(lines)
                    return Patch(
                        file_path=fname,
                        old_code=source,
                        new_code=new_source,
                        description=f"Use '{'is' if op == '==' else 'is not'} None' "
                                    f"instead of '{op} None' at line {lineno + 1}",
                    )
            return None

        if ftype == "magic_number":
            val = finding.get("value")
            lineno = finding.get("lineno", 1) - 1
            if val is not None and 0 <= lineno < len(lines):
                # Replace the literal with a named constant at module level
                const_name = f"{finding.get('function', 'f')}_{abs(int(val))}_CONST"
                # Name-mangling guard: identifiers starting with `__` get
                # mangled when referenced inside class bodies.
                const_name = const_name.lstrip("_") or "CONST"
                old_line = lines[lineno]
                val_str = str(val)
                if val_str in old_line:
                    new_line = old_line.replace(val_str, const_name, 1)
                    if new_line != old_line:
                        lines[lineno] = new_line
                        # Add constant at top of file after imports
                        import_end = 0
                        for i, l in enumerate(lines):
                            if l.startswith(("import ", "from ")):
                                import_end = i + 1
                            elif import_end > 0 and not l.strip():
                                import_end = i + 1
                            elif import_end > 0 and l.strip():
                                break
                        inserted = 0
                        for i in range(import_end, len(lines)):
                            if lines[i].strip().startswith(("#", "class ", "def ")) or not lines[i].strip():
                                indent = ""
                                if lines[i].strip().startswith(("class ", "def ")):
                                    pass  # insert before
                                else:
                                    pass
                                lines.insert(i + inserted, "")
                                lines.insert(i + inserted,
                                    f"{const_name} = {val_str}  # selfevolve: extracted from line {finding['lineno']}")
                                inserted += 2
                                break
                        new_source = "\n".join(lines)
                        if new_source != source:
                            return Patch(
                                file_path=fname,
                                old_code=source,
                                new_code=new_source,
                                description=f"Extract magic number {val} as constant {const_name}",
                            )

        if ftype == "duplicated_block":
            lineno_a = finding.get("lineno_a", 0)
            lineno_b = finding.get("lineno_b", 0)
            nlines = finding.get("lines", 0)
            if not (0 <= lineno_a < lineno_b < len(lines) and nlines >= 5
                    and lineno_b + nlines <= len(lines)):
                return None
            block = lines[lineno_a:lineno_a + nlines]
            # Guards: block must be straight-line code (no control flow,
            # no imports), and any variable assigned inside the block must
            # not be used outside it. Otherwise fall back to a comment flag.
            joined = "\n".join(block)
            if re.search(r"^\s*(return|break|continue|raise|yield|try|finally|"
                         r"for|while|if|elif|else|def|class|with|import|except)\b",
                         joined, re.M):
                return self._flag_duplicate(fname, source, lines,
                                            lineno_a, lineno_b)
            assigned = set()
            for bl in block:
                m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", bl)
                if m:
                    assigned.add(m.group(1))
            if assigned:
                outside = "\n".join(
                    lines[:lineno_a] + lines[lineno_a + nlines:lineno_b] +
                    lines[lineno_b + nlines:]
                )
                if any(name in outside for name in assigned):
                    return self._flag_duplicate(fname, source, lines,
                                            lineno_a, lineno_b)
            # Extract to a module-level helper and replace both call sites.
            indent = " " * (len(block[0]) - len(block[0].lstrip()))
            helper = f"_dedupe_block_{lineno_a + 1}"
            dedented = [bl[len(indent):] if bl.startswith(indent) else bl
                        for bl in block]
            helper_lines = [f"def {helper}():"] + \
                           [f"    {bl}" for bl in dedented]
            new_lines = list(lines)
            call_indent_b = " " * (len(new_lines[lineno_b]) -
                                   len(new_lines[lineno_b].lstrip()))
            new_lines[lineno_b:lineno_b + nlines] = \
                [f"{call_indent_b}{helper}()"]
            call_indent_a = " " * (len(new_lines[lineno_a]) -
                                   len(new_lines[lineno_a].lstrip()))
            new_lines[lineno_a:lineno_a + nlines] = \
                [f"{call_indent_a}{helper}()"]
            idx = self._find_insertion_index(new_lines)
            new_lines[idx:idx] = [""] + helper_lines + [""]
            new_source = "\n".join(new_lines)
            if new_source != source:
                return Patch(
                    file_path=fname,
                    old_code=source,
                    new_code=new_source,
                    description=f"Extract duplicated block "
                                f"(L{lineno_a + 1}, L{lineno_b + 1}) "
                                f"into helper {helper}()",
                )

        if ftype == "missing_try":
            return None

        return None

    @staticmethod
    def _find_insertion_index(lines: list[str]) -> int:
        """Index after imports (and trailing blank lines) where new
        module-level definitions should be inserted."""
        import_end = 0
        for i, l in enumerate(lines):
            if l.startswith(("import ", "from ")):
                import_end = i + 1
            elif import_end > 0 and not l.strip():
                import_end = i + 1
            elif import_end > 0 and l.strip():
                break
        return import_end

    def _flag_duplicate(self, fname: str, source: str,
                        lines: list[str], lineno_a: int,
                        lineno_b: int) -> Patch | None:
        """Fallback for duplicated blocks we cannot safely extract:
        insert a flag comment above the second occurrence."""
        if 0 <= lineno_b < len(lines):
            indent = " " * (len(lines[lineno_b]) - len(lines[lineno_b].lstrip()))
            comment = (f"{indent}# DUPLICATE: blocks at L{lineno_a + 1} and "
                       f"L{lineno_b + 1} - extract to shared function\n")
            new_lines = list(lines)
            new_lines.insert(lineno_b, comment)
            new_source = "\n".join(new_lines)
            if new_source != source:
                return Patch(
                    file_path=fname,
                    old_code=source,
                    new_code=new_source,
                    description=f"Flag duplicated block "
                                f"(L{lineno_a + 1}, L{lineno_b + 1}) "
                                f"for extraction",
                )
        return None

    def _test_patch(self, patch: Patch, related_files: list[str]) -> dict:
        """Test a single patch in a sandbox.

        Returns dict with passed (bool), test_output, duration.
        """
        # Prepare the whole package so the test suite can import every module.
        prepare_files = list(related_files)
        pkg_dir = self.source_root / "selfevolve"
        if pkg_dir.is_dir():
            for src in sorted(pkg_dir.rglob("*.py")):
                rel = src.relative_to(self.source_root)
                if str(rel) not in prepare_files:
                    prepare_files.append(str(rel))

        with Sandbox(str(self.source_root)) as sandbox:
            sandbox.prepare(prepare_files)
            applied = sandbox.apply_patch(patch)
            if not applied:
                return {
                    "patch_id": patch.id,
                    "passed": False,
                    "error": "Failed to apply patch in sandbox",
                    "duration_s": 0,
                }

            test_result = sandbox.run_test()

            passed = test_result["success"]
            result = {
                "patch_id": patch.id,
                "passed": passed,
                "file": patch.file_path,
                "description": patch.description,
                "test_stdout": test_result["stdout"][:500],
                "test_stderr": test_result["stderr"][:500],
                "duration_s": test_result["duration_s"],
            }
            return result

    def _promote_patch(self, patch: Patch) -> bool:
        """Promote a tested patch to production."""
        success = patch.apply(str(self.source_root))
        if success:
            self.patches_promoted.append(patch)
            self.agent.modifications.append({
                "hypothesis_id": patch.id,
                "patch": patch.description,
                "applied_at": time.time(),
                "cycle": self.improvement_cycle,
            })
        return success

    def summary(self) -> str:
        ok = "[OK]"
        fail = "[--]"
        lines = [
            "+- RecursiveSelfImprover --",
            "|  Cycles: %d" % self.improvement_cycle,
            "|  Patches generated: %d" % (len(self.patches_applied) + len(self.patches_rejected)),
            "|  Patches promoted: %d" % len(self.patches_promoted),
            "|  Patches rejected: %d" % len(self.patches_rejected),
            "|  Auto-promote: %s" % self.auto_promote,
            "|  LLM patches: %d" % self.llm_patch_count,
        ]
        for p in self.patches_promoted[-5:]:
            lines.append("|    %s %s" % (ok, p.description[:60]))
        for p in self.patches_rejected[-3:]:
            lines.append("|    %s %s" % (fail, p.description[:60]))
        lines.append("+--")
        return "\n".join(lines)