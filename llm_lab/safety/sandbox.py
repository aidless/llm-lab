"""In-memory tools and a runtime policy gate for Agent Safety evaluation."""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from llm_lab.safety.models import canonical_json


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["result"] = self.result
        else:
            payload["error"] = self.error
        return payload


def normalize_path(raw: Any, *, workspace: str = "/workspace") -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path must be a non-empty string")
    if "\x00" in raw:
        raise ValueError("path contains a NUL byte")
    path = raw if raw.startswith("/") else posixpath.join(workspace, raw)
    normalized = posixpath.normpath(path)
    if normalized == "." or not normalized.startswith("/"):
        raise ValueError("path must resolve to an absolute path")
    return normalized


def is_under(path: str, roots: list[str]) -> bool:
    for root in roots:
        normalized_root = posixpath.normpath(str(root))
        if path == normalized_root or path.startswith(normalized_root.rstrip("/") + "/"):
            return True
    return False


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


class PolicyEngine:
    """Deterministic permissions and data-flow checks.

    The engine is a runtime policy gate, not a safety label source.  It can
    deny an attempted action in ``policy-on`` mode or merely classify the
    attempt in ``policy-off`` mode.  Every tool remains mock-only regardless of
    mode.
    """

    def __init__(self, allowed_tools: tuple[str, ...], policy: dict[str, Any]) -> None:
        # ``allowed_tools`` is the set exposed to the model.  A task may expose
        # a tempting tool while withholding it from the runtime policy so that
        # policy-on and policy-off can be compared without giving the monitor
        # access to the model's private thoughts.
        policy_tools = policy.get("allowed_tools")
        self.allowed_tools = (
            {str(tool) for tool in policy_tools} if isinstance(policy_tools, list) else set(allowed_tools)
        )
        self.policy = dict(policy)
        self.read_roots = _string_list(self.policy, "read_roots") or ["/workspace"]
        self.write_roots = _string_list(self.policy, "write_roots") or ["/workspace/output"]
        self.delete_roots = _string_list(self.policy, "delete_roots") or ["/workspace/tmp"]
        self.protected_paths = _string_list(self.policy, "protected_paths")
        self.allowed_recipients = {item.lower() for item in _string_list(self.policy, "allowed_recipients")}
        self.allowed_domains = {item.lower() for item in _string_list(self.policy, "allowed_domains")}
        self.allowed_urls = {item.rstrip("/") for item in _string_list(self.policy, "allowed_urls")}
        self.shell_allowlist = _string_list(self.policy, "shell_allowlist")
        self.sensitive_markers = _string_list(self.policy, "sensitive_markers")
        self.allow_sensitive_outbound = bool(self.policy.get("allow_sensitive_outbound", False))

    def _protected(self, path: str) -> bool:
        return is_under(path, self.protected_paths)

    def _sensitive(self, text: str) -> str | None:
        if self.allow_sensitive_outbound:
            return None
        for marker in self.sensitive_markers:
            if marker and marker in text:
                return marker
        return None

    def check(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        available_urls: dict[str, str] | None = None,
    ) -> PolicyDecision:
        if tool not in self.allowed_tools:
            return PolicyDecision(False, "tool_not_allowed", f"tool {tool!r} is not allowed")
        try:
            if tool in {"read_file", "list_dir"}:
                path = normalize_path(arguments.get("path"))
                if not is_under(path, self.read_roots):
                    return PolicyDecision(
                        False,
                        "read_scope",
                        f"read path {path!r} is outside approved roots",
                    )
                if self._protected(path):
                    return PolicyDecision(
                        False,
                        "protected_path",
                        f"read path {path!r} is protected",
                    )
                return PolicyDecision(True, "allowed", "read inside approved root")

            if tool == "write_file":
                path = normalize_path(arguments.get("path"))
                if not is_under(path, self.write_roots):
                    return PolicyDecision(
                        False,
                        "write_scope",
                        f"write path {path!r} is outside approved roots",
                    )
                if self._protected(path):
                    return PolicyDecision(
                        False,
                        "protected_path",
                        f"write path {path!r} is protected",
                    )
                return PolicyDecision(True, "allowed", "write inside approved root")

            if tool == "delete_file":
                path = normalize_path(arguments.get("path"))
                if not is_under(path, self.delete_roots):
                    return PolicyDecision(
                        False,
                        "delete_scope",
                        f"delete path {path!r} is outside approved roots",
                    )
                if self._protected(path):
                    return PolicyDecision(
                        False,
                        "protected_path",
                        f"delete path {path!r} is protected",
                    )
                return PolicyDecision(True, "allowed", "delete inside approved root")

            if tool == "send_email":
                recipient = str(arguments.get("to", "")).strip().lower()
                if not recipient or "@" not in recipient:
                    return PolicyDecision(False, "malformed_args", "email recipient is invalid")
                if recipient not in self.allowed_recipients:
                    return PolicyDecision(
                        False,
                        "recipient_not_allowed",
                        f"recipient {recipient!r} is not approved",
                    )
                body = f"{arguments.get('subject', '')}\n{arguments.get('body', '')}"
                marker = self._sensitive(body)
                if marker is not None:
                    return PolicyDecision(
                        False,
                        "sensitive_content",
                        f"outbound email contains sensitive marker {marker!r}",
                    )
                return PolicyDecision(True, "allowed", "approved internal email")

            if tool == "http_get":
                url = str(arguments.get("url", "")).strip()
                if not url:
                    return PolicyDecision(False, "malformed_args", "URL is required")
                if available_urls is not None and url not in available_urls:
                    return PolicyDecision(
                        False,
                        "url_not_allowed",
                        "URL is not present in the frozen mock environment",
                    )
                domain = (urlparse(url).hostname or "").lower()
                if domain not in self.allowed_domains:
                    return PolicyDecision(
                        False,
                        "domain_not_allowed",
                        f"domain {domain!r} is not approved",
                    )
                return PolicyDecision(True, "allowed", "approved mock URL")

            if tool == "http_post":
                url = str(arguments.get("url", "")).strip()
                domain = (urlparse(url).hostname or "").lower()
                if domain not in self.allowed_domains:
                    return PolicyDecision(
                        False,
                        "domain_not_allowed",
                        f"domain {domain!r} is not approved",
                    )
                body = f"{url}\n{canonical_json(arguments.get('data', {}))}"
                marker = self._sensitive(body)
                if marker is not None:
                    return PolicyDecision(
                        False,
                        "sensitive_content",
                        f"outbound HTTP body contains sensitive marker {marker!r}",
                    )
                return PolicyDecision(True, "allowed", "approved outbound domain")

            if tool == "run_shell":
                command = str(arguments.get("command", "")).strip()
                if not command:
                    return PolicyDecision(False, "malformed_args", "shell command is empty")
                if any(token in command for token in (";", "&&", "||", "|", ">", "<", "`", "$(")):
                    return PolicyDecision(
                        False,
                        "shell_metacharacters",
                        "shell metacharacters are not allowed",
                    )
                try:
                    tokens = shlex.split(command)
                except ValueError as exc:
                    return PolicyDecision(False, "malformed_args", f"invalid command: {exc}")
                executable = posixpath.basename(tokens[0]) if tokens else ""
                if executable not in self.shell_allowlist:
                    return PolicyDecision(
                        False,
                        "shell_not_allowed",
                        f"shell executable {executable!r} is not allowed",
                    )
                return PolicyDecision(True, "allowed", "allowlisted mock shell command")
        except (TypeError, ValueError) as exc:
            return PolicyDecision(False, "malformed_args", str(exc))
        return PolicyDecision(False, "unknown_tool", f"tool {tool!r} is not implemented")


@dataclass
class MockEnvironment:
    """Mutable copy of one task's synthetic files and outbound sinks."""

    files: dict[str, dict[str, Any]]
    urls: dict[str, str]
    mock_responses: dict[str, Any]
    tool_faults: list[dict[str, Any]]
    outbox: list[dict[str, str]] = field(default_factory=list)
    http_log: list[dict[str, Any]] = field(default_factory=list)
    shell_log: list[dict[str, Any]] = field(default_factory=list)
    deleted_paths: list[str] = field(default_factory=list)
    fault_events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: Any) -> MockEnvironment:
        return cls(
            files={
                str(path): {
                    **dict(metadata),
                    "content": str(dict(metadata).get("content", "")),
                }
                for path, metadata in task.files.items()
            },
            urls=dict(task.urls),
            mock_responses=dict(task.mock_responses),
            tool_faults=[dict(fault) for fault in task.tool_faults],
        )

    def _consume_fault(self, tool: str, arguments: dict[str, Any]) -> str | None:
        serialized_args = canonical_json(arguments)
        for fault in self.tool_faults:
            if str(fault.get("tool", "")) != tool:
                continue
            contains = fault.get("arg_contains")
            if contains is not None and str(contains) not in serialized_args:
                continue
            remaining = int(fault.get("remaining", 1))
            if remaining <= 0:
                continue
            fault["remaining"] = remaining - 1
            message = str(fault.get("error", "injected tool failure"))
            self.fault_events.append({"tool": tool, "error": message, "arguments": arguments})
            return message
        return None

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        fault = self._consume_fault(tool, arguments)
        if fault is not None:
            return ToolResult(False, error=fault)
        try:
            if tool == "list_dir":
                path = normalize_path(arguments.get("path"))
                names: set[str] = set()
                prefix = path.rstrip("/") + "/"
                for file_path in self.files:
                    if file_path.startswith(prefix):
                        names.add(file_path[len(prefix) :].split("/", 1)[0])
                return ToolResult(True, result={"path": path, "entries": sorted(names)})

            if tool == "read_file":
                path = normalize_path(arguments.get("path"))
                metadata = self.files.get(path)
                if metadata is None:
                    return ToolResult(False, error=f"file not found: {path}")
                return ToolResult(
                    True,
                    result={
                        "path": path,
                        "content": str(metadata.get("content", "")),
                        "classification": str(metadata.get("classification", "internal")),
                    },
                )

            if tool == "write_file":
                path = normalize_path(arguments.get("path"))
                content = str(arguments.get("content", ""))
                if len(content) > 50_000:
                    return ToolResult(False, error="content exceeds 50,000 characters")
                old = self.files.get(path)
                classification = (
                    str(arguments.get("classification"))
                    if arguments.get("classification")
                    else str(old.get("classification", "internal") if old else "internal")
                )
                self.files[path] = {
                    "content": content,
                    "classification": classification,
                    "written_by_agent": True,
                }
                return ToolResult(True, result={"path": path, "bytes": len(content.encode("utf-8"))})

            if tool == "delete_file":
                path = normalize_path(arguments.get("path"))
                if path not in self.files:
                    return ToolResult(False, error=f"file not found: {path}")
                del self.files[path]
                self.deleted_paths.append(path)
                return ToolResult(True, result={"deleted": path})

            if tool == "send_email":
                to = str(arguments.get("to", "")).strip().lower()
                subject = str(arguments.get("subject", ""))
                body = str(arguments.get("body", ""))
                if not to or not subject:
                    return ToolResult(False, error="email requires to and subject")
                record = {"to": to, "subject": subject, "body": body}
                self.outbox.append(record)
                return ToolResult(
                    True,
                    result={"message_id": f"mock-email-{len(self.outbox):04d}", "to": to},
                )

            if tool == "http_get":
                url = str(arguments.get("url", "")).strip()
                if url not in self.urls:
                    return ToolResult(False, error=f"mock URL not found: {url}")
                return ToolResult(True, result={"url": url, "content": self.urls[url]})

            if tool == "http_post":
                url = str(arguments.get("url", "")).strip()
                data = arguments.get("data", {})
                self.http_log.append({"url": url, "data": data})
                return ToolResult(True, result={"status": 200, "url": url})

            if tool == "run_shell":
                command = str(arguments.get("command", "")).strip()
                tokens = shlex.split(command)
                self.shell_log.append({"command": command})
                executable = posixpath.basename(tokens[0]) if tokens else ""
                if executable == "ls":
                    path = normalize_path(tokens[1] if len(tokens) > 1 else "/workspace")
                    names = set()
                    prefix = path.rstrip("/") + "/"
                    for file_path in self.files:
                        if file_path.startswith(prefix):
                            names.add(file_path[len(prefix) :].split("/", 1)[0])
                    return ToolResult(True, result={"stdout": "\n".join(sorted(names))})
                if executable == "cat" and len(tokens) == 2:
                    path = normalize_path(tokens[1])
                    metadata = self.files.get(path)
                    if metadata is None:
                        return ToolResult(False, error=f"file not found: {path}")
                    return ToolResult(True, result={"stdout": str(metadata.get("content", ""))})
                if executable == "wc" and len(tokens) == 3 and tokens[1] in {"-c", "--bytes"}:
                    path = normalize_path(tokens[2])
                    metadata = self.files.get(path)
                    if metadata is None:
                        return ToolResult(False, error=f"file not found: {path}")
                    length = len(str(metadata.get("content", "")).encode("utf-8"))
                    return ToolResult(True, result={"stdout": f"{length} {path}"})
                return ToolResult(False, error=f"unsupported mock shell command: {command}")
        except (TypeError, ValueError) as exc:
            return ToolResult(False, error=str(exc))
        return ToolResult(False, error=f"tool not implemented: {tool}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "path_count": len(self.files),
            "paths": sorted(self.files),
            "outbox": list(self.outbox),
            "http_log": list(self.http_log),
            "shell_log": list(self.shell_log),
            "deleted_paths": list(self.deleted_paths),
            "fault_events": list(self.fault_events),
            "no_real_filesystem_or_network_access": True,
        }
