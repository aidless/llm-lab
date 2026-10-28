"""Fail-closed ownership checks for destructive artifact output paths."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

ArtifactType = Literal["dataset", "evidence", "review"]

MARKER_FILENAME = ".agent-safety-eval-owned"
MARKER_TOOL = "agent-safety-eval"
MARKER_FORMAT_VERSION = 1
ARTIFACT_TYPES = frozenset({"dataset", "evidence", "review"})

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
KNOWN_OUTPUT_ROOTS: dict[ArtifactType, Path] = {
    "dataset": PROJECT_ROOT / "dataset",
    "evidence": PROJECT_ROOT / "evidence",
    "review": PROJECT_ROOT / "review",
}


class OutputPathError(ValueError):
    """An output path is unsafe to create, replace, or trust."""


def _artifact_type(value: str) -> ArtifactType:
    if value not in ARTIFACT_TYPES:
        raise OutputPathError(f"unknown artifact type: {value!r}")
    return value  # type: ignore[return-value]


def _raw_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise OutputPathError("output path must be text")
    if "\x00" in raw:
        raise OutputPathError("output path contains a NUL byte")
    path = Path(raw).expanduser()
    if ".." in path.parts:
        raise OutputPathError(f"output path must not contain '..': {raw!r}")
    return path


def _symlink_component(path: Path) -> Path | None:
    """Return the first existing symlink component, including a broken target."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _same_or_ancestor(candidate: Path, target: Path) -> bool:
    try:
        target.relative_to(candidate)
    except ValueError:
        return False
    return True


def _validate_target(value: str | os.PathLike[str], artifact_type: str) -> tuple[Path, Path]:
    expected_type = _artifact_type(artifact_type)
    path = _raw_path(value)
    symlink = _symlink_component(path)
    if symlink is not None:
        raise OutputPathError(f"output path crosses a symlink: {symlink}")
    lexical = path if path.is_absolute() else Path.cwd() / path
    resolved = path.resolve(strict=False)

    protected_roots = {
        Path("/").resolve(),
        WORKSPACE_ROOT.resolve(),
        PROJECT_ROOT.resolve(),
        PACKAGE_ROOT.resolve(),
    }
    for protected in protected_roots:
        if _same_or_ancestor(resolved, protected):
            raise OutputPathError(f"refusing protected root or ancestor: {resolved}")
        if _same_or_ancestor(lexical.resolve(strict=False), protected):
            raise OutputPathError(f"refusing protected root or ancestor: {lexical}")

    for known_type, root in KNOWN_OUTPUT_ROOTS.items():
        known = root.resolve(strict=False)
        if resolved == known:
            if expected_type != known_type:
                raise OutputPathError(
                    f"{known} is reserved for {known_type!r}, not {expected_type!r}"
                )
            break
        if _same_or_ancestor(resolved, known):
            raise OutputPathError(f"refusing ancestor of known output root: {known}")
    return path, resolved


def _read_marker(directory: Path) -> dict[str, Any]:
    marker_path = directory / MARKER_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise OutputPathError(f"missing regular ownership marker: {marker_path}")
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutputPathError(f"invalid ownership marker: {marker_path}") from exc
    if not isinstance(payload, dict):
        raise OutputPathError(f"ownership marker is not an object: {marker_path}")
    expected = {
        "format_version": MARKER_FORMAT_VERSION,
        "tool": MARKER_TOOL,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise OutputPathError(f"ownership marker has invalid {key}: {marker_path}")
    if payload.get("artifact_type") not in ARTIFACT_TYPES:
        raise OutputPathError(f"ownership marker has invalid artifact_type: {marker_path}")
    if set(payload) != {"format_version", "tool", "artifact_type"}:
        raise OutputPathError(f"ownership marker has unexpected fields: {marker_path}")
    return payload


def _write_marker(directory: Path, artifact_type: ArtifactType) -> None:
    marker_path = directory / MARKER_FILENAME
    payload = {
        "format_version": MARKER_FORMAT_VERSION,
        "tool": MARKER_TOOL,
        "artifact_type": artifact_type,
    }
    fd, temporary = tempfile.mkstemp(
        dir=directory,
        prefix=f".{MARKER_FILENAME}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _non_marker_entries(directory: Path) -> list[Path]:
    return [entry for entry in directory.iterdir() if entry.name != MARKER_FILENAME]


def claim_output(
    output: str | os.PathLike[str],
    artifact_type: str,
    *,
    replace: bool = False,
    allow_existing_owned: bool = False,
) -> Path:
    """Claim a generated-artifact directory, replacing only an owned exact target.

    New and empty directories are claimed by writing an ownership marker.
    Existing non-empty directories must already carry a matching marker.
    Review packets are never replaced because they can contain pending human
    decisions.
    """
    expected_type = _artifact_type(artifact_type)
    _, path = _validate_target(output, expected_type)

    if path.is_symlink():
        raise OutputPathError(f"output path is a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise OutputPathError(f"output path is not a directory: {path}")

    nonempty = path.is_dir() and any(path.iterdir())
    if nonempty:
        marker = _read_marker(path)
        if marker["artifact_type"] != expected_type:
            raise OutputPathError(
                f"owned directory is {marker['artifact_type']!r}, not {expected_type!r}: {path}"
            )
        if replace:
            if expected_type == "review" and _non_marker_entries(path):
                raise OutputPathError(f"refusing to replace a review directory: {path}")
            # Revalidate immediately before the only recursive deletion path.
            if path.is_symlink() or not path.is_dir():
                raise OutputPathError(f"refusing to replace unsafe target: {path}")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=False)
            _write_marker(path, expected_type)
            return path
        if allow_existing_owned:
            return path
        raise OutputPathError(
            f"output directory is not empty; pass --force to replace owned {expected_type} artifacts: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)
    _write_marker(path, expected_type)
    return path


def preflight_empty_output(
    output: str | os.PathLike[str],
    artifact_type: str,
) -> Path:
    """Validate that a review packet target is new or an empty owned directory."""
    expected_type = _artifact_type(artifact_type)
    _, path = _validate_target(output, expected_type)
    if path.is_symlink():
        raise OutputPathError(f"output path is a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise OutputPathError(f"output path is not a directory: {path}")
    if path.is_dir():
        entries = list(path.iterdir())
        if entries:
            marker = _read_marker(path)
            if marker["artifact_type"] != expected_type:
                raise OutputPathError(
                    f"owned directory is {marker['artifact_type']!r}, not {expected_type!r}: {path}"
                )
            labels = ", ".join(sorted(entry.name for entry in entries))
            raise OutputPathError(f"refusing to overwrite existing output contents: {labels}")
    return path


def claim_empty_output(
    output: str | os.PathLike[str],
    artifact_type: str,
) -> Path:
    """Claim a new or empty directory without ever replacing existing contents."""
    path = preflight_empty_output(output, artifact_type)
    path.mkdir(parents=True, exist_ok=True)
    _write_marker(path, _artifact_type(artifact_type))
    return path


def require_owned_output(
    output: str | os.PathLike[str],
    artifact_type: str,
) -> Path:
    """Require an existing non-empty owned directory before in-place updates."""
    expected_type = _artifact_type(artifact_type)
    _, path = _validate_target(output, expected_type)
    if path.is_symlink() or not path.is_dir():
        raise OutputPathError(f"owned output directory is missing: {path}")
    marker = _read_marker(path)
    if marker["artifact_type"] != expected_type:
        raise OutputPathError(
            f"owned directory is {marker['artifact_type']!r}, not {expected_type!r}: {path}"
        )
    return path
