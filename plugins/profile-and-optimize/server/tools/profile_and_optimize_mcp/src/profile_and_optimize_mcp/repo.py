"""Repository discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path


def _operator_path(value: str, *, label: str) -> Path:
    """Normalize a complete path chosen by the local server operator."""
    if not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty local path without NUL bytes")

    # codeql[py/path-injection]
    return Path(value).expanduser().resolve()


def find_repo_root(start: Path | None = None) -> Path:
    """Return the bundled server root using product files as markers."""

    env_root = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    current = _operator_path(env_root, label="PROFILE_AND_OPTIMIZE_REPO_ROOT") if env_root else (start or Path.cwd())
    current = current.resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if (
            (current / "pyproject.toml").is_file()
            and (current / "mcp_surface.py").is_file()
            and (current / "tools").is_dir()
        ):
            return current
        current = current.parent
    raise RuntimeError("cannot locate the profile-and-optimize server root. Set PROFILE_AND_OPTIMIZE_REPO_ROOT")


def repo_path(*parts: str) -> Path:
    return find_repo_root() / Path(*parts)
