"""Repository discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the bundled server root using product files as markers."""

    env_root = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    current = Path(env_root).expanduser() if env_root else (start or Path.cwd())
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
    raise RuntimeError(
        "cannot locate the profile-and-optimize server root. Set PROFILE_AND_OPTIMIZE_REPO_ROOT"
    )


def repo_path(*parts: str) -> Path:
    return find_repo_root() / Path(*parts)
