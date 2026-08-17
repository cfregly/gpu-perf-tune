"""Shared helpers for the perf-report CLI verbs."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGNS_DIR = SERVER_ROOT / "experiments" / "artifacts" / "perf-tune-report" / "campaigns"
CAMPAIGNS_ENV = "PERFREPORT_CAMPAIGNS_DIR"

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_SAFE_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def resolve_operator_path(value: str, *, label: str) -> Path:
    """Normalize a complete local path selected by the process operator.

    Use this only for CLI and environment overrides that intentionally select
    the whole path. Child names joined beneath an artifact root must instead
    pass through ``safe_path_segment`` and a containment check.
    """
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty local path without NUL bytes")

    # codeql[py/path-injection]
    return Path(value).expanduser().resolve()


def resolve_campaigns_dir(override: str | None = None) -> Path:
    """Resolve the campaigns root, honoring (1) explicit override, (2) env,
    (3) the server-local ``experiments/artifacts/perf-tune-report/campaigns``
    default."""
    if override:
        return resolve_operator_path(override, label="campaigns directory")
    env = os.environ.get(CAMPAIGNS_ENV)
    if env:
        return resolve_operator_path(env, label=CAMPAIGNS_ENV)
    return DEFAULT_CAMPAIGNS_DIR.expanduser().resolve()


def safe_path_segment(value: str, *, label: str) -> str:
    """Validate one path segment before joining it under an artifact root."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or _SAFE_SEGMENT_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be one safe path segment")
    return value


def resolve_cell_dir(campaign_dir: Path, cell_id: str) -> Path:
    """Resolve a cell directory and prove it remains under campaign/cells."""
    safe_cell_id = safe_path_segment(cell_id, label="cell-id")
    cells_root = (Path(campaign_dir).resolve() / "cells").resolve()
    cell_dir = (cells_root / safe_cell_id).resolve()
    if not cell_dir.is_relative_to(cells_root):
        raise ValueError("cell path escapes the campaign cells root")
    return cell_dir


def resolve_campaign_dir(slug_or_path: str, campaigns_root: Path | None = None) -> Path:
    """If the argument looks like an absolute / relative path that exists,
    return that. Otherwise treat it as a campaign slug under campaigns_root."""
    candidate = resolve_operator_path(slug_or_path, label="campaign")
    if candidate.exists():
        return candidate
    try:
        slug = safe_path_segment(slug_or_path, label="campaign")
    except ValueError as exc:
        raise SystemExit(f"FATAL: invalid campaign path or slug: {exc}") from exc
    root = (campaigns_root or resolve_campaigns_dir()).resolve()
    direct = (root / slug).resolve()
    if not direct.is_relative_to(root):
        raise SystemExit("FATAL: campaign slug escapes the campaigns directory")
    if direct.exists():
        return direct
    # Glob: <slug> matched as suffix of any campaign dir name.
    for entry in sorted(root.glob(f"*-{slug}")):
        if entry.is_dir():
            resolved_entry = entry.resolve()
            if not resolved_entry.is_relative_to(root):
                raise SystemExit("FATAL: campaign symlink escapes the campaigns directory")
            return resolved_entry
    raise SystemExit(
        f"FATAL: could not resolve campaign {slug_or_path!r}; "
        f"checked {candidate}, {direct}, and *-{slug_or_path} under {root}"
    )


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "campaign"


def utc_timestamp_slug() -> str:
    """``YYYYMMDDTHHMMSSZ`` per the workspace's evidence-bundle-init convention."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def emit(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        for k in sorted(payload):
            print(f"{k}: {payload[k]}")


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - yaml is in core deps
        raise SystemExit("FATAL: PyYAML is required to load campaign configs") from exc
    return yaml.safe_load(path.read_text())


def synthetic_fixture_path() -> Path:
    """Path to the bundled synthetic_atlas.jsonl, regardless of caller cwd."""
    here = Path(__file__).resolve().parent
    return here / "fixtures" / "synthetic_atlas.jsonl"
