"""Shared helpers for the perf-report CLI verbs."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGNS_DIR = SERVER_ROOT / "experiments" / "artifacts" / "perf-tune-report" / "campaigns"
CAMPAIGNS_ENV = "PERFREPORT_CAMPAIGNS_DIR"

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_SAFE_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def resolve_campaigns_dir(override: str | None = None) -> Path:
    """Resolve the campaigns root, honoring (1) explicit override, (2) env,
    (3) the server-local ``experiments/artifacts/perf-tune-report/campaigns``
    default."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get(CAMPAIGNS_ENV)
    if env:
        return Path(env).expanduser().resolve()
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
    candidate = Path(slug_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    root = campaigns_root or resolve_campaigns_dir()
    direct = root / slug_or_path
    if direct.exists():
        return direct.resolve()
    # Glob: <slug> matched as suffix of any campaign dir name.
    for entry in sorted(root.glob(f"*-{slug_or_path}")):
        if entry.is_dir():
            return entry.resolve()
    raise SystemExit(
        f"FATAL: could not resolve campaign {slug_or_path!r}; "
        f"checked {candidate}, {direct}, and *-{slug_or_path} under {root}"
    )


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "campaign"


def utc_timestamp_slug() -> str:
    """``YYYYMMDDTHHMMSSZ`` per the workspace's evidence-bundle-init convention."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
