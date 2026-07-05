"""Helpers for the perf-baseline registry.

Pure functions; no MCP / Slurm / network dependencies. Unit-testable in
isolation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, second precision, suitable for filenames."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_slug() -> str:
    """Compact UTC timestamp suitable as a directory name (no colons)."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_of_path(path: Path) -> str:
    """SHA-256 hex digest of the file (or, for a directory, of its tar-like
    canonical concatenation of every file's bytes + relative path)."""
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for sub in sorted(path.rglob("*")):
            if not sub.is_file():
                continue
            rel = sub.relative_to(path).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            with sub.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        return h.hexdigest()
    raise FileNotFoundError(f"sha256_of_path: {path!s} is neither a file nor a directory")


def registry_dir_for(repo_root: Path, family: str, measurement: str) -> Path:
    """Return ``<repo_root>/experiments/artifacts/perf-baselines/<family>/<measurement>/``."""
    return repo_root / "experiments" / "artifacts" / "perf-baselines" / family / measurement


def registry_entry_dir(repo_root: Path, family: str, measurement: str, slug: str) -> Path:
    """Return the immutable per-registration directory path."""
    return registry_dir_for(repo_root, family, measurement) / slug


def write_baseline_json(
    entry_dir: Path,
    *,
    family: str,
    measurement: str,
    value: float | None,
    unit: str | None,
    source_path: Path,
    source_sha256: str,
    schema_path: Path | None,
    registered_at_utc: str,
    operator_user: str,
    hostname: str,
    uname: str,
    profile_and_optimize_sha: str | None,
    notes: str,
) -> Path:
    """Write the canonical baseline.json into the entry directory.

    Caller is responsible for ensuring entry_dir exists.
    """
    payload: dict[str, Any] = {
        "family": family,
        "measurement": measurement,
        "value": value,
        "unit": unit,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "schema_path": str(schema_path) if schema_path is not None else None,
        "registered_at_utc": registered_at_utc,
        "registered_by": {
            "team": "the MLPerf team",
            "operator_user": operator_user,
        },
        "workstation": {"hostname": hostname, "uname": uname},
        "profile_and_optimize_sha": profile_and_optimize_sha,
        "notes": notes,
    }
    out = entry_dir / "baseline.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def write_source_md(
    entry_dir: Path,
    *,
    family: str,
    measurement: str,
    operator_user: str,
    hostname: str,
    registered_at_utc: str,
    profile_and_optimize_sha: str | None,
    source_path: Path,
    source_sha256: str,
    notes: str,
) -> Path:
    """Write the human-readable SOURCE.md provenance file."""
    sha_short = (profile_and_optimize_sha or "(unknown)")[:12]
    body = f"""# SOURCE: perf-baseline {family} / {measurement}

**Registered at (UTC):** `{registered_at_utc}`
**Registered by:** the MLPerf team (operator: `{operator_user}` on `{hostname}`)
**profile-and-optimize SHA at registration:** `{sha_short}`

## Source

- Path at registration time: `{source_path}`
- SHA-256: `{source_sha256}`
- A canonical snapshot of the source is stored next to this file as
  `source-snapshot.<ext>` (or `source-snapshot/` if the source was a directory).

## Notes

{notes or "_(none)_"}

## Per workspace CLAUDE.md "Team Attribution"

This baseline is attributed to the MLPerf team, not an individual. The operator user
above is captured as audit trail; no individual is asserted as "submission
lead" / "submission owner" / "MLPerf lead".
"""
    out = entry_dir / "SOURCE.md"
    out.write_text(body)
    return out


def append_index(registry_dir: Path, *, slug: str, registered_at_utc: str, value: float | None, unit: str | None, notes: str) -> Path:
    """Append a row to INDEX.md (creating it with a header on first write)."""
    index_path = registry_dir / "INDEX.md"
    if not index_path.exists():
        index_path.write_text(
            "# Perf-baseline registry index\n\n"
            "Chronological history. One row per registration. Newer entries at the bottom.\n\n"
            "| UTC timestamp | Slug | Value | Unit | Notes |\n"
            "| --- | --- | --- | --- | --- |\n"
        )
    value_str = "_(structured)_" if value is None else f"`{value}`"
    unit_str = "_(none)_" if not unit else f"`{unit}`"
    notes_short = (notes[:60] + "...") if notes and len(notes) > 60 else (notes or "")
    row = f"| `{registered_at_utc}` | [`{slug}`]({slug}/) | {value_str} | {unit_str} | {notes_short} |\n"
    with index_path.open("a") as f:
        f.write(row)
    return index_path


def discover_profile_and_optimize_sha(repo_root: Path) -> str | None:
    """Attempt to discover the current git SHA of the bundled server.

    Walks up from repo_root looking for a .git directory; if found, reads HEAD.
    Returns None if git metadata is unavailable (e.g. the plugin was installed
    via tarball rather than git clone).
    """
    current = repo_root.resolve()
    while current != current.parent:
        git_dir = current / ".git"
        if git_dir.exists():
            try:
                head = (git_dir / "HEAD").read_text().strip()
                if head.startswith("ref: "):
                    ref_path = git_dir / head[len("ref: "):]
                    if ref_path.exists():
                        return ref_path.read_text().strip()
                else:
                    return head
            except OSError:
                return None
        current = current.parent
    return None


def gather_workstation_facts() -> tuple[str, str, str]:
    """Return (hostname, uname-string, operator-user)."""
    hostname = socket.gethostname()
    uname = " ".join(os.uname()) if hasattr(os, "uname") else "(unavailable)"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "(unknown)"
    return hostname, uname, user
