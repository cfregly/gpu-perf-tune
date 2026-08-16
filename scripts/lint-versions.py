#!/usr/bin/env python3
"""Assert public version surfaces agree with the root VERSION file.

The root VERSION file is the release version source of truth. This lint keeps
the adapter manifest, package metadata, public version banner, runtime version,
and latest changelog entry aligned with it.

How it works:

1. Read and validate the numeric SemVer in ``VERSION``.
2. For each (file, regex) below, extract the X.Y.Z it advertises and compare.
   The regex captures an optional ``v`` prefix; only the numeric triple is
   compared, so "v0.3.0" and "0.3.0" are treated as equal.

Exit codes:
  0 - clean (every surface matches VERSION).
  1 - >=1 header disagrees.
  2 - fatal (VERSION is missing or invalid).

Run from the repo root:

    python3 scripts/lint-versions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
NUMERIC_SEMVER = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")

# (relative doc path, human label, compiled regex with one capture group for X.Y.Z).
VERSION_HEADERS = (
    (
        "plugins/profile-and-optimize/.claude-plugin/plugin.json",
        'Claude adapter manifest "version" field',
        re.compile(r'"version"\s*:\s*"(\d+\.\d+\.\d+)"'),
    ),
    (
        "plugins/profile-and-optimize/README.md",
        'plugin README "Version" banner',
        re.compile(r"\*\*Version\s+v?(\d+\.\d+\.\d+)\*\*"),
    ),
    (
        "plugins/profile-and-optimize/server/pyproject.toml",
        'server package "version" field',
        re.compile(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE),
    ),
    (
        "plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/pyproject.toml",
        'MCP package "version" field',
        re.compile(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE),
    ),
    (
        "plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/__init__.py",
        'MCP package "__version__" field',
        re.compile(r'^__version__\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE),
    ),
    (
        "CHANGELOG.md",
        "latest changelog release",
        re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE),
    ),
)


def release_version() -> str:
    try:
        raw = VERSION_FILE.read_text()
    except OSError as exc:
        print(f"FATAL: cannot read {VERSION_FILE}: {exc}", file=sys.stderr)
        sys.exit(2)
    version = raw.removesuffix("\n")
    if not NUMERIC_SEMVER.fullmatch(version):
        print(
            f"FATAL: {VERSION_FILE} must contain numeric SemVer X.Y.Z "
            "with at most one final newline",
            file=sys.stderr,
        )
        sys.exit(2)
    return version


def main() -> int:
    expected = release_version()
    print(f"[lint-versions] VERSION source of truth: {expected}")
    findings: list[str] = []
    for rel, label, pattern in VERSION_HEADERS:
        path = REPO_ROOT / rel
        if not path.is_file():
            findings.append(f"{rel}: NOT FOUND (still in VERSION_HEADERS, remove or fix path)")
            continue
        m = pattern.search(path.read_text())
        if not m:
            findings.append(f"{rel}: no version banner matched for {label} (regex drift?)")
            continue
        found = m.group(1)
        if found != expected:
            findings.append(f"{rel}: {label} says {found} (expected {expected})")
    if not findings:
        print(f"[ok] every public version surface matches VERSION ({expected})")
        return 0
    print(f"[FAIL] {len(findings)} version drift(s) found:", file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
