#!/usr/bin/env python3
"""Assert the human-facing version headers agree with plugin.json's version.

Background: the v1.29.0 rescan found plugin.json at v1.28.0 while
README.md's "Status:" line still said v1.22.0, the plugin README's "Version"
header said 1.15.0, and CATALOG.md's "Plugin version:" said v1.27.0. The
count lints (`lint-tool-counts.py`, `lint-skill-counts.py`) never looked at
the version headers, so the drift sailed through `make all`. This lint closes
that gap: plugin.json is the single source of truth for the version, and every
doc that prints a top-of-file version banner must match it.

How it works:

1. Read ``plugins/profile-and-optimize/.claude-plugin/plugin.json`` -> ``version``.
2. For each (doc, regex) below, extract the X.Y.Z it advertises and compare.
   The regex captures an optional ``v`` prefix; only the numeric triple is
   compared, so "v1.29.0" and "1.29.0" are treated as equal.

Exit codes:
  0 - clean (every header matches plugin.json).
  1 - >=1 header disagrees.
  2 - fatal (cannot read plugin.json / version field).

Run from the repo root:

    python3 scripts/lint-versions.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_JSON = REPO_ROOT / "plugins" / "profile-and-optimize" / ".claude-plugin" / "plugin.json"

# (relative doc path, human label, compiled regex with one capture group for X.Y.Z).
VERSION_HEADERS = (
    (
        "plugins/profile-and-optimize/README.md",
        'plugin README "Version" banner',
        re.compile(r"\*\*Version\s+v?(\d+\.\d+\.\d+)\*\*"),
    ),
)


def plugin_version() -> str:
    try:
        data = json.loads(PLUGIN_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FATAL: cannot read {PLUGIN_JSON}: {exc}", file=sys.stderr)
        sys.exit(2)
    version = data.get("version")
    if not version:
        print(f"FATAL: no `version` field in {PLUGIN_JSON}", file=sys.stderr)
        sys.exit(2)
    return str(version)


def main() -> int:
    expected = plugin_version()
    print(f"[lint-versions] plugin.json version (source of truth): {expected}")
    findings: list[str] = []
    for rel, label, pattern in VERSION_HEADERS:
        path = REPO_ROOT / rel
        if not path.is_file():
            findings.append(f"{rel}: NOT FOUND (still in VERSION_HEADERS — remove or fix path)")
            continue
        m = pattern.search(path.read_text())
        if not m:
            findings.append(f"{rel}: no version banner matched for {label} (regex drift?)")
            continue
        found = m.group(1)
        if found != expected:
            findings.append(f"{rel}: {label} says {found} (expected {expected})")
    if not findings:
        print(f"[ok] every version banner matches plugin.json ({expected})")
        return 0
    print(f"[FAIL] {len(findings)} version drift(s) found:", file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
