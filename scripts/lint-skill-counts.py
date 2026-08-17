#!/usr/bin/env python3
"""Assert every doc that names the skill count agrees with the on-disk tree.

The installable skill directories are the source of truth. This lint rejects
stale totals in public manifests, docs, and command help.

How it works:

1. Walk `plugins/profile-and-optimize/skills/` and count subdirectories with a
   `SKILL.md`. That count is the single source
   of truth.
2. For each doc in `DOCS_TO_LINT` below, grep for any line that names a
   skill count. Accept the line only if the number matches the on-disk
   count.
3. Historical release notes are outside the linted file set.

Exit codes:
  0 - clean (every doc agrees with the on-disk count).
  1 - >=1 doc disagrees.
  2 - fatal (e.g. no skill directories found).

Run from the repo root:

    python3 scripts/lint-skill-counts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "plugins" / "profile-and-optimize" / "skills"

# Docs in scope. Each entry is a path relative to the repo root. Patterns
# below are matched against each doc; any line containing a number
# adjacent to the listed pattern must match the on-disk skill count.
DOCS_TO_LINT = (
    "README.md",
    "CONTRIBUTING.md",
    "REVIEWERS.md",
    "SECURITY.md",
    "Makefile",
    "plugins/profile-and-optimize/README.md",
    "plugins/profile-and-optimize/.claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
)

# Total-count patterns. These intentionally match only claims that present
# themselves as the canonical total. Scoped family breakdowns do not match.
TOTAL_PATTERNS = (
    # "Skills: 44" / "**Skills:** 44"
    re.compile(r"\bSkills:\s*\*{0,2}\s*(\d+)\b"),
    # "44 skills total" / "44 task-oriented skills" / "44 total"
    re.compile(r"\b(\d+)\s+(?:task-oriented\s+)?skills?\s+total\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+task-oriented\s+(?:work)?(?:flows?|skills?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+skills?\s+total\b", re.IGNORECASE),
    # "ships 44 task-oriented skills"
    re.compile(r"\bships\s+(\d+)\s+(?:task-oriented\s+)?skills?\b", re.IGNORECASE),
    # plugin.json "44 skills total" in description prose
    re.compile(r'"description"[^"]*?(\d+)\s+skills?\s+total', re.IGNORECASE),
)

# Word-form total counts (e.g. "Forty-one task-oriented workflows").
SPELLED_OUT_PATTERN = re.compile(
    r"\b(twenty|twenty-one|twenty-two|thirty|thirty-one|thirty-five|forty|forty-one|forty-four)"
    r"\s+task-oriented\s+(?:workflows?|skills?)\b",
    re.IGNORECASE,
)
WORD_TO_INT = {
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "thirty": 30,
    "thirty-one": 31,
    "thirty-five": 35,
    "forty": 40,
    "forty-one": 41,
    "forty-four": 44,
}


def count_on_disk() -> int:
    """Return the on-disk installable skill count."""
    if not SKILLS_DIR.is_dir():
        print(f"FATAL: skills dir not found at {SKILLS_DIR}", file=sys.stderr)
        sys.exit(2)
    count = 0
    for child in SKILLS_DIR.iterdir():
        if not child.is_dir():
            continue
        if not (child / "SKILL.md").is_file():
            continue
        count += 1
    if count == 0:
        print(f"FATAL: no skills found under {SKILLS_DIR}", file=sys.stderr)
        sys.exit(2)
    return count


def lint_doc(path: Path, expected: int) -> list[str]:
    """Return a list of disagreement-finding strings (empty if clean)."""
    findings: list[str] = []
    if not path.is_file():
        return [f"{path}: NOT FOUND (still in DOCS_TO_LINT — remove or fix path)"]
    text = path.read_text()
    for line_no, line in enumerate(text.splitlines(), start=1):
        seen_nums: set[int] = set()
        for pattern in TOTAL_PATTERNS:
            for m in pattern.finditer(line):
                num = int(m.group(1))
                if num == expected or num in seen_nums:
                    continue
                seen_nums.add(num)
                findings.append(f"{path}:{line_no}: reports {num} skills (expected {expected}): {line.strip()[:160]}")
        for m in SPELLED_OUT_PATTERN.finditer(line):
            num = WORD_TO_INT[m.group(1).lower()]
            if num == expected or num in seen_nums:
                continue
            seen_nums.add(num)
            findings.append(
                f"{path}:{line_no}: reports {m.group(0)} ({num}) skills (expected {expected}): {line.strip()[:160]}"
            )
    return findings


def main() -> int:
    expected = count_on_disk()
    print(f"[lint-skill-counts] on-disk count: {expected} skills (under {SKILLS_DIR.relative_to(REPO_ROOT)})")
    all_findings: list[str] = []
    for rel in DOCS_TO_LINT:
        all_findings.extend(lint_doc(REPO_ROOT / rel, expected))
    if not all_findings:
        print(f"[ok] every doc in scope agrees with the on-disk count of {expected}")
        return 0
    print(f"[FAIL] {len(all_findings)} skill-count drift(s) found:", file=sys.stderr)
    for finding in all_findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
