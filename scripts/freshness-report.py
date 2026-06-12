#!/usr/bin/env python3
"""Emit a per-skill freshness report based on `last_validated` frontmatter.

Output: one row per SKILL.md sorted by days-since-validation (oldest first).

Severity buckets (configurable via flags):
  > 180 days: RED  (skill content is more than 6 months old; re-validate)
  >  90 days: YELLOW
  <= 90 days: GREEN

Exit codes:
  0 = all GREEN
  1 = >=1 YELLOW (does not fail; informational)
  2 = >=1 RED   (fails CI)

Per workspace convention, RED is reserved for profile-and-optimize-authored skills;
adapted skills get a free pass since their freshness tracks their upstream
source, not this repo's validation cadence.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "plugins" / "profile-and-optimize" / "skills"


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    fm: dict[str, str] = {}
    for line in text[4:end].splitlines():
        m = re.match(r"^(\w[\w_-]*)\s*:\s*(.+?)\s*$", line)
        if m:
            fm[m.group(1)] = m.group(2)
    return fm


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-skill freshness report.")
    parser.add_argument("--yellow-days", type=int, default=90)
    parser.add_argument("--red-days", type=int, default=180)
    parser.add_argument("--include-template", action="store_true", help="Also report on _template/SKILL.md (skip by default)")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    rows: list[tuple[int, str, str, bool]] = []  # (days, skill, date_str, is_adapted)
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        if skill_md.parent.name == "_template" and not args.include_template:
            continue
        text = skill_md.read_text()
        fm = _parse_frontmatter(text)
        lv = fm.get("last_validated", "")
        try:
            lv_date = datetime.strptime(lv, "%Y-%m-%d").date()
        except ValueError:
            rows.append((-1, skill_md.parent.name, lv or "(missing)", False))
            continue
        days = (today - lv_date).days
        is_adapted = "## Origin" in text
        rows.append((days, skill_md.parent.name, lv, is_adapted))

    rows.sort(key=lambda r: -r[0])

    print(f"# SKILL.md freshness report (today: {today})\n")
    print("| Days | Skill | last_validated | Adapted? | Bucket |")
    print("| --- | --- | --- | --- | --- |")
    red_count = 0
    yellow_count = 0
    green_count = 0
    for days, skill, lv, adapted in rows:
        if days < 0:
            bucket = "MISSING"
            red_count += 1
        elif days > args.red_days:
            bucket = "RED" if not adapted else "RED (adapted; refresh from upstream)"
            if not adapted:
                red_count += 1
            else:
                yellow_count += 1
        elif days > args.yellow_days:
            bucket = "YELLOW"
            yellow_count += 1
        else:
            bucket = "GREEN"
            green_count += 1
        adapted_str = "yes" if adapted else "no"
        print(f"| {days if days >= 0 else '?'} | {skill} | {lv} | {adapted_str} | {bucket} |")

    print()
    print(f"## Summary: {green_count} GREEN / {yellow_count} YELLOW / {red_count} RED / total {len(rows)}")
    print()
    print(f"Thresholds: YELLOW > {args.yellow_days}d, RED > {args.red_days}d.")
    print(f"Adapted skills get RED -> WARN (refresh from upstream).")

    if red_count > 0:
        return 2
    if yellow_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
