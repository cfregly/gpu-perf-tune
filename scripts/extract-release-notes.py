#!/usr/bin/env python3
"""Print one exact dated CHANGELOG section for a numeric release tag."""

from __future__ import annotations

import argparse
import re
import sys


VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
HEADING = re.compile(
    rf"^## \[({VERSION.pattern})\] - ([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})$"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="numeric version or v-prefixed tag")
    parser.add_argument("changelog", nargs="?", default="CHANGELOG.md")
    args = parser.parse_args()

    requested = args.version.removeprefix("v")
    if not VERSION.fullmatch(requested):
        print(f"release-notes: invalid numeric version {args.version!r}", file=sys.stderr)
        return 2

    path = args.changelog
    try:
        source = argparse.FileType("r", encoding="utf-8")(args.changelog)
    except argparse.ArgumentTypeError as error:
        print(f"release-notes: cannot read {path}: {error}", file=sys.stderr)
        return 2
    try:
        lines = source.read().splitlines()
    finally:
        if source is not sys.stdin:
            source.close()

    start: int | None = None
    for index, line in enumerate(lines):
        match = HEADING.fullmatch(line)
        if match and match.group(1) == requested:
            start = index
            break
    if start is None:
        print(f"release-notes: no dated CHANGELOG section for v{requested}", file=sys.stderr)
        return 1

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## ["):
            end = index
            break
    print("\n".join(lines[start:end]).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
