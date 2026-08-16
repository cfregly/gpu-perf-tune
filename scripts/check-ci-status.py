#!/usr/bin/env python3
"""Require one successful ci push workflow run for the exact release SHA."""

from __future__ import annotations

import json
import sys


def main() -> int:
    sha = sys.argv[1] if len(sys.argv) == 2 else "requested commit"
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        print(f"release: could not parse GitHub check runs: {error}", file=sys.stderr)
        return 2

    if not isinstance(payload, list):
        print("release: expected a JSON list of GitHub Actions runs", file=sys.stderr)
        return 2

    matching = [
        run
        for run in payload
        if run.get("workflowName") == "ci"
        and run.get("event") == "push"
        and run.get("headBranch") == "main"
        and run.get("headSha") == sha
    ]
    if not any(
        run.get("status") == "completed" and run.get("conclusion") == "success"
        for run in matching
    ):
        states = ", ".join(
            f"{run.get('status')}/{run.get('conclusion')}"
            for run in matching
        ) or "missing"
        print(
            f"release: no successful ci push workflow run for {sha} ({states})",
            file=sys.stderr,
        )
        return 1

    print(f"[ok] ci push workflow succeeded for {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
