#!/usr/bin/env python3
"""Validate experiment provenance blocks for the commit hook."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _server_root() -> Path:
    return Path(__file__).resolve().parents[1] / "server"


def _source_path(repo_root: Path, directory: str) -> tuple[Path, str]:
    candidate = (repo_root / directory / "SOURCE.md").resolve()
    try:
        relative = candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"changed path leaves repository: {directory!r}") from exc
    return candidate, relative.as_posix()


def _staged_source_text(repo_root: Path, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f":{relative_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "path is not present in the Git index"
        raise ValueError(f"unable to read staged file: {detail}")
    return result.stdout


def audit(repo_root: Path, directories: list[str]) -> list[str]:
    server_root = _server_root()
    if not server_root.is_dir():
        return [f"profile_and_optimize server root not found: {server_root}"]
    sys.path.insert(0, str(server_root))

    try:
        from tools.perf_tune_report.provenance import parse_text, validate
    except Exception as exc:  # noqa: BLE001 - enabled safety gate must fail closed
        return [f"unable to load provenance validator: {exc}"]

    problems: list[str] = []
    for directory in directories:
        try:
            _, relative_path = _source_path(repo_root, directory)
            provenance = parse_text(_staged_source_text(repo_root, relative_path))
        except Exception as exc:  # noqa: BLE001 - malformed staged input must fail closed
            problems.append(f"{directory}/SOURCE.md: {exc}")
            continue
        if provenance is None:
            problems.append(f"{directory}/SOURCE.md: missing provenance block")
            continue
        for problem in validate(provenance):
            problems.append(f"{directory}/SOURCE.md: {problem}")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--gate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--changed-only", nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        print(f"repository root not found: {repo_root}", file=sys.stderr)
        return 2
    problems = audit(repo_root, args.changed_only)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
