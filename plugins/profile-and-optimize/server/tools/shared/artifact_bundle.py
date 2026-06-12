#!/usr/bin/env python3
"""Validate durable MLPerf artifact bundle shape.

This is intentionally a small structural validator instead of a JSON Schema
runtime dependency. The schema file under ``tuning/schemas/`` documents the
contract for reviewers; this module enforces the same required files in the
operator tools and campaign DAG.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("summary.md", "SOURCE.md", "run-context.json")
OPTIONAL_RAW_DIRS = ("raw", "provenance")


@dataclass(frozen=True)
class BundleIssue:
    path: Path
    message: str

    def render(self) -> str:
        return f"{self.path}: {self.message}"


def _load_json(path: Path, issues: list[BundleIssue]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(BundleIssue(path, f"failed to read JSON: {exc}"))
        return None
    if not isinstance(payload, dict):
        issues.append(BundleIssue(path, "must contain a JSON object"))
        return None
    return payload


def validate_bundle_shape(
    bundle_dir: Path,
    *,
    expected_family: str | None = None,
    expected_run_id: str | None = None,
) -> list[BundleIssue]:
    """Return structural issues for one artifact bundle."""

    issues: list[BundleIssue] = []
    if not bundle_dir.is_dir():
        return [BundleIssue(bundle_dir, "missing artifact bundle directory")]

    for name in REQUIRED_FILES:
        path = bundle_dir / name
        if not path.is_file():
            issues.append(BundleIssue(path, "missing required artifact bundle file"))

    summary_path = bundle_dir / "summary.md"
    if summary_path.is_file() and not summary_path.read_text(encoding="utf-8", errors="replace").strip():
        issues.append(BundleIssue(summary_path, "summary.md must not be empty"))

    source_path = bundle_dir / "SOURCE.md"
    if source_path.is_file() and not source_path.read_text(encoding="utf-8", errors="replace").strip():
        issues.append(BundleIssue(source_path, "SOURCE.md must not be empty"))

    context_path = bundle_dir / "run-context.json"
    context = _load_json(context_path, issues) if context_path.is_file() else None
    if context is not None:
        for key in ("schema_version", "family", "run_id", "created_at_utc"):
            value = context.get(key)
            if value in (None, ""):
                issues.append(BundleIssue(context_path, f"missing required field {key!r}"))
        if expected_family is not None and context.get("family") != expected_family:
            issues.append(
                BundleIssue(
                    context_path,
                    f"family {context.get('family')!r} does not match expected {expected_family!r}",
                )
            )
        if expected_run_id is not None and context.get("run_id") != expected_run_id:
            issues.append(
                BundleIssue(
                    context_path,
                    f"run_id {context.get('run_id')!r} does not match expected {expected_run_id!r}",
                )
            )

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--family", help="Expected run-context.json family")
    parser.add_argument("--run-id", help="Expected run-context.json run_id")
    args = parser.parse_args(argv)

    issues = validate_bundle_shape(
        args.bundle_dir.resolve(),
        expected_family=args.family,
        expected_run_id=args.run_id,
    )
    if issues:
        print("Artifact bundle shape validation failed:")
        for issue in issues:
            print(f"- {issue.render()}")
        return 1
    print(f"Artifact bundle shape validation passed: {args.bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
