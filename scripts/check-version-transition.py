#!/usr/bin/env python3
"""Reject project version regressions and describe the current transition."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)
LEGACY_VERSION_PATH = "plugins/profile-and-optimize/.claude-plugin/plugin.json"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_version(raw: str, source: str) -> tuple[int, int, int]:
    value = raw.strip()
    if not VERSION_PATTERN.fullmatch(value):
        raise ValueError(f"{source} must contain numeric X.Y.Z SemVer, found {value!r}")
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def read_blob(ref: str, path: str) -> str | None:
    result = git("show", f"{ref}:{path}", check=False)
    return result.stdout if result.returncode == 0 else None


def parent_of(ref: str) -> str | None:
    result = git("rev-parse", f"{ref}^", check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    shallow = git("rev-parse", "--is-shallow-repository", check=False)
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        raise ValueError(
            f"cannot resolve the parent of {ref} in a shallow repository. Fetch full history first"
        )
    return None


def version_at(ref: str) -> tuple[str, str] | None:
    root_version = read_blob(ref, "VERSION")
    if root_version is not None:
        return root_version.strip(), f"{ref}:VERSION"

    manifest = read_blob(ref, LEGACY_VERSION_PATH)
    if manifest is not None:
        try:
            value = json.loads(manifest)["version"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"could not read version from {ref}:{LEGACY_VERSION_PATH}: {error}") from error
        if not isinstance(value, str):
            raise ValueError(f"version in {ref}:{LEGACY_VERSION_PATH} must be a string")
        return value, f"{ref}:{LEGACY_VERSION_PATH}"

    return None


def resolve_transition(ref: str | None) -> tuple[str, str, str | None, str]:
    if ref is not None:
        resolved_ref = git("rev-parse", f"{ref}^{{commit}}").stdout.strip()
        current_blob = read_blob(resolved_ref, "VERSION")
        if current_blob is None:
            raise ValueError(f"{ref} does not contain VERSION")
        current_raw = current_blob.strip()
        base_ref = parent_of(resolved_ref)
        current_source = f"{resolved_ref}:VERSION"
    else:
        version_path = Path("VERSION")
        if not version_path.is_file():
            raise ValueError("VERSION is missing from the worktree")
        current_raw = version_path.read_text(encoding="utf-8").strip()
        head_blob = read_blob("HEAD", "VERSION")
        if head_blob is None or head_blob.strip() != current_raw:
            base_ref = "HEAD"
            current_source = "worktree VERSION"
        else:
            base_ref = parent_of("HEAD")
            current_source = "HEAD:VERSION"

    current_key = parse_version(current_raw, current_source)
    if base_ref is None:
        return "bootstrap", current_raw, None, "no parent version"

    previous = version_at(base_ref)
    if previous is None:
        return "bootstrap", current_raw, None, f"no version surface at {base_ref}"

    previous_raw, previous_source = previous
    previous_key = parse_version(previous_raw, previous_source)
    if current_key > previous_key:
        status = "increase"
    elif current_key == previous_key:
        status = "same"
    else:
        status = "regression"
    return status, current_raw, previous_raw, previous_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", help="check the committed VERSION at this ref")
    parser.add_argument(
        "--require-increase",
        action="store_true",
        help="reject an unchanged version, for an unpublished release candidate",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="print only bootstrap, increase, or same",
    )
    args = parser.parse_args()

    try:
        status, current, previous, previous_source = resolve_transition(args.ref)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"version transition: {error}", file=sys.stderr)
        return 2

    if status == "regression":
        print(
            f"version transition: VERSION {current} is lower than {previous} from {previous_source}",
            file=sys.stderr,
        )
        return 1
    if args.require_increase and status == "same":
        print(
            f"version transition: VERSION {current} does not increase {previous} from {previous_source}",
            file=sys.stderr,
        )
        return 1

    if args.status_only:
        print(status)
    elif status == "bootstrap":
        print(f"[ok] VERSION {current} starts the root VERSION history")
    else:
        print(f"[ok] VERSION {current} is {status} relative to {previous} from {previous_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
