"""Perf-baseline registry: record + diff.

Workload-agnostic. Two verbs:

- ``record`` writes a new immutable registry entry under
  ``<repo_root>/experiments/artifacts/perf-baselines/<family>/<measurement>/<slug>/``
  with provenance (sha256, git-SHA, operator user, hostname, UTC timestamp).
- ``diff`` compares a current measurement against a registered baseline,
  returning per-dimension deltas + a GREEN / YELLOW / RED verdict.

See the skills [``perf-baseline-record``](../../../skills/perf-baseline-record/SKILL.md)
and [``perf-baseline-diff``](../../../skills/perf-baseline-diff/SKILL.md) for
the operator-facing workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from jsonschema import SchemaError, ValidationError
from jsonschema import validate as validate_json

from tools.perf_baseline.helpers import (
    append_index,
    diff_entry_dir,
    discover_profile_and_optimize_sha,
    gather_workstation_facts,
    registry_dir_for,
    registry_entry_dir,
    sha256_of_path,
    utc_now_iso,
    utc_now_slug,
    write_baseline_json,
    write_source_md,
)

CONTRACT: dict[str, dict[str, Any]] = {
    "record": {
        "safety": "writes_artifacts",
        "required": ("--family", "--measurement", "--source", "--direction"),
        "optional": ("--value", "--unit", "--schema", "--notes", "--repo-root", "--json"),
        "json": True,
        "ack": None,
        "description": "Register a new perf-baseline entry under experiments/artifacts/perf-baselines/.",
    },
    "diff": {
        "safety": "writes_artifacts",
        "required": ("--baseline", "--current"),
        "optional": ("--tolerance-percent", "--tolerance-absolute", "--repo-root", "--json"),
        "json": True,
        "ack": None,
        "description": "Diff a current measurement against a registered baseline.",
    },
}

DIRECTIONS = {"two-sided", "higher-is-better", "lower-is-better"}


def _operator_path(value: str, *, label: str) -> Path:
    """Normalize a complete path chosen by the local CLI operator."""
    if not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty local path without NUL bytes")

    # codeql[py/path-injection]
    return Path(value).expanduser().resolve()


def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return _operator_path(arg, label="--repo-root")
    env = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    if env:
        return _operator_path(env, label="PROFILE_AND_OPTIMIZE_REPO_ROOT")
    # Fall back to walking up from cwd using product files as markers.
    current = Path.cwd().resolve()
    while current != current.parent:
        if (
            (current / "pyproject.toml").is_file()
            and (current / "mcp_surface.py").is_file()
            and (current / "tools").is_dir()
        ):
            return current
        current = current.parent
    raise SystemExit("FATAL: cannot resolve repo root; pass --repo-root or set PROFILE_AND_OPTIMIZE_REPO_ROOT")


def _emit(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for k, v in sorted(payload.items()):
            print(f"{k}: {v}")


def _validate_source_against_schema(source: Path, schema_path: Path) -> None:
    if not source.is_file():
        raise ValueError("--schema requires --source to be a JSON file")
    try:
        source_payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"--source is not valid JSON: {error}") from error
    try:
        schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"--schema is not valid JSON: {error}") from error
    try:
        validate_json(instance=source_payload, schema=schema_payload)
    except SchemaError as error:
        raise ValueError(f"--schema is not a valid JSON Schema: {error.message}") from error
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"--source fails JSON Schema at {location}: {error.message}") from error


def cmd_record(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"FATAL: --source does not exist: {source}", file=sys.stderr)
        return 2
    if args.value is not None and not math.isfinite(args.value):
        print("FATAL: --value must be a finite number", file=sys.stderr)
        return 2
    direction = getattr(args, "direction", None)
    if direction not in DIRECTIONS:
        print(
            "FATAL: --direction must be two-sided, higher-is-better, or lower-is-better",
            file=sys.stderr,
        )
        return 2

    schema_path: Path | None = None
    if args.schema:
        schema_path = Path(args.schema).expanduser().resolve()
        if not schema_path.is_file():
            print(f"FATAL: --schema must be a file: {schema_path}", file=sys.stderr)
            return 2
        try:
            _validate_source_against_schema(source, schema_path)
        except ValueError as error:
            print(f"FATAL: {error}", file=sys.stderr)
            return 2

    source_sha256 = sha256_of_path(source)
    slug = utc_now_slug()
    registered_at_utc = utc_now_iso()
    hostname, uname, operator_user = gather_workstation_facts()
    profile_and_optimize_sha = discover_profile_and_optimize_sha(repo_root)

    try:
        entry_dir = registry_entry_dir(repo_root, args.family, args.measurement, slug)
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    if entry_dir.exists():
        # Pathologically unlikely (same-second collision) but be defensive.
        print(f"FATAL: registry entry already exists: {entry_dir}", file=sys.stderr)
        return 2
    entry_dir.mkdir(parents=True, exist_ok=False)

    # Snapshot the source.
    if source.is_file():
        snap = entry_dir / f"source-snapshot{source.suffix}"
        shutil.copy2(source, snap)
    else:
        snap = entry_dir / "source-snapshot"
        shutil.copytree(source, snap)

    baseline_path = write_baseline_json(
        entry_dir,
        family=args.family,
        measurement=args.measurement,
        value=args.value,
        direction=direction,
        unit=args.unit,
        source_path=source,
        source_sha256=source_sha256,
        schema_path=schema_path,
        registered_at_utc=registered_at_utc,
        operator_user=operator_user,
        hostname=hostname,
        uname=uname,
        profile_and_optimize_sha=profile_and_optimize_sha,
        notes=args.notes or "",
    )
    write_source_md(
        entry_dir,
        family=args.family,
        measurement=args.measurement,
        operator_user=operator_user,
        hostname=hostname,
        registered_at_utc=registered_at_utc,
        profile_and_optimize_sha=profile_and_optimize_sha,
        source_path=source,
        source_sha256=source_sha256,
        notes=args.notes or "",
    )
    append_index(
        registry_dir_for(repo_root, args.family, args.measurement),
        slug=slug,
        registered_at_utc=registered_at_utc,
        value=args.value,
        unit=args.unit,
        notes=args.notes or "",
    )

    payload = {
        "tool": "perf_baseline_record",
        "library": "perf_baseline",
        "verb": "record",
        "safety": CONTRACT["record"]["safety"],
        "entry_dir": str(entry_dir),
        "baseline_json": str(baseline_path),
        "slug": slug,
        "source_sha256": source_sha256,
        "registered_at_utc": registered_at_utc,
        "direction": direction,
    }
    _emit(payload, as_json=args.json)
    return 0


def _read_baseline(baseline_dir: Path) -> dict[str, Any]:
    p = baseline_dir / "baseline.json"
    if not p.exists():
        raise SystemExit(f"FATAL: not a baseline directory (no baseline.json): {baseline_dir}")
    return json.loads(p.read_text())


def _delta_percent(baseline: float, current: float) -> float | None:
    if baseline == 0.0:
        return 0.0 if current == 0.0 else None
    return 100.0 * (current - baseline) / baseline


def _exceeds_tolerance(
    delta: dict[str, Any],
    tolerance_pct: float,
    tolerance_absolute: float | None,
    direction: str,
) -> bool:
    if tolerance_absolute is not None:
        magnitude_exceeds = abs(float(delta["delta"])) > tolerance_absolute
    else:
        delta_pct = delta["delta_pct"]
        magnitude_exceeds = delta_pct is None or abs(float(delta_pct)) > tolerance_pct
    if not magnitude_exceeds:
        return False
    numeric_delta = float(delta["delta"])
    if direction == "higher-is-better":
        return numeric_delta < 0
    if direction == "lower-is-better":
        return numeric_delta > 0
    return True


def _classify(
    deltas: list[dict[str, Any]],
    tolerance_pct: float,
    tolerance_absolute: float | None,
    direction: str,
) -> str:
    over = [delta for delta in deltas if _exceeds_tolerance(delta, tolerance_pct, tolerance_absolute, direction)]
    if not over:
        return "GREEN"
    if any(delta.get("headline") is True for delta in over):
        return "RED"
    if len(over) <= 2:
        return "YELLOW"
    return "RED"


def _render_diff_markdown(
    *,
    family: str,
    measurement: str,
    direction: str,
    tolerance_pct: float,
    tolerance_absolute: float | None,
    verdict: str,
    deltas: list[dict[str, Any]],
) -> str:
    """Render the promised human-readable diff artifact."""
    tolerance = f"absolute {tolerance_absolute:g}" if tolerance_absolute is not None else f"{tolerance_pct:g}%"
    lines = [
        f"# Performance baseline diff: {family} / {measurement}",
        "",
        f"- Direction: `{direction}`",
        f"- Tolerance: `{tolerance}`",
        f"- Verdict: **{verdict}**",
        "",
        "| Key | Baseline | Current | Delta | Delta % | Adverse beyond tolerance |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for delta in deltas[:20]:
        delta_pct = delta.get("delta_pct")
        delta_pct_text = "n/a" if delta_pct is None else f"{float(delta_pct):.6g}%"
        adverse = _exceeds_tolerance(
            delta,
            tolerance_pct,
            tolerance_absolute,
            direction,
        )
        lines.append(
            f"| {delta['key']} | {float(delta['baseline']):.6g} | "
            f"{float(delta['current']):.6g} | {float(delta['delta']):.6g} | "
            f"{delta_pct_text} | {'yes' if adverse else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "GREEN means no adverse change exceeded tolerance."
                if direction != "two-sided"
                else "GREEN means no two-sided deviation exceeded tolerance."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def cmd_diff(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    baseline_dir = Path(args.baseline).expanduser().resolve()
    current = Path(args.current).expanduser().resolve()

    if not baseline_dir.is_dir():
        print(f"FATAL: --baseline must be a registry entry directory: {baseline_dir}", file=sys.stderr)
        return 2
    if not current.exists():
        print(f"FATAL: --current does not exist: {current}", file=sys.stderr)
        return 2

    baseline = _read_baseline(baseline_dir)
    family = baseline.get("family")
    measurement = baseline.get("measurement")
    if not isinstance(family, str) or not isinstance(measurement, str):
        print("FATAL: baseline family and measurement must be strings", file=sys.stderr)
        return 2
    direction = baseline.get("direction", "two-sided")
    if direction not in DIRECTIONS:
        print(f"FATAL: baseline direction is invalid: {direction!r}", file=sys.stderr)
        return 2
    tolerance_pct = float(args.tolerance_percent)
    tolerance_absolute = args.tolerance_absolute
    if not math.isfinite(tolerance_pct) or tolerance_pct < 0:
        print("FATAL: --tolerance-percent must be a finite nonnegative number", file=sys.stderr)
        return 2
    if tolerance_absolute is not None and (not math.isfinite(tolerance_absolute) or tolerance_absolute < 0):
        print("FATAL: --tolerance-absolute must be a finite nonnegative number", file=sys.stderr)
        return 2
    current_sha = sha256_of_path(current)

    # Determine shape.
    baseline_value = baseline.get("value")
    if baseline_value is not None:
        # Scalar shape.
        try:
            baseline_value = float(baseline_value)
        except (TypeError, ValueError):
            print("FATAL: scalar baseline value must be numeric", file=sys.stderr)
            return 2
        if not math.isfinite(baseline_value):
            print("FATAL: scalar baseline value must be finite", file=sys.stderr)
            return 2
        try:
            current_value = float(current.read_text().strip()) if current.is_file() else None
        except ValueError:
            current_value = None
        if current_value is None:
            # Try to read it as JSON with a "value" field.
            try:
                parsed = json.loads(current.read_text()) if current.is_file() else None
                if isinstance(parsed, dict) and "value" in parsed:
                    current_value = float(parsed["value"])
            except Exception:  # noqa: BLE001
                current_value = None
        if current_value is None:
            print(
                'FATAL: baseline is scalar but --current did not parse as a number or {"value": ...} JSON',
                file=sys.stderr,
            )
            return 2
        if not math.isfinite(current_value):
            print("FATAL: current scalar value must be finite", file=sys.stderr)
            return 2
        delta = current_value - baseline_value
        delta_pct = _delta_percent(baseline_value, current_value)
        deltas = [
            {
                "key": measurement,
                "baseline": baseline_value,
                "current": current_value,
                "delta": delta,
                "delta_pct": delta_pct,
                "headline": True,
            }
        ]
    else:
        # Structured dict shape: baseline source-snapshot is JSON; current is JSON.
        snap = next(baseline_dir.glob("source-snapshot*"), None)
        if snap is None or not snap.is_file():
            print(
                "FATAL: baseline is structured but source-snapshot is missing or a "
                "directory. Structured directory diff is not supported.",
                file=sys.stderr,
            )
            return 2
        try:
            baseline_dict = json.loads(snap.read_text())
            current_dict = json.loads(current.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FATAL: structured diff requires JSON inputs; {exc}", file=sys.stderr)
            return 2
        if not isinstance(baseline_dict, dict) or not isinstance(current_dict, dict):
            print("FATAL: structured diff requires both inputs to be JSON objects (key->value)", file=sys.stderr)
            return 2
        deltas = []
        for k in set(baseline_dict) & set(current_dict):
            try:
                b = float(baseline_dict[k])
                c = float(current_dict[k])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(b) or not math.isfinite(c):
                continue
            d = c - b
            dp = _delta_percent(b, c)
            deltas.append({"key": k, "baseline": b, "current": c, "delta": d, "delta_pct": dp})
        if not deltas:
            print("FATAL: structured diff found no shared finite numeric values", file=sys.stderr)
            return 2
        deltas.sort(
            key=lambda delta: (
                -abs(float(delta["delta"]))
                if tolerance_absolute is not None
                else -(math.inf if delta["delta_pct"] is None else abs(float(delta["delta_pct"]))),
                str(delta["key"]),
            )
        )

    verdict = _classify(deltas, tolerance_pct, tolerance_absolute, direction)

    # Write the diff bundle.
    try:
        diff_root = diff_entry_dir(
            repo_root,
            family,
            measurement,
            utc_now_slug(),
        )
    except ValueError as error:
        print(f"FATAL: invalid baseline metadata: {error}", file=sys.stderr)
        return 2
    diff_root.mkdir(parents=True, exist_ok=False)
    diff_json = {
        "tool": "perf_baseline_diff",
        "library": "perf_baseline",
        "verb": "diff",
        "safety": CONTRACT["diff"]["safety"],
        "family": family,
        "measurement": measurement,
        "baseline_dir": str(baseline_dir),
        "baseline_source_sha256": baseline.get("source_sha256"),
        "current_source_sha256": current_sha,
        "tolerance_percent": tolerance_pct,
        "tolerance_absolute": tolerance_absolute,
        "direction": direction,
        "verdict": verdict,
        "deltas_top20": deltas[:20],
        "deltas_count": len(deltas),
    }
    (diff_root / "diff.json").write_text(json.dumps(diff_json, indent=2, sort_keys=True) + "\n")
    (diff_root / "baseline-ref.txt").write_text(str(baseline_dir) + "\n")
    if current.is_file():
        shutil.copy2(current, diff_root / f"current-snapshot{current.suffix}")
    else:
        shutil.copytree(current, diff_root / "current-snapshot")
    (diff_root / "diff.md").write_text(
        _render_diff_markdown(
            family=family,
            measurement=measurement,
            direction=direction,
            tolerance_pct=tolerance_pct,
            tolerance_absolute=tolerance_absolute,
            verdict=verdict,
            deltas=deltas,
        )
    )

    payload = {
        "tool": "perf_baseline_diff",
        "library": "perf_baseline",
        "verb": "diff",
        "safety": CONTRACT["diff"]["safety"],
        "diff_dir": str(diff_root),
        "verdict": verdict,
        "deltas_count": len(deltas),
        "tolerance_percent": tolerance_pct,
        "tolerance_absolute": tolerance_absolute,
        "direction": direction,
    }
    _emit(payload, as_json=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and diff perf baselines under experiments/artifacts/perf-baselines/."
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    record = sub.add_parser("record", description=CONTRACT["record"]["description"])
    record.add_argument("--family", required=True, help="e.g. llama31_8b, gb300-cluster, deepseek-v3-inference")
    record.add_argument("--measurement", required=True, help="e.g. nccl_busbw, step_time, nvlink_pairwise_bw")
    record.add_argument("--source", required=True, help="Path to the source data file or directory")
    record.add_argument(
        "--value", type=float, default=None, help="Scalar value (for scalar baselines); omit for structured"
    )
    record.add_argument("--unit", default=None, help="Unit string (e.g. GB/s, ms, tokens/s, mfu)")
    record.add_argument(
        "--direction",
        choices=tuple(sorted(DIRECTIONS)),
        required=True,
        help="Comparison direction. Choose two-sided only when either deviation is adverse.",
    )
    record.add_argument("--schema", default=None, help="Optional path to a JSON schema for the source")
    record.add_argument("--notes", default=None, help="Free-text notes captured in baseline.json + SOURCE.md")
    record.add_argument("--repo-root", default=None, help="Override PROFILE_AND_OPTIMIZE_REPO_ROOT")
    record.add_argument("--json", action="store_true", help="Emit JSON envelope")
    record.set_defaults(func=cmd_record)

    diff = sub.add_parser("diff", description=CONTRACT["diff"]["description"])
    diff.add_argument("--baseline", required=True, help="Path to a registered baseline entry directory")
    diff.add_argument("--current", required=True, help="Path to the current measurement (file)")
    diff.add_argument("--tolerance-percent", type=float, default=5.0, help="Per-dimension tolerance %% (default: 5)")
    diff.add_argument(
        "--tolerance-absolute",
        type=float,
        default=None,
        help="Absolute tolerance. When set, it replaces percent tolerance for classification.",
    )
    diff.add_argument("--repo-root", default=None, help="Override PROFILE_AND_OPTIMIZE_REPO_ROOT")
    diff.add_argument("--json", action="store_true", help="Emit JSON envelope")
    diff.set_defaults(func=cmd_diff)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
