"""CLI entry point for the fleet-wide hang detector.

Usage::

    python -m tools.profiling.hang_detector \\
        --fixture tools/profiling/hang_detector/tests/fixtures/gpusd-snapshot-2048n-mod32-hang.json \\
        --stride 32 --lag-threshold 1 --json

    # Live cluster (operator opt-in):
    python -m tools.profiling.hang_detector \\
        --live-cluster --jobid 11523 --stride 32 --output /mnt/data/.../timeline.jsonl

See ``docs/profiling-and-perf-discovery.md`` "Piece (b): Fleet-wide
hang detector" for the design rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import run_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.profiling.hang_detector",
        description=(
            "Fleet-wide NCCL collective hang detector. Reads GPUSD per-rank "
            "metadata, buckets ranks by `rank % stride`, and flags buckets "
            "whose median seq_num lags the leader bucket."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture",
        type=Path,
        help="Path to a GPUSD-shaped fixture JSON. See gpusd_scraper.py "
        "for the schema. Mutually exclusive with --live-cluster.",
    )
    source.add_argument(
        "--live-cluster",
        action="store_true",
        help="Scrape live GPUSD endpoints over the cluster network. "
        "Requires --nodelist-file and the `requests` package.",
    )

    parser.add_argument(
        "--nodelist-file",
        type=Path,
        help=(
            "When --live-cluster is set, read one hostname per line from "
            "this file. Operator's recipe: capture the active job's "
            "NodeList via sacct (use --format=NodeList with a wide width "
            "specifier) and split commas to newlines. See "
            "tools/profiling/README.md for the exact one-liner."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9420,
        help="GPUSD metrics endpoint port (default 9420).",
    )

    stride_group = parser.add_mutually_exclusive_group()
    stride_group.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Bucket modulus (default 32 matches the MOD-32 hang signature). "
        "Mutually exclusive with --auto-stride.",
    )
    stride_group.add_argument(
        "--auto-stride",
        dest="auto_stride",
        action="store_true",
        help="Sweep a small set of candidate strides instead of one. Defaults "
        "to [8, 16, 32, 64]; override via --candidate-strides. Mutually "
        "exclusive with --stride. Useful when investigating a hang whose "
        "stride pattern is not known up front.",
    )
    parser.add_argument(
        "--candidate-strides",
        dest="candidate_strides",
        default="8,16,32,64",
        help="Comma-separated list of strides to try under --auto-stride "
        "(default: 8,16,32,64). Ignored when --auto-stride is not set.",
    )
    parser.add_argument(
        "--lag-threshold",
        type=int,
        default=1,
        help="Minimum bucket-median seq_num lag to flag (default 1).",
    )

    parser.add_argument(
        "--jobid",
        default="unknown",
        help="Identifier carried through to the JSONL row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="When set, append the result as one JSONL row to this path. "
        "Parent directory must exist.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full result dict to stdout (default just emits a "
        "one-line summary).",
    )
    return parser


def _parse_candidate_strides(spec: str) -> list[int]:
    """Parse a comma-separated stride list. Raises ValueError on invalid input.

    Each entry must be a positive integer; duplicates are dropped while
    preserving first-seen order.
    """
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if not parts:
        raise ValueError("--candidate-strides must contain at least one stride")
    strides: list[int] = []
    seen: set[int] = set()
    for raw in parts:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"--candidate-strides entry not an integer: {raw!r}") from exc
        if value < 1:
            raise ValueError(f"--candidate-strides entry must be >= 1: {value}")
        if value in seen:
            continue
        seen.add(value)
        strides.append(value)
    return strides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    nodelist: list[str] = []
    if args.live_cluster:
        if not args.nodelist_file:
            print(
                "ERROR: --live-cluster requires --nodelist-file (see help).",
                file=sys.stderr,
            )
            return 2
        if not args.nodelist_file.is_file():
            print(
                f"ERROR: nodelist file not found: {args.nodelist_file}",
                file=sys.stderr,
            )
            return 2
        nodelist = [
            line.strip()
            for line in args.nodelist_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # Auto-stride sweep: invoke run_detector once per candidate stride
    # and aggregate. The single-stride path keeps the original output
    # shape; the sweep path wraps with mode=auto_stride.
    if args.auto_stride:
        try:
            candidates = _parse_candidate_strides(args.candidate_strides)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        per_stride: list[dict] = []
        aggregated_alerts: list[dict] = []
        for stride in candidates:
            single = run_detector(
                fixture_path=args.fixture,
                live_cluster=bool(args.live_cluster),
                nodelist=nodelist or None,
                port=args.port,
                stride=stride,
                lag_threshold=args.lag_threshold,
                output_path=args.output,
                jobid=args.jobid,
            )
            per_stride.append(single)
            aggregated_alerts.extend(single["alerts"])
        # Stable sort: largest lag first, then by stride, then by bucket.
        aggregated_alerts.sort(
            key=lambda a: (-a["lag"], a["stride"], a["lagging_bucket"])
        )
        result = {
            "schema_version": 1,
            "mode": "auto_stride",
            "jobid": args.jobid,
            "lag_threshold": args.lag_threshold,
            "strides_checked": candidates,
            "rank_count": per_stride[0]["rank_count"] if per_stride else 0,
            "alerts": aggregated_alerts,
            "per_stride": per_stride,
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            if aggregated_alerts:
                buckets_by_stride = {
                    stride: sorted(
                        a["lagging_bucket"] for a in aggregated_alerts if a["stride"] == stride
                    )
                    for stride in candidates
                }
                stride_summary = ", ".join(
                    f"stride={s}->{buckets_by_stride[s] or '[]'}" for s in candidates
                )
                print(
                    f"HANG-DETECTED: {len(aggregated_alerts)} alert(s) across "
                    f"strides {candidates}. {stride_summary}."
                )
            else:
                print(
                    f"OK: no stride-pattern lag detected across strides "
                    f"{candidates} on {result['rank_count']} rank(s)."
                )
        return 1 if aggregated_alerts else 0

    # Single-stride path (original behavior; stride defaults to 32 when
    # neither --stride nor --auto-stride is set).
    stride = args.stride if args.stride is not None else 32
    result = run_detector(
        fixture_path=args.fixture,
        live_cluster=bool(args.live_cluster),
        nodelist=nodelist or None,
        port=args.port,
        stride=stride,
        lag_threshold=args.lag_threshold,
        output_path=args.output,
        jobid=args.jobid,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        alerts = result["alerts"]
        if alerts:
            print(
                f"HANG-DETECTED: {len(alerts)} bucket(s) lagging at stride={stride}. "
                f"Affected buckets: {sorted(a['lagging_bucket'] for a in alerts)}."
            )
        else:
            print(
                f"OK: no stride-pattern lag detected at stride={stride} "
                f"across {result['rank_count']} rank(s)."
            )
    # Non-zero exit when alerts present so callers can chain.
    return 1 if result["alerts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
