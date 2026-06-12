"""Operator-facing profiling CLI surface.

Wires the two operator-grade profiling scripts (`host_overhead`,
`profile_diff`) into a single multi-verb argparse CLI so the MCP surface
(`mcp_surface.py`) can register one MCP tool per verb. Each subverb's
flags come from the underlying script's `populate_parser(parser)`
helper; each subverb's run path comes from the script's `run(args)`
helper.

The standalone scripts continue to be invokable directly
(`python3 -m tools.pipeline.submission.profile.<script>`); this CLI is
the umbrella entrypoint reachable as `python3 -m profile <verb>` via
the shim package at `mlperf-6.0-training/profile/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pipeline.submission.profile import (  # noqa: E402
    host_overhead,
    profile_diff,
)


CONTRACT: dict[str, dict[str, Any]] = {
    "host-overhead": {
        "safety": "writes_artifacts",
        "required": ("subverb",),
        # All flags live on the subverb parsers (top, record, dump).
        "optional": (),
        "json": True,
        "ack": None,
    },
    "profile-diff": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": (
            "--baseline", "--baseline-csv-dir", "--candidate",
            "--candidate-csv-dir", "--baseline-label", "--candidate-label",
            "--out", "--json-out", "--limit", "--scratch", "--json",
        ),
        "json": True,
        "ack": None,
    },
}


_VERB_DESCRIPTIONS: dict[str, str] = {
    "host-overhead": (
        "py-spy CPU sampler for the rank-0 process. Subverb: top "
        "(live display) | record (flamegraph + summary for --duration "
        "seconds) | dump (one-shot stack snapshot)."
    ),
    "profile-diff": (
        "Diff two nsys-rep files (or pre-extracted nsys-stats CSV dirs) "
        "and emit per-area NVTX / kernel / CUDA-API / NCCL delta tables."
    ),
}


_DISPATCH = {
    "host-overhead": host_overhead.run,
    "profile-diff": profile_diff.run,
}


_POPULATE = {
    "host-overhead": host_overhead.populate_parser,
    "profile-diff": profile_diff.populate_parser,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb, description in _VERB_DESCRIPTIONS.items():
        verb_parser = sub.add_parser(verb, description=description)
        _POPULATE[verb](verb_parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return _DISPATCH[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
