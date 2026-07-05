#!/usr/bin/env python3
"""Sample the host-side Python stack of a running rank-0 process.

This is the host-side complement to ``profile_run.sh`` (which captures the
GPU-side timeline through nsys + NVTX). The GPU timeline does not show
CPU / Python overhead; this tool catches Megatron-LM ``9d976bcd``-class
CPU-overhead regressions that would otherwise be invisible.

The implementation is a thin wrapper around ``py-spy``:

* ``py-spy top`` for live monitoring (no artifact).
* ``py-spy record`` for a flamegraph + a text top-N summary.
* ``py-spy dump`` for a one-shot stack snapshot.

Per ``mlperf-6.0-training/CLAUDE.md`` "Self-Contained Repository
Boundary": no required runtime deps. ``py-spy`` is a system-installed
tool; if it is not on PATH the script aborts with the install hint
rather than silently falling back to ``cProfile``.

Per ``CLAUDE.md`` "Fail Fast, No Silent Fallbacks":

* Missing ``py-spy`` is a fatal error with a remediation hint.
* Missing or unreachable PID is a fatal error.
* Output paths are created on demand; missing parent dirs do NOT abort,
  but a non-writable target does.

Per ``CLAUDE.md`` "Artifact Anchor": when ``--art-dir`` is supplied the
flamegraph + the top-N summary land under
``<art-dir>/host-overhead-flame.svg`` and ``<art-dir>/host-overhead.txt``
respectively. Otherwise the operator picks the output paths.

Usage:

    # Live top:
    python3 tools/pipeline/submission/profile/host_overhead.py \\
        top --pid 12345

    # Record 30s of stacks to a flamegraph + a sorted top-N summary:
    python3 tools/pipeline/submission/profile/host_overhead.py \\
        record --pid 12345 \\
        --art-dir experiments/artifacts/campaign/llama31_8b/<run-id>/profiling \\
        --duration 30

    # One-shot dump (cheapest):
    python3 tools/pipeline/submission/profile/host_overhead.py \\
        dump --pid 12345 --out host-overhead-dump.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DURATION_SEC = 30
DEFAULT_RATE_HZ = 100
DEFAULT_TOP_N = 25

PYSPY_INSTALL_HINT = (
    "py-spy not found on PATH. Install with `pip install py-spy` (or, "
    "inside the v6.0 image, `pip install --upgrade py-spy` against the "
    "container's pip; py-spy is not currently bundled in NVIDIA's "
    "MLPerf images). Per CLAUDE.md fail-fast, host_overhead.py refuses "
    "to silently fall back to cProfile."
)


def require_pyspy() -> str:
    path = shutil.which("py-spy")
    if path is None:
        print(f"host_overhead: {PYSPY_INSTALL_HINT}", file=sys.stderr)
        sys.exit(2)
    return path


def cmd_top(args: argparse.Namespace) -> int:
    """Live CPU-time-by-function display. Streams to stderr until ^C."""
    pyspy = require_pyspy()
    cmd = [pyspy, "top", "--pid", str(args.pid)]
    if args.rate is not None:
        cmd += ["--rate", str(args.rate)]
    if args.subprocesses:
        cmd += ["--subprocesses"]
    if args.gil:
        cmd += ["--gil"]
    if args.idle:
        cmd += ["--idle"]
    return subprocess.call(cmd)


def _record_flamegraph(
    *, pyspy: str, pid: int, duration: int, rate: int, subprocesses: bool, gil: bool, idle: bool, out: Path,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        pyspy,
        "record",
        "--pid",
        str(pid),
        "--rate",
        str(rate),
        "--duration",
        str(duration),
        "--output",
        str(out),
        "--format",
        "flamegraph",
    ]
    if subprocesses:
        cmd += ["--subprocesses"]
    if gil:
        cmd += ["--gil"]
    if idle:
        cmd += ["--idle"]
    rc = subprocess.call(cmd)
    return rc


def _record_top_summary(
    *, pyspy: str, pid: int, duration: int, rate: int, subprocesses: bool, gil: bool, idle: bool, out: Path, top_n: int,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    speedscope = out.with_suffix(".speedscope.json")
    cmd = [
        pyspy,
        "record",
        "--pid",
        str(pid),
        "--rate",
        str(rate),
        "--duration",
        str(duration),
        "--output",
        str(speedscope),
        "--format",
        "speedscope",
    ]
    if subprocesses:
        cmd += ["--subprocesses"]
    if gil:
        cmd += ["--gil"]
    if idle:
        cmd += ["--idle"]
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    summary = _summarise_speedscope(speedscope, top_n=top_n)
    out.write_text(summary, encoding="utf-8")
    return 0


def _summarise_speedscope(path: Path, *, top_n: int) -> str:
    """Speedscope's JSON gives us per-frame self-time totals; aggregate them."""
    import json

    if not path.is_file():
        return "(speedscope output missing)\n"
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return f"(speedscope decode failed: {exc})\n"
    shared = payload.get("shared", {})
    frames = shared.get("frames", []) or []
    profiles = payload.get("profiles", []) or []
    self_time: dict[int, int] = {}
    total_units = 0
    for prof in profiles:
        samples = prof.get("samples", []) or []
        weights = prof.get("weights", []) or []
        if not samples:
            continue
        if len(weights) != len(samples):
            continue
        for stack, weight in zip(samples, weights):
            if not stack:
                continue
            total_units += weight
            top = stack[-1]
            if isinstance(top, int):
                self_time[top] = self_time.get(top, 0) + weight
    rows: list[tuple[str, int]] = []
    for frame_idx, weight in self_time.items():
        if 0 <= frame_idx < len(frames):
            frame = frames[frame_idx]
            name = frame.get("name") or "(anon)"
            file_ = frame.get("file") or ""
            line = frame.get("line")
            label = name
            if file_:
                if line is not None:
                    label = f"{name}  ({file_}:{line})"
                else:
                    label = f"{name}  ({file_})"
            rows.append((label, weight))
    rows.sort(key=lambda r: r[1], reverse=True)
    out_lines: list[str] = []
    out_lines.append("# host_overhead.py top-N self-time")
    out_lines.append(f"# total samples: {total_units}")
    out_lines.append(f"# top {top_n} frames by self-time (descending)")
    out_lines.append("")
    out_lines.append(f"{'Self %':>7s}  {'Self':>10s}  Frame")
    out_lines.append("-" * 70)
    for label, weight in rows[:top_n]:
        pct = (100.0 * weight / total_units) if total_units > 0 else 0.0
        out_lines.append(f"{pct:>6.2f}%  {weight:>10d}  {label}")
    return "\n".join(out_lines) + "\n"


def cmd_record(args: argparse.Namespace) -> int:
    pyspy = require_pyspy()
    if args.art_dir is not None:
        flame_out = args.flame_out or args.art_dir / "host-overhead-flame.svg"
        text_out = args.text_out or args.art_dir / "host-overhead.txt"
    else:
        if args.flame_out is None or args.text_out is None:
            print(
                "host_overhead: provide --art-dir OR both --flame-out and --text-out",
                file=sys.stderr,
            )
            return 2
        flame_out = args.flame_out
        text_out = args.text_out

    flame_rc = _record_flamegraph(
        pyspy=pyspy,
        pid=args.pid,
        duration=args.duration,
        rate=args.rate,
        subprocesses=args.subprocesses,
        gil=args.gil,
        idle=args.idle,
        out=flame_out,
    )
    if flame_rc != 0:
        print(
            f"host_overhead: py-spy record (flamegraph) exited {flame_rc}",
            file=sys.stderr,
        )
        return flame_rc

    text_rc = _record_top_summary(
        pyspy=pyspy,
        pid=args.pid,
        duration=args.duration,
        rate=args.rate,
        subprocesses=args.subprocesses,
        gil=args.gil,
        idle=args.idle,
        out=text_out,
        top_n=args.top_n,
    )
    if text_rc != 0:
        print(
            f"host_overhead: py-spy record (speedscope) exited {text_rc}",
            file=sys.stderr,
        )
        return text_rc

    print(f"host_overhead: wrote flamegraph to {flame_out}", file=sys.stderr)
    print(f"host_overhead: wrote top-N summary to {text_out}", file=sys.stderr)
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    pyspy = require_pyspy()
    cmd = [pyspy, "dump", "--pid", str(args.pid)]
    if args.subprocesses:
        cmd += ["--subprocesses"]
    if args.locals:
        cmd += ["--locals"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        print(f"host_overhead: py-spy dump failed: {exc}", file=sys.stderr)
        return 2
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(proc.stdout, encoding="utf-8")
        print(f"host_overhead: wrote dump to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def populate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach this script's subverbs + flags to a caller-supplied parser."""
    sub = parser.add_subparsers(dest="command", required=True)

    common_pid = argparse.ArgumentParser(add_help=False)
    common_pid.add_argument("--pid", type=int, required=True,
                            help="PID of the rank-0 (or any) process to sample.")
    common_pid.add_argument("--rate", type=int, default=DEFAULT_RATE_HZ,
                            help=f"Sampling rate in Hz. Default {DEFAULT_RATE_HZ}.")
    common_pid.add_argument("--subprocesses", action="store_true",
                            help="Profile all subprocesses (e.g. dataloader workers).")
    common_pid.add_argument("--gil", action="store_true",
                            help="Only sample threads holding the Python GIL.")
    common_pid.add_argument("--idle", action="store_true",
                            help="Include idle/blocked threads in the sample stream.")

    top = sub.add_parser("top", parents=[common_pid], help="Live CPU-time-by-function display (Ctrl-C to exit).")
    top.add_argument("--json", action="store_true",
                     help="No-op; py-spy top is interactive and prints to the terminal.")
    top.set_defaults(func=cmd_top)

    record = sub.add_parser("record", parents=[common_pid],
                            help="Record stacks for --duration seconds; emit flamegraph + top-N summary.")
    record.add_argument("--duration", type=int, default=DEFAULT_DURATION_SEC,
                        help=f"Sampling duration in seconds. Default {DEFAULT_DURATION_SEC}.")
    record.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of top frames in the text summary. Default {DEFAULT_TOP_N}.")
    record.add_argument("--art-dir", type=Path, default=None,
                        help="Artifact dir; flame -> host-overhead-flame.svg; "
                             "text -> host-overhead.txt.")
    record.add_argument("--flame-out", type=Path, default=None,
                        help="Override the flamegraph output path.")
    record.add_argument("--text-out", type=Path, default=None,
                        help="Override the top-N text summary output path.")
    record.add_argument("--json", action="store_true",
                        help="No-op; record emits the flamegraph SVG + top-N text summary.")
    record.set_defaults(func=cmd_record)

    dump = sub.add_parser("dump", parents=[common_pid],
                          help="One-shot stack snapshot (cheapest).")
    dump.add_argument("--out", type=Path, default=None,
                      help="Write the dump here (default: stdout).")
    dump.add_argument("--locals", action="store_true",
                      help="Show local variables in each frame.")
    dump.add_argument("--json", action="store_true",
                      help="No-op; py-spy dump emits a stack snapshot to --out / stdout.")
    dump.set_defaults(func=cmd_dump)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    populate_parser(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    return int(args.func(args))


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
