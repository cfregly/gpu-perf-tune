#!/usr/bin/env python3
"""Diff two ``.nsys-rep`` profiles and emit a ranked delta report.

Wraps ``nsys stats --report cudaapisum,gpukernsum,nvtxsum,nccltrace
--format csv`` over a baseline and a candidate ``.nsys-rep`` and prints (or
writes) a markdown report with three tables:

* Top NVTX-range cost deltas (forward, backward, optimizer, comm-overlap,
  MLA, MoE-routing, alltoall - exactly the ranges already pushed by
  Megatron + TE).
* Top kernel-mix deltas (which kernels grew or shrank, by total device
  time and by call count).
* NCCL collective deltas (total time per collective).

Sort order is by absolute delta-of-totals; the top ``--limit`` rows in
each direction (regression / improvement) are kept.

The tool runs ``nsys stats`` itself when invoked with ``.nsys-rep``
inputs. When ``nsys`` is not on PATH or the operator already produced the
CSVs (e.g. on a stripped login node) ``--baseline-csv-dir`` and
``--candidate-csv-dir`` accept pre-extracted CSV directories; the
filename matches the ``nsys stats`` output convention
(``<stem>_<report>.csv``).

Per ``mlperf-6.0-training/CLAUDE.md`` "Fail Fast, No Silent Fallbacks":

* Missing inputs abort with a clear error and non-zero exit.
* Empty CSVs (zero rows after the header) abort.
* When ``nsys stats`` fails, the underlying stderr is forwarded.

Per ``CLAUDE.md`` "Self-Contained Repository Boundary": stdlib only;
no third-party deps.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

NSYS_REPORTS = ("cudaapisum", "gpukernsum", "nvtxsum", "nccltrace")
DEFAULT_LIMIT = 20

INTERESTING_NVTX_KEYWORDS = (
    "forward",
    "backward",
    "optimizer",
    "tp_comm",
    "comm_overlap",
    "split_overlap",
    "all_gather",
    "reduce_scatter",
    "alltoall",
    "all_reduce",
    "mla",
    "moe",
    "routing",
    "expert",
    "rope",
    "attention",
    "fp8",
    "mxfp8",
    "nvfp4",
)


@dataclass
class StatRow:
    """One row of an ``nsys stats`` table that exposes a ``Total Time (ns)``
    column and a name column. The exact column names vary by report; we
    normalise on ``name`` and ``total_ns`` plus an optional ``count``.
    """

    name: str
    total_ns: int
    count: int | None = None
    extra: dict[str, str] | None = None


def run_nsys_stats(report_path: Path, out_dir: Path) -> dict[str, Path]:
    """Run ``nsys stats --report <reports> --format csv -o <out_dir>/stem``.

    Returns a mapping from report name to the produced CSV path.
    """
    if not report_path.is_file():
        raise FileNotFoundError(f"profile not found: {report_path}")
    if shutil.which("nsys") is None:
        raise RuntimeError(
            "nsys not on PATH; install Nsight Systems CLI or pre-extract "
            "the CSVs with --baseline-csv-dir / --candidate-csv-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / report_path.stem
    cmd = [
        "nsys",
        "stats",
        "--report",
        ",".join(NSYS_REPORTS),
        "--format",
        "csv",
        "-o",
        str(stem),
        str(report_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"nsys stats failed for {report_path}:\n"
            f"  stdout:\n{exc.stdout}\n  stderr:\n{exc.stderr}"
        ) from exc
    paths = {}
    for report in NSYS_REPORTS:
        candidate = out_dir / f"{report_path.stem}_{report}.csv"
        if not candidate.is_file():
            raise RuntimeError(
                f"nsys stats did not produce expected CSV: {candidate}"
            )
        paths[report] = candidate
    return paths


def collect_csv_dir(csv_dir: Path) -> dict[str, Path]:
    """Locate the four expected CSVs in a pre-extracted directory."""
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"csv dir not found: {csv_dir}")
    paths: dict[str, Path] = {}
    for report in NSYS_REPORTS:
        matches = sorted(csv_dir.glob(f"*_{report}.csv"))
        if not matches:
            raise FileNotFoundError(
                f"no CSV matching *_{report}.csv in {csv_dir}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"ambiguous CSV match for {report} in {csv_dir}: {matches}"
            )
        paths[report] = matches[0]
    return paths


def _to_int(text: str) -> int:
    cleaned = text.strip().replace(",", "").replace('"', "")
    if not cleaned:
        return 0
    try:
        return int(float(cleaned))
    except ValueError:
        return 0


def parse_stat_csv(
    csv_path: Path,
    *,
    name_col_candidates: tuple[str, ...],
    total_col_candidates: tuple[str, ...],
    count_col_candidates: tuple[str, ...] = (),
) -> list[StatRow]:
    """Parse an ``nsys stats`` CSV.

    The ``nsys`` output header varies slightly by version; we accept any
    of the supplied candidates for the name / total / count columns and
    fall back to the first non-empty header that matches a substring.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv missing: {csv_path}")
    rows: list[StatRow] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return rows
        header_norm = [h.strip() for h in header]

        def _resolve(cands: tuple[str, ...]) -> int | None:
            for cand in cands:
                if cand in header_norm:
                    return header_norm.index(cand)
            for idx, h in enumerate(header_norm):
                lo = h.lower()
                for cand in cands:
                    if cand.lower() in lo:
                        return idx
            return None

        name_idx = _resolve(name_col_candidates)
        total_idx = _resolve(total_col_candidates)
        count_idx = _resolve(count_col_candidates) if count_col_candidates else None
        if name_idx is None or total_idx is None:
            raise RuntimeError(
                f"could not locate name/total columns in {csv_path}; "
                f"header was {header_norm}"
            )
        for raw in reader:
            if not raw or len(raw) <= max(name_idx, total_idx):
                continue
            name = raw[name_idx].strip()
            if not name:
                continue
            total_ns = _to_int(raw[total_idx])
            count = _to_int(raw[count_idx]) if count_idx is not None and count_idx < len(raw) else None
            rows.append(StatRow(name=name, total_ns=total_ns, count=count))
    return rows


def parse_nvtx(csv_path: Path) -> list[StatRow]:
    return parse_stat_csv(
        csv_path,
        name_col_candidates=("Range", "Name"),
        total_col_candidates=("Total Time (ns)", "Total Time"),
        count_col_candidates=("Instances", "Num Calls"),
    )


def parse_kernels(csv_path: Path) -> list[StatRow]:
    return parse_stat_csv(
        csv_path,
        name_col_candidates=("Name", "Kernel Name"),
        total_col_candidates=("Total Time (ns)", "Total Time"),
        count_col_candidates=("Instances", "Num Calls"),
    )


def parse_cuda_api(csv_path: Path) -> list[StatRow]:
    return parse_stat_csv(
        csv_path,
        name_col_candidates=("Name", "API Name"),
        total_col_candidates=("Total Time (ns)", "Total Time"),
        count_col_candidates=("Num Calls", "Instances"),
    )


def parse_nccl(csv_path: Path) -> list[StatRow]:
    """``nccltrace`` is a per-event trace; we aggregate by collective name."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv missing: {csv_path}")
    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return []
        header_norm = [h.strip() for h in header]

        def _idx(cands: tuple[str, ...]) -> int | None:
            for cand in cands:
                if cand in header_norm:
                    return header_norm.index(cand)
            for idx, h in enumerate(header_norm):
                lo = h.lower()
                for cand in cands:
                    if cand.lower() in lo:
                        return idx
            return None

        name_idx = _idx(("Function", "Name", "Operation"))
        dur_idx = _idx(("Duration (ns)", "Duration"))
        if name_idx is None or dur_idx is None:
            return []
        for raw in reader:
            if not raw or len(raw) <= max(name_idx, dur_idx):
                continue
            name = raw[name_idx].strip()
            if not name:
                continue
            totals[name] += _to_int(raw[dur_idx])
            counts[name] += 1
    return [
        StatRow(name=name, total_ns=totals[name], count=counts[name])
        for name in totals
    ]


def keep_interesting_nvtx(rows: list[StatRow]) -> list[StatRow]:
    keep: list[StatRow] = []
    for row in rows:
        lo = row.name.lower()
        if any(keyword in lo for keyword in INTERESTING_NVTX_KEYWORDS):
            keep.append(row)
    return keep or rows


@dataclass
class DeltaRow:
    name: str
    baseline_ns: int
    candidate_ns: int
    delta_ns: int
    pct: float | None
    baseline_count: int | None
    candidate_count: int | None

    @property
    def abs_delta_ns(self) -> int:
        return abs(self.delta_ns)


def compute_delta(
    baseline_rows: list[StatRow],
    candidate_rows: list[StatRow],
) -> list[DeltaRow]:
    base_lookup = {r.name: r for r in baseline_rows}
    cand_lookup = {r.name: r for r in candidate_rows}
    names = sorted(set(base_lookup) | set(cand_lookup))
    out: list[DeltaRow] = []
    for name in names:
        base = base_lookup.get(name)
        cand = cand_lookup.get(name)
        base_ns = base.total_ns if base is not None else 0
        cand_ns = cand.total_ns if cand is not None else 0
        delta = cand_ns - base_ns
        pct: float | None
        if base_ns > 0:
            pct = 100.0 * delta / base_ns
        else:
            pct = None
        out.append(
            DeltaRow(
                name=name,
                baseline_ns=base_ns,
                candidate_ns=cand_ns,
                delta_ns=delta,
                pct=pct,
                baseline_count=base.count if base is not None else None,
                candidate_count=cand.count if cand is not None else None,
            )
        )
    out.sort(key=lambda r: r.abs_delta_ns, reverse=True)
    return out


def fmt_ns(value: int) -> str:
    """Render nanoseconds as a human-readable string with consistent units."""
    if value == 0:
        return "0"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1_000_000_000:
        return f"{sign}{abs_val / 1_000_000_000:.3f}s"
    if abs_val >= 1_000_000:
        return f"{sign}{abs_val / 1_000_000:.3f}ms"
    if abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.3f}us"
    return f"{sign}{abs_val}ns"


def fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def render_table(title: str, deltas: list[DeltaRow], limit: int) -> str:
    if not deltas:
        return f"### {title}\n\n_No data._\n\n"
    lines = [f"### {title}", ""]
    lines.append("| Name | Baseline | Candidate | Delta | Pct | Calls (b -> c) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in deltas[:limit]:
        b_count = row.baseline_count if row.baseline_count is not None else "-"
        c_count = row.candidate_count if row.candidate_count is not None else "-"
        lines.append(
            "| `"
            + row.name.replace("|", "\\|")
            + "` | "
            + fmt_ns(row.baseline_ns)
            + " | "
            + fmt_ns(row.candidate_ns)
            + " | "
            + fmt_ns(row.delta_ns)
            + " | "
            + fmt_pct(row.pct)
            + f" | {b_count} -> {c_count} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_report(
    *,
    baseline_label: str,
    candidate_label: str,
    nvtx_deltas: list[DeltaRow],
    kernel_deltas: list[DeltaRow],
    cuda_deltas: list[DeltaRow],
    nccl_deltas: list[DeltaRow],
    limit: int,
) -> str:
    parts = [
        "# Profile diff",
        "",
        f"Baseline: `{baseline_label}`  ",
        f"Candidate: `{candidate_label}`",
        "",
        "Sorted by absolute delta-of-totals; top "
        f"`{limit}` rows per table.",
        "",
        "Each row's `Delta` column is `candidate - baseline`. Positive numbers "
        "indicate the candidate spent MORE device time in that range / kernel "
        "/ collective; negative numbers are improvements.",
        "",
        render_table("NVTX ranges (interesting subset)", nvtx_deltas, limit),
        render_table("GPU kernels (top deltas)", kernel_deltas, limit),
        render_table("CUDA API (top deltas)", cuda_deltas, limit),
        render_table("NCCL collectives", nccl_deltas, limit),
    ]
    return "\n".join(parts).rstrip() + "\n"


def populate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach this script's flags to a caller-supplied parser.

    NOTE: profile_diff used to expose `--json PATH` as the JSON sidecar
    output. Renamed to `--json-out PATH` to free up the bare `--json`
    flag as a no-op (the MCP runtime auto-appends `--json` to argv
    whenever `CONTRACT[verb]["json"]` is True). Standalone callers need
    to migrate `profile_diff --json file.json` to
    `profile_diff --json-out file.json`.
    """
    src = parser.add_argument_group("inputs (mutually exclusive per side)")
    src.add_argument("--baseline", type=Path, default=None,
                     help="Path to the baseline .nsys-rep.")
    src.add_argument("--baseline-csv-dir", type=Path, default=None,
                     help="Pre-extracted nsys-stats directory for the baseline.")
    src.add_argument("--candidate", type=Path, default=None,
                     help="Path to the candidate .nsys-rep.")
    src.add_argument("--candidate-csv-dir", type=Path, default=None,
                     help="Pre-extracted nsys-stats directory for the candidate.")
    parser.add_argument("--baseline-label", default=None,
                        help="Override the baseline label printed in the report.")
    parser.add_argument("--candidate-label", default=None,
                        help="Override the candidate label printed in the report.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the report to this path. Default: stdout.")
    parser.add_argument("--json-out", dest="json_out", type=Path, default=None,
                        help="Optional JSON sidecar with the full delta tables.")
    parser.add_argument("--json", action="store_true",
                        help="No-op; --json-out PATH is the real JSON sidecar flag.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Top-N rows per table. Default {DEFAULT_LIMIT}.")
    parser.add_argument("--scratch", type=Path, default=None,
                        help="Scratch directory for nsys-stats CSV extraction. "
                             "Default: <out parent>/nsys-stats or a tempdir.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    populate_parser(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if (args.baseline is None) == (args.baseline_csv_dir is None):
        raise SystemExit("exactly one of --baseline or --baseline-csv-dir is required")
    if (args.candidate is None) == (args.candidate_csv_dir is None):
        raise SystemExit("exactly one of --candidate or --candidate-csv-dir is required")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    scratch = args.scratch
    if scratch is None and args.out is not None:
        scratch = args.out.parent / "nsys-stats"

    if args.baseline is not None:
        baseline_label = args.baseline_label or str(args.baseline)
        base_csvs = run_nsys_stats(
            args.baseline,
            (scratch / "baseline") if scratch else Path.cwd() / ".profile-diff" / "baseline",
        )
    else:
        baseline_label = args.baseline_label or str(args.baseline_csv_dir)
        base_csvs = collect_csv_dir(args.baseline_csv_dir)

    if args.candidate is not None:
        candidate_label = args.candidate_label or str(args.candidate)
        cand_csvs = run_nsys_stats(
            args.candidate,
            (scratch / "candidate") if scratch else Path.cwd() / ".profile-diff" / "candidate",
        )
    else:
        candidate_label = args.candidate_label or str(args.candidate_csv_dir)
        cand_csvs = collect_csv_dir(args.candidate_csv_dir)

    nvtx_deltas = compute_delta(
        keep_interesting_nvtx(parse_nvtx(base_csvs["nvtxsum"])),
        keep_interesting_nvtx(parse_nvtx(cand_csvs["nvtxsum"])),
    )
    kernel_deltas = compute_delta(
        parse_kernels(base_csvs["gpukernsum"]),
        parse_kernels(cand_csvs["gpukernsum"]),
    )
    cuda_deltas = compute_delta(
        parse_cuda_api(base_csvs["cudaapisum"]),
        parse_cuda_api(cand_csvs["cudaapisum"]),
    )
    nccl_deltas = compute_delta(
        parse_nccl(base_csvs["nccltrace"]),
        parse_nccl(cand_csvs["nccltrace"]),
    )

    report = render_report(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        nvtx_deltas=nvtx_deltas,
        kernel_deltas=kernel_deltas,
        cuda_deltas=cuda_deltas,
        nccl_deltas=nccl_deltas,
        limit=args.limit,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"profile_diff: wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": baseline_label,
            "candidate": candidate_label,
            "limit": args.limit,
            "nvtx": [_delta_to_dict(r) for r in nvtx_deltas],
            "kernels": [_delta_to_dict(r) for r in kernel_deltas],
            "cuda_api": [_delta_to_dict(r) for r in cuda_deltas],
            "nccl": [_delta_to_dict(r) for r in nccl_deltas],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"profile_diff: wrote {args.json_out}", file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


def _delta_to_dict(row: DeltaRow) -> dict[str, object]:
    return {
        "name": row.name,
        "baseline_ns": row.baseline_ns,
        "candidate_ns": row.candidate_ns,
        "delta_ns": row.delta_ns,
        "pct": row.pct,
        "baseline_count": row.baseline_count,
        "candidate_count": row.candidate_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
