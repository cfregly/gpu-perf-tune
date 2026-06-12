"""Bundle importer for **AIPerf** ``profile_export_aiperf.csv`` exports.

Reads an AIPerf variant directory laid out as::

    <variant_dir>/
      c1/profile_export_aiperf.csv
      c8/profile_export_aiperf.csv
      c16/profile_export_aiperf.csv
      c32/profile_export_aiperf.csv

(produced by ``aiperf profile`` runs, e.g. the
``perf-tune-glm51/cluster-probes/<ts>-aiperf-allvariants/raw/<variant>/`` campaign)
and writes a perf-report-compatible ``cells/<cell-id>/normalized.json``.

This closes the gap noted in the 2026-05-31 AIPerf campaign: the perf_tune_report
pipeline previously only imported vLLM ``bench serve`` text + ``drive_load.py``
JSONL, so raw AIPerf exports had no path into the perf-lake. This importer is
the missing arm.

AIPerf CSV format (two CSV sections in one file):

1. Per-metric percentile table, header
   ``Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std`` -- rows like
   ``Time to First Token (ms)``, ``Inter Token Latency (ms)``,
   ``Output Token Throughput Per User (tokens/sec/user)``.
2. Scalar summary, header ``Metric,Value`` -- rows like ``Request Count``,
   ``Request Throughput (requests/sec)``, ``Output Token Throughput (tokens/sec)``.

AtlasCell mapping:

- ``ttft_avg_ms``            = ``Time to First Token (ms)`` avg
- ``request_throughput_avg`` = ``Request Throughput (requests/sec)``
- ``output_tps_per_user``    = ``Output Token Throughput Per User (tokens/sec/user)`` avg
- ``output_tps_per_gpu``     = ``Output Token Throughput (tokens/sec)`` / ``tensor_parallel``

Status: a cell is ``full`` only when its ``Request Count`` reaches the expected
Replay turn count for that concurrency (the 2025_07 split: c=1->70, c=8->284,
c=16->559, c=32->1135 turns). Cells that complete materially fewer requests
(sticky-session conversations aborting on a failed/slow turn) are marked
``partial`` so the renderer + coverage block flag them rather than presenting an
unreliable throughput number as clean. Override the expected counts via
``overrides['expected_reqs']`` (dict ``{concurrency: count}``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import (
    BACKEND_AIPERF,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)

# Per-concurrency expected request (turn) counts for the Replay 2025_07 split,
# used to decide full-vs-partial. A cell completing < PARTIAL_FRACTION of the
# expected count is flagged partial. Overridable via overrides['expected_reqs'].
DEFAULT_EXPECTED_REQS: dict[int, int] = {1: 70, 8: 284, 16: 559, 32: 1135}
PARTIAL_FRACTION = 0.9

_CELL_DIR = __import__("re").compile(r"^c(\d+)$")


@dataclass(frozen=True)
class _AiperfCellFile:
    path: Path
    concurrency: int


def _enumerate_aiperf_cells(variant_dir: Path) -> list[_AiperfCellFile]:
    """Find ``c<N>/profile_export_aiperf.csv`` under a variant dir, sorted by c."""
    out: list[_AiperfCellFile] = []
    if not variant_dir.is_dir():
        return out
    for sub in variant_dir.iterdir():
        if not sub.is_dir():
            continue
        m = _CELL_DIR.match(sub.name)
        if not m:
            continue
        csv_path = sub / "profile_export_aiperf.csv"
        if csv_path.is_file():
            out.append(_AiperfCellFile(path=csv_path, concurrency=int(m.group(1))))
    out.sort(key=lambda x: x.concurrency)
    return out


def detect_aiperf_bundle(variant_dir: Path) -> bool:
    """True iff ``variant_dir`` has at least one ``c<N>/profile_export_aiperf.csv``."""
    return bool(_enumerate_aiperf_cells(variant_dir))


def _parse_aiperf_csv(path: Path) -> dict[str, float] | None:
    """Parse the two-section AIPerf CSV into a flat metric dict.

    Section-1 rows contribute ``<metric> .avg``; section-2 rows contribute the
    scalar ``Value``. Returns None if the file has neither section.
    """
    per_metric_avg: dict[str, float] = {}
    scalar: dict[str, float] = {}
    section = None  # 'percentile' | 'scalar'
    try:
        rows = list(csv.reader(path.open(newline="")))
    except OSError:
        return None
    for row in rows:
        if not row or all(not c.strip() for c in row):
            continue
        head = row[0].strip()
        if head == "Metric" and len(row) >= 3 and row[1].strip() == "avg":
            section = "percentile"
            continue
        if head == "Metric" and len(row) == 2 and row[1].strip() == "Value":
            section = "scalar"
            continue
        if section == "percentile" and len(row) >= 2:
            try:
                per_metric_avg[head] = float(row[1])
            except ValueError:
                pass
        elif section == "scalar" and len(row) >= 2:
            try:
                scalar[head] = float(row[1])
            except ValueError:
                pass
    if not per_metric_avg and not scalar:
        return None
    out: dict[str, float] = {}
    if "Time to First Token (ms)" in per_metric_avg:
        out["ttft_avg_ms"] = per_metric_avg["Time to First Token (ms)"]
    if "Output Token Throughput Per User (tokens/sec/user)" in per_metric_avg:
        out["output_tps_per_user"] = per_metric_avg[
            "Output Token Throughput Per User (tokens/sec/user)"
        ]
    if "Inter Token Latency (ms)" in per_metric_avg:
        out["itl_avg_ms"] = per_metric_avg["Inter Token Latency (ms)"]
    if "Request Throughput (requests/sec)" in scalar:
        out["request_throughput_avg"] = scalar["Request Throughput (requests/sec)"]
    if "Output Token Throughput (tokens/sec)" in scalar:
        out["output_tps_total"] = scalar["Output Token Throughput (tokens/sec)"]
    # Total (input+output) token throughput, when this AIPerf version emits it.
    for key in ("Total Token Throughput (tokens/sec)", "Token Throughput (tokens/sec)"):
        if key in scalar:
            out["total_tps_total"] = scalar[key]
            break
    if "Request Count" in scalar:
        out["request_count"] = scalar["Request Count"]
    if "Overall Usage Prompt Cache Read % (%)" in scalar:
        out["prompt_cache_read_pct"] = scalar["Overall Usage Prompt Cache Read % (%)"]
    # Mean ISL/OSL (shape): AIPerf emits these as percentile-table metrics in
    # newer versions; absent in older exports -> None downstream.
    for key in ("Input Sequence Length (tokens)", "Input Sequence Length"):
        if key in per_metric_avg:
            out["mean_input_tokens"] = per_metric_avg[key]
            break
    for key in ("Output Sequence Length (tokens)", "Output Sequence Length"):
        if key in per_metric_avg:
            out["mean_output_tokens"] = per_metric_avg[key]
            break
    return out


@dataclass(frozen=True)
class AiperfImportResult:
    campaign_dir: Path
    cell_id: str
    cell_dir: Path
    normalized_path: Path
    bundle_path: Path
    row_count: int
    concurrencies: list[int]
    status: str
    partial_cells: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_dir": str(self.campaign_dir),
            "cell_id": self.cell_id,
            "cell_dir": str(self.cell_dir),
            "normalized_path": str(self.normalized_path),
            "bundle_path": str(self.bundle_path),
            "row_count": self.row_count,
            "concurrencies": self.concurrencies,
            "status": self.status,
            "partial_cells": self.partial_cells,
            "importer": "aiperf_export",
        }


def import_aiperf_bundle(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
) -> AiperfImportResult:
    """Convert one AIPerf variant dir into ``cells/<cell-id>/normalized.json``.

    Args:
        bundle: a variant directory containing ``c<N>/profile_export_aiperf.csv``.
        campaign_dir: target campaign dir (created by ``campaign_init``).
        overrides: identity overrides (``cell_id``, ``model`` [required],
            ``hardware``, ``quant``, ``tensor_parallel``, ``parallel_strategy``,
            ``mtp``, ``max_num_batched_tokens``, ``notes``, ``expected_reqs``).
        dry_run: parse + validate, write nothing.
        captured_at: ISO-8601 stamp (default now).

    Raises:
        ValueError: bundle missing, no aiperf cells, or no ``model`` override.
    """
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"import_aiperf: bundle does not exist: {bundle}")
    cells = _enumerate_aiperf_cells(bundle)
    if not cells:
        raise ValueError(
            f"import_aiperf: no c<N>/profile_export_aiperf.csv under {bundle}"
        )
    ov = overrides or {}
    model = ov.get("model")
    if not model:
        raise ValueError("import_aiperf: --model is required (no inference of model)")
    cell_id = ov.get("cell_id") or bundle.name
    hardware = ov.get("hardware", "B200")
    quant = ov.get("quant", "NVFP4")
    tensor_parallel = int(ov.get("tensor_parallel", 8))
    parallel_strategy = ov.get("parallel_strategy", "TP")
    mtp = bool(ov.get("mtp", False))
    mbt = int(ov.get("max_num_batched_tokens", 12288))
    cache_mode = ov.get("cache_mode", "unknown")
    if cache_mode not in ("warm", "cold", "unknown"):
        cache_mode = "unknown"
    expected = dict(DEFAULT_EXPECTED_REQS)
    if isinstance(ov.get("expected_reqs"), dict):
        expected.update({int(k): int(v) for k, v in ov["expected_reqs"].items()})
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[AtlasCell] = []
    partial_cells: list[int] = []
    for cf in cells:
        m = _parse_aiperf_csv(cf.path)
        if m is None:
            continue
        rc = m.get("request_count")
        exp = expected.get(cf.concurrency)
        cell_partial = (
            rc is not None and exp is not None and rc < PARTIAL_FRACTION * exp
        )
        status = STATUS_PARTIAL if cell_partial else STATUS_FULL
        if cell_partial:
            partial_cells.append(cf.concurrency)
        otps = m.get("output_tps_total")
        ttps = m.get("total_tps_total")
        # AIPerf reports prompt-cache read as a 0..100 percentage; normalize to 0..1.
        pc_pct = m.get("prompt_cache_read_pct")
        prefix_cache_hit_rate = (pc_pct / 100.0) if pc_pct is not None else None
        notes = [f"imported from aiperf {bundle.name}"]
        if rc is not None and exp is not None:
            notes.append(f"reqs {int(rc)}/{exp}")
        if cell_partial:
            notes.append("PARTIAL: low completion (sticky-session aborts)")
        rows.append(
            AtlasCell(
                cell_id=cell_id,
                model=model,
                hardware=hardware,
                quant=quant,
                tensor_parallel=tensor_parallel,
                parallel_strategy=parallel_strategy,
                mtp=mtp,
                max_num_batched_tokens=mbt,
                concurrency=cf.concurrency,
                status=status,
                ttft_avg_ms=m.get("ttft_avg_ms"),
                request_throughput_avg=m.get("request_throughput_avg"),
                output_tps_per_user=m.get("output_tps_per_user"),
                output_tps_per_gpu=(otps / tensor_parallel) if otps else None,
                total_tps_per_gpu=(ttps / tensor_parallel) if ttps else None,
                itl_avg_ms=m.get("itl_avg_ms"),
                mean_input_tokens=m.get("mean_input_tokens"),
                mean_output_tokens=m.get("mean_output_tokens"),
                prefix_cache_hit_rate=prefix_cache_hit_rate,
                cache_mode=cache_mode,
                backend=BACKEND_AIPERF,
                raw_path=str(cf.path),
                captured_at=captured_at,
                notes=" | ".join(notes),
                extra={
                    "itl_avg_ms": m.get("itl_avg_ms"),
                    "request_count": m.get("request_count"),
                    "expected_request_count": exp,
                    "prompt_cache_read_pct": m.get("prompt_cache_read_pct"),
                    "imported_from_aiperf": str(bundle),
                },
            )
        )
    if not rows:
        raise ValueError(f"import_aiperf: no parseable AIPerf CSVs in {bundle}")

    concurrencies = sorted({r.concurrency for r in rows})
    overall_status = STATUS_PARTIAL if partial_cells else STATUS_FULL
    cell_dir = campaign_dir / "cells" / cell_id
    normalized_path = cell_dir / "normalized.json"
    if dry_run:
        return AiperfImportResult(
            campaign_dir=campaign_dir, cell_id=cell_id, cell_dir=cell_dir,
            normalized_path=normalized_path, bundle_path=bundle,
            row_count=len(rows), concurrencies=concurrencies,
            status=overall_status, partial_cells=sorted(partial_cells),
        )
    cell_dir.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(
        json.dumps([r.to_dict() for r in rows], indent=2, sort_keys=True)
    )
    (cell_dir / "status.txt").write_text(overall_status + "\n")
    (cell_dir / "backend.txt").write_text(BACKEND_AIPERF + "\n")
    (cell_dir / "SOURCE.md").write_text(
        f"# {cell_id}\n\n"
        f"- imported_from: {bundle} (AIPerf)\n"
        f"- captured_at:   {captured_at}\n"
        f"- concurrencies: {concurrencies}\n"
        f"- row_count:     {len(rows)}\n"
        f"- status:        {overall_status}\n"
        f"- partial_cells (low completion): {sorted(partial_cells)}\n"
    )
    return AiperfImportResult(
        campaign_dir=campaign_dir, cell_id=cell_id, cell_dir=cell_dir,
        normalized_path=normalized_path, bundle_path=bundle,
        row_count=len(rows), concurrencies=concurrencies,
        status=overall_status, partial_cells=sorted(partial_cells),
    )
