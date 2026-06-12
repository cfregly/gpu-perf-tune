"""import_workloads: a bench-all-workloads.sh output dir -> dataset-tagged campaign cells.

``bench-all-workloads.sh`` writes one ``<tag>-c<c>.txt`` per (workload, concurrency) plus a
``bench-workloads.json`` (tag -> dataset / ISL / OSL, mirroring
``perf-tune-report/configs/workloads.yaml``). This importer parses each ``<tag>-c<c>.txt`` (reusing
``raw_bench_compare._parse_sweep_file`` for the metric extraction) and emits one
``cells/<tag>/normalized.json`` per workload tag, with each concurrency a row tagged with the
workload's ``dataset`` + typed ISL/OSL -- closing the ``dataset=unknown`` gap at the source so
``atlas_aggregate -> publish`` lands the full multi-workload suite.

The serve-config identity (model / hardware / TP / kv-cache / image / ...) is NOT in the bench
output, so it is supplied by the caller (the same overrides ``import_perf_bench`` takes).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report.raw_bench_compare import _parse_sweep_file
from tools.perf_tune_report.runners.common import (
    write_backend_file,
    write_normalized_json,
    write_status_file,
)
from tools.perf_tune_report.schema import BACKEND_VLLM_SWEEP, STATUS_FULL, AtlasCell

# bench-all-workloads.sh output file: <tag>-c<c>.txt (tag may contain hyphens, e.g. aa-1k).
# Greedy ``.+`` anchors on the LAST ``-c<digits>`` so multi-hyphen tags resolve correctly.
_TAG_C_RE = re.compile(r"^(?P<tag>.+)-c(?P<c>\d+)\.txt$")


@dataclass
class WorkloadsImportResult:
    """Outcome of one bench-all-workloads import."""

    campaign_dir: Path
    n_cells: int
    n_rows: int
    tags: list[str]
    skipped: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_dir": str(self.campaign_dir),
            "n_cells": self.n_cells,
            "n_rows": self.n_rows,
            "tags": self.tags,
            "skipped": self.skipped,
        }


def _load_workload_shapes(bench_dir: Path) -> dict[str, dict[str, Any]]:
    """tag -> {dataset, isl, osl} from bench-workloads.json (empty dict if absent/malformed)."""
    f = bench_dir / "bench-workloads.json"
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    shapes: dict[str, dict[str, Any]] = {}
    workloads = data.get("workloads", []) if isinstance(data, dict) else []
    for w in workloads:
        tag = w.get("tag") if isinstance(w, dict) else None
        if tag:
            shapes[str(tag)] = {
                "dataset": w.get("dataset") or str(tag),
                "isl": w.get("isl"),
                "osl": w.get("osl"),
            }
    return shapes


def _row_from_metrics(
    *,
    tag: str,
    dataset: str,
    isl: Any,
    osl: Any,
    concurrency: int,
    metrics: dict[str, Any],
    identity: dict[str, Any],
    captured_at: str,
    raw_path: str,
) -> AtlasCell:
    """One AtlasCell from a parsed <tag>-c<c>.txt + the caller's serve identity."""
    tp = int(identity["tensor_parallel"])
    out_tps = metrics.get("output_tps")
    tpot = metrics.get("tpot_med_ms")
    total_tps = metrics.get("total_tps")
    n = metrics.get("n_reqs")
    tin, tgen = metrics.get("total_input_tokens"), metrics.get("total_generated_tokens")
    # Prefer the MEASURED mean (totals/requests); fall back to the NOMINAL shape from
    # bench-workloads.json when the bench output omitted token totals.
    mean_in = (tin / n) if (tin and n) else (float(isl) if isl else None)
    mean_out = (tgen / n) if (tgen and n) else (float(osl) if osl else None)
    return AtlasCell(
        cell_id=tag,
        model=identity["model"],
        hardware=identity["hardware"],
        quant=identity["quant"],
        tensor_parallel=tp,
        parallel_strategy=identity["parallel_strategy"],
        mtp=False,
        max_num_batched_tokens=int(identity["max_num_batched_tokens"]),
        concurrency=concurrency,
        status=STATUS_FULL,
        ttft_avg_ms=metrics.get("ttft_med_ms"),
        request_throughput_avg=metrics.get("req_per_s"),
        output_tps_per_user=(1000.0 / tpot) if tpot else None,
        output_tps_per_gpu=(out_tps / tp) if out_tps else None,
        total_tps_per_gpu=(total_tps / tp) if total_tps else None,
        tpot_median_ms=tpot,
        itl_avg_ms=metrics.get("itl_med_ms"),
        mean_input_tokens=mean_in,
        mean_output_tokens=mean_out,
        cache_mode="cold",
        dataset=dataset,
        cudagraph_mode=identity["cudagraph_mode"],
        gpu_memory_utilization=identity["gpu_memory_utilization"],
        kv_cache_dtype=identity["kv_cache_dtype"],
        image=identity["image"],
        bench_backend=identity["bench_backend"],
        backend=BACKEND_VLLM_SWEEP,
        raw_path=raw_path,
        captured_at=captured_at,
        notes=f"imported from bench-all-workloads ({dataset}, c={concurrency})",
        extra={"imported_from_workloads": True},
    )


def import_workloads(
    bench_dir: Path,
    campaign_dir: Path,
    *,
    model: str,
    hardware: str,
    tensor_parallel: int,
    quant: str = "NVFP4",
    parallel_strategy: str = "TP",
    max_num_batched_tokens: int = 0,
    kv_cache_dtype: str = "unknown",
    image: str = "unknown",
    cudagraph_mode: str = "full",
    gpu_memory_utilization: float | None = None,
    bench_backend: str = "openai",
    dry_run: bool = False,
) -> WorkloadsImportResult:
    """Import a bench-all-workloads.sh output dir into per-workload campaign cells.

    One ``cells/<tag>/normalized.json`` per workload tag (each concurrency a row), tagged with
    the workload ``dataset`` + typed ISL/OSL from ``bench-workloads.json``. The serve-config
    identity is supplied by the caller (the bench output does not carry it). ``kv_cache_dtype`` /
    ``image`` default to ``"unknown"`` and ``gpu_memory_utilization`` to ``None``; supply the real
    serve values for a publish-``--strict``-clean campaign (those are required-context fields).
    """
    if not bench_dir.is_dir():
        raise FileNotFoundError(f"bench dir not found: {bench_dir}")
    shapes = _load_workload_shapes(bench_dir)
    identity = {
        "model": model,
        "hardware": hardware,
        "quant": quant,
        "tensor_parallel": tensor_parallel,
        "parallel_strategy": parallel_strategy,
        "max_num_batched_tokens": max_num_batched_tokens,
        "kv_cache_dtype": kv_cache_dtype,
        "image": image,
        "cudagraph_mode": cudagraph_mode,
        "gpu_memory_utilization": gpu_memory_utilization,
        "bench_backend": bench_backend,
    }

    # Group <tag>-c<c>.txt files by workload tag.
    by_tag: dict[str, list[tuple[int, Path]]] = {}
    for fp in sorted(bench_dir.glob("*-c*.txt")):
        m = _TAG_C_RE.match(fp.name)
        if not m:
            continue
        by_tag.setdefault(m.group("tag"), []).append((int(m.group("c")), fp))

    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cells_root = campaign_dir / "cells"
    tags_done: list[str] = []
    skipped: list[str] = []
    n_rows = 0
    for tag, files in sorted(by_tag.items()):
        shape = shapes.get(tag, {"dataset": tag, "isl": None, "osl": None})
        rows: list[AtlasCell] = []
        for concurrency, fp in sorted(files):
            metrics = _parse_sweep_file(fp, concurrency=concurrency)
            if metrics is None:
                continue
            rows.append(
                _row_from_metrics(
                    tag=tag,
                    dataset=shape["dataset"],
                    isl=shape["isl"],
                    osl=shape["osl"],
                    concurrency=concurrency,
                    metrics=metrics,
                    identity=identity,
                    captured_at=captured_at,
                    raw_path=str(fp),
                )
            )
        if not rows:
            skipped.append(tag)
            continue
        if not dry_run:
            cell_dir = cells_root / tag
            write_normalized_json(cell_dir, rows)
            write_status_file(cell_dir, STATUS_FULL)
            write_backend_file(cell_dir, BACKEND_VLLM_SWEEP)
        tags_done.append(tag)
        n_rows += len(rows)

    return WorkloadsImportResult(
        campaign_dir=campaign_dir,
        n_cells=len(tags_done),
        n_rows=n_rows,
        tags=tags_done,
        skipped=skipped,
    )
