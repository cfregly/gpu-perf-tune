"""vllm-sweep backend: drives `vllm bench sweep serve_workload`.

The runner translates one :class:`CellConfig` into the inputs that the
upstream ``vllm bench sweep serve_workload`` CLI expects, invokes it
against a live cluster endpoint (or dry-runs the command for verification),
then normalizes the per-run JSON output into the canonical AtlasCell rows
under ``<campaign>/cells/<cell-id>/normalized.json``.

Upstream CLI reference:
- ``vllm/vllm/benchmarks/sweep/serve_workload.py``
- ``vllm/vllm/benchmarks/sweep/serve.py``

Field mapping from the upstream per-run JSON to AtlasCell:

- ``median_ttft_ms``           -> ``ttft_avg_ms``
- ``request_throughput``       -> ``request_throughput_avg``
- ``output_throughput / TP``   -> ``output_tps_per_gpu``
- ``output_throughput / C``    -> ``output_tps_per_user``

The cell's overall status comes from the sweep's exit code + per-concurrency
completeness check (all requested concurrencies present -> full; some but
not all -> partial; none -> failed).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.perf_tune_report.runners.common import (
    CellConfig,
    utc_now_iso,
    write_backend_file,
    write_normalized_json,
    write_status_file,
)
from tools.perf_tune_report.schema import (
    BACKEND_VLLM_SWEEP,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)


@dataclass(frozen=True)
class VllmSweepResult:
    cell_dir: Path
    status: str
    row_count: int
    command: list[str]
    dry_run: bool


def build_command(
    cell: CellConfig,
    *,
    bench_cmd: str,
    serve_cmd: str,
    output_dir: Path,
) -> list[str]:
    """Compose the `vllm bench sweep serve_workload` shell invocation.

    ``serve_cmd`` and ``bench_cmd`` are operator-supplied base commands (per
    upstream contract). The cell's TP / EP-or-TP / MTP / max_num_batched_tokens
    knobs are passed as a per-cell ``--serve-params`` JSON file; concurrencies
    become the ``--workload-iters`` list with ``--workload-var max_concurrency``.
    """
    serve_params_file = output_dir / "serve-params.json"
    serve_params = [
        {
            "_benchmark_name": cell.cell_id,
            "tensor_parallel_size": cell.tensor_parallel,
            "max_num_batched_tokens": cell.max_num_batched_tokens,
            **cell.extras.get("serve_params_extra", {}),
        }
    ]
    if cell.parallel_strategy == "EP":
        serve_params[0]["enable_expert_parallel"] = True
    if cell.mtp:
        # Honor the cell's K so the SERVED config matches the RECORDED num_speculative_tokens
        # (was hardcoded to 1, which silently served K=1 even when the cell asked for K=2/3).
        _spx_spec = (cell.extras.get("serve_params_extra", {}) or {}).get("speculative_config") or {}
        _k = (_spx_spec.get("num_speculative_tokens")
              or cell.extras.get("num_speculative_tokens")
              or cell.extras.get("spec_decode_k")
              or 1)
        serve_params[0]["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(_k)}
    output_dir.mkdir(parents=True, exist_ok=True)
    serve_params_file.write_text(json.dumps(serve_params, indent=2) + "\n")

    return [
        "vllm",
        "bench",
        "sweep",
        "serve_workload",
        "--serve-cmd",
        serve_cmd,
        "--bench-cmd",
        bench_cmd,
        "--workload-var",
        "max_concurrency",
        "--workload-iters",
        ",".join(str(c) for c in cell.concurrencies),
        "--serve-params",
        str(serve_params_file),
        "--output-dir",
        str(output_dir / "raw"),
        "--experiment-name",
        cell.cell_id,
        "--num-runs",
        str(cell.extras.get("num_runs", 1)),
    ]


def _parse_per_run_json(per_run_path: Path) -> dict:
    return json.loads(per_run_path.read_text())


def _descriptor_from_cell(cell: CellConfig) -> dict:
    """Extract the full-context descriptor fields from the cell config (AGENTS.md
    "Every performance number carries its full context"). Sourced from
    ``cell.extras`` (+ the ``serve_params_extra`` knobs); never fabricated -- an
    unspecified field stays "unknown"/None so the methodology gate flags it.
    ``cudagraph_mode`` defaults to "full" (the vLLM serving default) unless
    ``enforce_eager`` is set, since a sweep serves with cudagraph on by default."""
    extras = cell.extras or {}
    spx = extras.get("serve_params_extra", {}) or {}

    def pick(*keys):
        for src in (extras, spx):
            for k in keys:
                v = src.get(k)
                if v is not None:
                    return v
        return None

    enforce_eager = bool(pick("enforce_eager"))
    cudagraph_mode = pick("cudagraph_mode") or ("eager" if enforce_eager else "full")
    gmu = pick("gpu_memory_utilization", "gpu_memory_util")
    # Serving-variant knobs (2026-06-07): the typed variant descriptor so variant_key
    # distinguishes MTP-K / async / prefix-caching. Derived from the cell (never fabricated:
    # an unspecified knob stays None). num_speculative_tokens mirrors what build_command serves.
    spec_cfg = spx.get("speculative_config") if isinstance(spx.get("speculative_config"), dict) else {}
    nst = pick("num_speculative_tokens", "spec_decode_k")
    if nst is None and spec_cfg:
        nst = spec_cfg.get("num_speculative_tokens")
    if nst is None and cell.mtp:
        nst = 1
    return {
        "dataset": pick("dataset") or "unknown",
        "cudagraph_mode": cudagraph_mode,
        "gpu_memory_utilization": float(gmu) if isinstance(gmu, (int, float)) else None,
        "kv_cache_dtype": pick("kv_cache_dtype") or "unknown",
        "image": pick("image", "vllm_version", "vllm_commit") or "unknown",
        "num_speculative_tokens": int(nst) if nst is not None else None,
        "async_scheduling": pick("async_scheduling"),
        "max_num_seqs": pick("max_num_seqs"),
        "enable_prefix_caching": pick("enable_prefix_caching", "prefix_cache"),
        "bench_backend": pick("bench_backend"),
    }


def normalize_outputs(cell: CellConfig, raw_dir: Path, cell_dir: Path) -> tuple[list[AtlasCell], str]:
    """Read upstream JSON outputs and convert to canonical AtlasCell rows."""
    captured_at = utc_now_iso()
    descriptor = _descriptor_from_cell(cell)
    rows: list[AtlasCell] = []
    measured_concurrencies: set[int] = set()

    if raw_dir.is_dir():
        for run_json in sorted(raw_dir.rglob("*.json")):
            try:
                data = _parse_per_run_json(run_json)
            except (OSError, json.JSONDecodeError):
                continue
            # Be lenient about the upstream schema -- different vLLM versions
            # emit slightly different field sets. We only need 5 fields.
            try:
                concurrency = int(
                    data.get("max_concurrency")
                    or data.get("max_concurrent_requests")
                    or 0
                )
            except (TypeError, ValueError):
                concurrency = 0
            if concurrency <= 0:
                continue
            ttft_ms = data.get("median_ttft_ms") or data.get("mean_ttft_ms")
            req_throughput = data.get("request_throughput")
            output_throughput = data.get("output_throughput") or data.get("output_throughput_tps")
            if ttft_ms is None or req_throughput is None or output_throughput is None:
                continue
            tps_per_gpu = float(output_throughput) / max(cell.tensor_parallel, 1)
            tps_per_user = float(output_throughput) / max(concurrency, 1)

            rows.append(
                AtlasCell(
                    cell_id=cell.cell_id,
                    model=cell.model,
                    hardware=cell.hardware,
                    quant=cell.quant,
                    tensor_parallel=cell.tensor_parallel,
                    parallel_strategy=cell.parallel_strategy,
                    mtp=cell.mtp,
                    max_num_batched_tokens=cell.max_num_batched_tokens,
                    concurrency=concurrency,
                    status=STATUS_FULL,  # provisional; status updated below
                    ttft_avg_ms=float(ttft_ms),
                    request_throughput_avg=float(req_throughput),
                    output_tps_per_user=tps_per_user,
                    output_tps_per_gpu=tps_per_gpu,
                    backend=BACKEND_VLLM_SWEEP,
                    raw_path=str(run_json.relative_to(cell_dir.parent.parent)),
                    captured_at=captured_at,
                    **descriptor,
                )
            )
            measured_concurrencies.add(concurrency)

    requested = set(cell.concurrencies)
    if not measured_concurrencies:
        status = STATUS_FAILED
        return [], status
    if measured_concurrencies < requested:
        status = STATUS_PARTIAL
    else:
        status = STATUS_FULL
    # Stamp the resolved status onto each row.
    rows = [
        AtlasCell(**{**r.to_dict(), "status": status}) for r in rows
    ]
    return rows, status


def run_cell(
    cell: CellConfig,
    campaign_dir: Path,
    *,
    serve_cmd: str,
    bench_cmd: str,
    dry_run: bool = False,
    subprocess_runner=subprocess.run,
) -> VllmSweepResult:
    """Execute one cell's vLLM-sweep run end-to-end."""
    cell_dir = campaign_dir / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_backend_file(cell_dir, BACKEND_VLLM_SWEEP)

    cmd = build_command(
        cell, bench_cmd=bench_cmd, serve_cmd=serve_cmd, output_dir=cell_dir
    )

    (cell_dir / "commands").mkdir(exist_ok=True)
    (cell_dir / "commands" / "vllm-sweep.cmd").write_text(
        shlex.join(cmd) + "\n"
    )

    if dry_run:
        write_status_file(cell_dir, STATUS_FAILED)  # not yet run
        return VllmSweepResult(
            cell_dir=cell_dir,
            status="dry-run",
            row_count=0,
            command=cmd,
            dry_run=True,
        )

    proc = subprocess_runner(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    (cell_dir / "commands" / "vllm-sweep.stdout").write_text(proc.stdout or "")
    (cell_dir / "commands" / "vllm-sweep.stderr").write_text(proc.stderr or "")
    (cell_dir / "commands" / "vllm-sweep.exit").write_text(str(proc.returncode) + "\n")

    rows, status = normalize_outputs(cell, cell_dir / "raw", cell_dir)
    write_normalized_json(cell_dir, rows)
    write_status_file(cell_dir, status)

    return VllmSweepResult(
        cell_dir=cell_dir,
        status=status,
        row_count=len(rows),
        command=cmd,
        dry_run=False,
    )
