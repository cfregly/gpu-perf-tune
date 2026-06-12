"""aiperf backend: drives the inference-tools/perf-bench runbook.

Automates phases 2-8 of the vendored ``server/inference-tools/perf-bench/SKILL.md``
runbook (bench-pod create/wait, dataset copy, concurrency sweep via ``aiperf
profile``, log download). Parses AIPerf's JSON/log output into canonical
AtlasCell rows under ``<campaign>/cells/<cell-id>/normalized.json``.

The runner is deliberately thin: it shells out to ``kubectl exec`` for the
sweep commands and parses the per-concurrency ``aiperf-c<N>.json`` AIPerf
emits by default. Operator must provide:

- target endpoint (cluster-internal service URL)
- bench pod name + namespace + kube context
- dataset split (defaults to ``2025_07`` per SKILL.md)
- conversation count (defaults to all in split)

Status mapping mirrors vllm_sweep:
- all requested concurrencies completed -> full
- some but not all completed -> partial
- none completed -> failed
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
    BACKEND_AIPERF,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)


@dataclass(frozen=True)
class AiperfResult:
    cell_dir: Path
    status: str
    row_count: int
    commands: list[list[str]]
    dry_run: bool


def build_command(
    cell: CellConfig,
    concurrency: int,
    *,
    namespace: str,
    bench_pod: str,
    kube_context: str,
    endpoint_url: str,
    served_model: str,
    dataset_split: str,
    conversation_count: int | None = None,
) -> list[str]:
    """One ``kubectl exec ... aiperf profile`` invocation for one concurrency."""
    inner = [
        "aiperf",
        "profile",
        "--model",
        served_model,
        "--url",
        endpoint_url,
        "--input-file",
        f"/tmp/replay-data/{dataset_split}.jsonl",
        "--custom-dataset-type",
        "mooncake-trace",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--use-server-token-count",
        "--tokenizer-trust-remote-code",
        "--connection-reuse-strategy",
        "sticky-user-sessions",
        "--dataset-sampling-strategy",
        "sequential",
        "--concurrency",
        str(concurrency),
        "--output-artifact-dir",
        f"/tmp/aiperf-{cell.cell_id}-c{concurrency}",
    ]
    if conversation_count is not None:
        inner += ["--conversation-num", str(conversation_count)]

    return [
        "kubectl",
        "exec",
        "-n",
        namespace,
        "--context",
        kube_context,
        bench_pod,
        "--",
        *inner,
    ]


def _parse_aiperf_json(json_path: Path) -> dict | None:
    try:
        return json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _metric_value(source: dict, *keys: str) -> float | None:
    """Return the first present metric, unwrapping aiperf>=0.9 ``{"avg": ...}``.

    aiperf 0.9 emits each metric as a nested ``{"unit": ..., "avg": ...}`` object
    rather than a flat scalar; older versions emitted a bare float. Accept both.
    """
    for key in keys:
        if key not in source:
            continue
        v = source[key]
        if isinstance(v, dict):
            v = v.get("avg")
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _extract_metrics(report: dict) -> dict | None:
    """AIPerf reports vary by version. Try the common locations.

    aiperf >=0.9 nests each metric as ``{"unit": ..., "avg": ...}`` and renames
    output throughput to ``output_token_throughput`` and ttft to
    ``time_to_first_token``; the older flat-scalar keys are still accepted.
    """
    # AIPerf often nests metrics under ``results`` or at top level.
    candidates = [report, report.get("results", {}), report.get("summary", {})]
    for source in candidates:
        if not isinstance(source, dict):
            continue
        ttft = _metric_value(
            source, "time_to_first_token", "median_ttft_ms", "mean_ttft_ms", "ttft_avg_ms"
        )
        rt = _metric_value(source, "request_throughput", "request_throughput_avg")
        ot = _metric_value(
            source,
            "output_token_throughput",
            "output_throughput",
            "output_throughput_tps",
            "output_tps",
        )
        if ttft is not None and rt is not None and ot is not None:
            # Reasoning-model split (optional; None for non-reasoning models, then
            # TTFO == TTFT). AIPerf TTFT = first token of ANY type (incl. reasoning);
            # TTFO = first non-reasoning/answer token. Capturing both stops the
            # under-reporting that hid MiniMax-M2.7's ~4s think phase behind a
            # 0.10s "TTFT" (minimax-aabench). reasoning_token_count quantifies the think.
            ttfo = _metric_value(
                source, "time_to_first_output_token", "median_ttfo_ms", "mean_ttfo_ms"
            )
            reasoning_tokens = _metric_value(
                source, "reasoning_token_count", "reasoning_tokens", "output_reasoning_token_count"
            )
            # TTFO denominator guard: AIPerf averages ttfo over only the requests
            # that emitted an answer token (all-reasoning requests are excluded).
            ttfo_coverage = None
            tfo_d, tft_d = source.get("time_to_first_output_token"), source.get("time_to_first_token")
            if isinstance(tfo_d, dict) and isinstance(tft_d, dict):
                c_o, c_t = tfo_d.get("count"), tft_d.get("count")
                if isinstance(c_o, (int, float)) and isinstance(c_t, (int, float)) and c_t:
                    ttfo_coverage = float(c_o) / float(c_t)
            elif tfo_d is None and isinstance(tft_d, dict) and tft_d.get("count"):
                # reasoning parser ON but ZERO requests answered: aiperf omits the
                # ttfo metric entirely. That is coverage 0, not unknown.
                ttfo_coverage = 0.0
            return {
                "ttft_ms": ttft,
                "request_throughput": rt,
                "output_throughput": ot,
                "ttfo_ms": ttfo,
                "reasoning_tokens": reasoning_tokens,
                "ttfo_coverage": ttfo_coverage,
            }
    return None


def normalize_outputs(cell: CellConfig, raw_dir: Path, cell_dir: Path) -> tuple[list[AtlasCell], str]:
    """Parse per-concurrency AIPerf reports into AtlasCell rows."""
    captured_at = utc_now_iso()
    rows: list[AtlasCell] = []
    measured: set[int] = set()

    if raw_dir.is_dir():
        for report_path in sorted(raw_dir.rglob("*.json")):
            # Filename convention: aiperf-<cell_id>-c<concurrency>/profile_export_aiperf.json
            try:
                # Parse "c<N>" from the parent directory name.
                parent = report_path.parent.name
                concurrency = int(parent.rsplit("c", 1)[-1])
            except (ValueError, IndexError):
                continue
            report = _parse_aiperf_json(report_path)
            if not report:
                continue
            metrics = _extract_metrics(report)
            if not metrics:
                continue
            tps_per_gpu = metrics["output_throughput"] / max(cell.tensor_parallel, 1)
            tps_per_user = metrics["output_throughput"] / max(concurrency, 1)
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
                    status=STATUS_FULL,  # provisional
                    ttft_avg_ms=metrics["ttft_ms"],
                    ttfo_avg_ms=metrics.get("ttfo_ms"),
                    reasoning_token_count=metrics.get("reasoning_tokens"),
                    ttfo_coverage=metrics.get("ttfo_coverage"),
                    request_throughput_avg=metrics["request_throughput"],
                    output_tps_per_user=tps_per_user,
                    output_tps_per_gpu=tps_per_gpu,
                    backend=BACKEND_AIPERF,
                    raw_path=str(report_path.relative_to(cell_dir.parent.parent)),
                    captured_at=captured_at,
                )
            )
            measured.add(concurrency)

    requested = set(cell.concurrencies)
    if not measured:
        return [], STATUS_FAILED
    status = STATUS_FULL if measured >= requested else STATUS_PARTIAL
    rows = [AtlasCell(**{**r.to_dict(), "status": status}) for r in rows]
    return rows, status


def run_cell(
    cell: CellConfig,
    campaign_dir: Path,
    *,
    namespace: str,
    bench_pod: str,
    kube_context: str,
    endpoint_url: str,
    served_model: str,
    dataset_split: str = "2025_07",
    conversation_count: int | None = None,
    dry_run: bool = False,
    subprocess_runner=subprocess.run,
) -> AiperfResult:
    """Execute one cell's AIPerf sweep end-to-end (one ``aiperf profile`` per concurrency)."""
    cell_dir = campaign_dir / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_backend_file(cell_dir, BACKEND_AIPERF)

    commands: list[list[str]] = []
    for c in cell.concurrencies:
        cmd = build_command(
            cell,
            c,
            namespace=namespace,
            bench_pod=bench_pod,
            kube_context=kube_context,
            endpoint_url=endpoint_url,
            served_model=served_model,
            dataset_split=dataset_split,
            conversation_count=conversation_count,
        )
        commands.append(cmd)

    (cell_dir / "commands").mkdir(exist_ok=True)
    (cell_dir / "commands" / "aiperf-sweep.cmd").write_text(
        "\n".join(shlex.join(c) for c in commands) + "\n"
    )

    if dry_run:
        write_status_file(cell_dir, STATUS_FAILED)
        return AiperfResult(
            cell_dir=cell_dir,
            status="dry-run",
            row_count=0,
            commands=commands,
            dry_run=True,
        )

    raw_dir = cell_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    exits: list[int] = []
    for cmd, c in zip(commands, cell.concurrencies):
        proc = subprocess_runner(cmd, capture_output=True, text=True, check=False)
        exits.append(proc.returncode)
        stdout_chunks.append(f"# concurrency={c} exit={proc.returncode}\n{proc.stdout or ''}")
        stderr_chunks.append(f"# concurrency={c} exit={proc.returncode}\n{proc.stderr or ''}")
        # Pull the AIPerf artifact dir back into raw/.
        kubectl_cp = [
            "kubectl",
            "cp",
            "--context",
            kube_context,
            f"{namespace}/{bench_pod}:/tmp/aiperf-{cell.cell_id}-c{c}",
            str(raw_dir / f"c{c}"),
        ]
        subprocess_runner(kubectl_cp, capture_output=True, text=True, check=False)

    (cell_dir / "commands" / "aiperf-sweep.stdout").write_text("\n".join(stdout_chunks))
    (cell_dir / "commands" / "aiperf-sweep.stderr").write_text("\n".join(stderr_chunks))
    (cell_dir / "commands" / "aiperf-sweep.exit").write_text(
        "\n".join(str(e) for e in exits) + "\n"
    )

    rows, status = normalize_outputs(cell, raw_dir, cell_dir)
    write_normalized_json(cell_dir, rows)
    write_status_file(cell_dir, status)

    return AiperfResult(
        cell_dir=cell_dir,
        status=status,
        row_count=len(rows),
        commands=commands,
        dry_run=False,
    )
