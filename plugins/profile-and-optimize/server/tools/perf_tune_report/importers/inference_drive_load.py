"""Bundle importer for ``inference-perf-bench`` evidence dirs that use the
``drive_load.py`` JSONL load-driver pattern (Kimi K2.6 layout).

Sibling to ``inference_perf_bench.py`` (which handles the vLLM
``bench serve`` text-output pattern). Added in profile-and-optimize v1.21.0.

Source format conventions
-------------------------

Bundles produced by the Kimi-side ``drive_load.py`` (see
``perf-tune-kimi/scripts/drive_load.py``) and consumed by the
``inference-perf-bench`` skill record one request per JSONL line, captured
either as:

- **Multi-concurrency**: ``<bundle>/bench-c<NNN>/raw/load.jsonl``  (one
  subdir per sweep point, one JSONL per subdir)
- **Single-concurrency**: ``<bundle>/raw/load.jsonl``  (one JSONL at the
  bundle root)

Each line is:

.. code-block:: json

    {
      "shape": "long-short" | "mixed" | "long" | "short",
      "started_at": <epoch_seconds>,
      "duration_s": <float, total request latency>,
      "http_status": <int>,
      "prompt_tokens": <int>,
      "completion_tokens": <int>,
      "error": null | "<reason>"
    }

Aggregation contract (per JSONL file = one concurrency point)
-------------------------------------------------------------

For each load.jsonl we compute:

============================  =========================================================
Field                         Formula
============================  =========================================================
n_total                       count(lines)
n_ok                          count(error is None and http_status == 200)
n_fail                        n_total - n_ok
effective_duration_s          max(started_at + duration_s | ok) - min(started_at | ok)
req_per_s                     n_ok / effective_duration_s
total_input_tokens            sum(prompt_tokens | ok)
total_output_tokens           sum(completion_tokens | ok)
output_tps                    total_output_tokens / effective_duration_s
total_tps                     (input + output) / effective_duration_s
output_tps_per_user           median(completion_tokens / duration_s | ok)
ttft_median_ms                median(ttft_s | ok) * 1000, or None if no ttft_s recorded
============================  =========================================================

AtlasCell mapping
-----------------

============================  =========================================
AtlasCell field               Source
============================  =========================================
ttft_avg_ms                   median(ttft_s)*1000 when the run recorded ttft_s
                              (drive_load.py --stream-all / streaming shape);
                              else None (see note)
request_throughput_avg        req_per_s
output_tps_per_user           median(completion_tokens / duration_s)
output_tps_per_gpu            output_tps / tensor_parallel
============================  =========================================

TTFT note: drive_load.py records a real per-request ``ttft_s`` only when run
with ``--stream-all`` (or the ``streaming`` shape), which streams SSE and
times the first token chunk. When present we aggregate median(ttft_s) so the
cell is plot-ready. When absent (non-streaming canonical shapes) we leave
``ttft_avg_ms`` None -- we do NOT derive it from ``duration_s`` because that
would manufacture a number; re-run with ``--stream-all`` (or use a
vLLM-bench-serve bundle) to capture TTFT.

Concurrency point detection
---------------------------

Concurrency is parsed from the subdirectory name ``bench-c<NNN>`` (e.g.
``bench-c015`` -> 15, ``bench-c300`` -> 300). For single-c bundles the
concurrency MUST be provided via ``--concurrency`` (CLI override) or the
``partial_results_per_c`` key in the bundle's ``inference_perfbench_v1.json``.

Cross-reference with inference_perfbench_v1.json
------------------------------------------------

If the bundle has ``inference_perfbench_v1.json`` with a
``partial_results_per_c`` block, we use it as a *sanity check* on the
JSONL-derived numbers (warning if rps disagrees by >10%), not as the
primary source of truth. The JSONL is authoritative because it has the
raw per-request data.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import (
    BACKEND_VLLM_SWEEP,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)
from tools.perf_tune_report.importers.zymtrace_kernels import (
    KernelImportResult,
    import_zymtrace_kernels,
)


# Multi-concurrency layout: bundle/bench-c<NNN>/raw/load.jsonl
_BENCH_C_DIR = re.compile(r"^bench-c(\d+)$")


@dataclass(frozen=True)
class _JSONLFileInfo:
    """One concurrency point's load.jsonl path + parsed concurrency value."""

    path: Path
    concurrency: int


def _enumerate_jsonl_files(
    bundle: Path, *, concurrency_override: int | None = None
) -> list[_JSONLFileInfo]:
    """Find all load.jsonl files in the bundle.

    Two layouts supported:
    - Multi-c:  bundle/bench-c<NNN>/raw/load.jsonl (one entry per match)
    - Single-c: bundle/raw/load.jsonl (one entry, needs --concurrency override)

    Returns list sorted by concurrency.
    """
    files: list[_JSONLFileInfo] = []

    # Multi-concurrency: look for bench-c<NNN>/ subdirs
    for sub in bundle.iterdir():
        if not sub.is_dir():
            continue
        m = _BENCH_C_DIR.match(sub.name)
        if not m:
            continue
        jsonl_path = sub / "raw" / "load.jsonl"
        if jsonl_path.is_file():
            files.append(_JSONLFileInfo(path=jsonl_path, concurrency=int(m.group(1))))

    # Single-concurrency: bundle/raw/load.jsonl
    if not files:
        single = bundle / "raw" / "load.jsonl"
        if single.is_file():
            if concurrency_override is None:
                raise ValueError(
                    f"import_drive_load: single-concurrency bundle "
                    f"({single}) requires --concurrency override; bench-c<NNN>/ "
                    f"subdir convention is preferred for new bundles."
                )
            files.append(_JSONLFileInfo(path=single, concurrency=concurrency_override))

    files.sort(key=lambda x: x.concurrency)
    return files


@dataclass(frozen=True)
class _AggregatedMetrics:
    """Per-concurrency aggregates derived from one load.jsonl."""

    n_total: int
    n_ok: int
    n_fail: int
    effective_duration_s: float
    req_per_s: float
    total_input_tokens: int
    total_output_tokens: int
    output_tps: float
    total_tps: float
    output_tps_per_user: float | None  # median; None if no OK requests
    ttft_median_ms: float | None  # median(ttft_s)*1000; None if no ttft_s recorded


def _aggregate_jsonl(file_path: Path) -> _AggregatedMetrics | None:
    """Stream-parse one load.jsonl into aggregates.

    Returns None if no OK requests are present (e.g. a sweep point that
    crashed entirely with all-error lines).
    """
    n_total = 0
    n_ok = 0
    n_fail = 0
    total_input = 0
    total_output = 0
    starts: list[float] = []
    ends: list[float] = []
    per_user_tps: list[float] = []  # completion_tokens / duration_s, OK requests only
    ttfts_s: list[float] = []  # ttft_s, OK requests that recorded it (--stream-all)

    with file_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_fail += 1
                continue

            is_ok = rec.get("error") is None and rec.get("http_status") == 200
            if not is_ok:
                n_fail += 1
                continue

            n_ok += 1
            pt = int(rec.get("prompt_tokens", 0) or 0)
            ct = int(rec.get("completion_tokens", 0) or 0)
            total_input += pt
            total_output += ct

            started = float(rec.get("started_at", 0.0) or 0.0)
            duration = float(rec.get("duration_s", 0.0) or 0.0)
            if started > 0 and duration > 0:
                starts.append(started)
                ends.append(started + duration)
                if ct > 0:
                    per_user_tps.append(ct / duration)
            # ttft_s is present only when the run used --stream-all (or the
            # streaming shape). When present it makes the cell plot-ready.
            ttft = rec.get("ttft_s")
            if ttft is not None and float(ttft) > 0:
                ttfts_s.append(float(ttft))

    if n_ok == 0 or not starts:
        return None

    eff_dur = max(ends) - min(starts)
    if eff_dur <= 0:
        return None

    rps = n_ok / eff_dur
    output_tps = total_output / eff_dur
    total_tps = (total_input + total_output) / eff_dur
    median_per_user = statistics.median(per_user_tps) if per_user_tps else None
    ttft_median_ms = statistics.median(ttfts_s) * 1000.0 if ttfts_s else None

    return _AggregatedMetrics(
        n_total=n_total,
        n_ok=n_ok,
        n_fail=n_fail,
        effective_duration_s=eff_dur,
        req_per_s=rps,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        output_tps=output_tps,
        total_tps=total_tps,
        output_tps_per_user=median_per_user,
        ttft_median_ms=ttft_median_ms,
    )


@dataclass(frozen=True)
class DriveLoadImportResult:
    """Summary returned by ``import_drive_load_bundle``.

    Same shape as the inference-perf-bench importer's ImportResult so the
    CLI emits a consistent JSON payload regardless of which importer was
    dispatched.
    """

    campaign_dir: Path
    cell_id: str
    cell_dir: Path
    normalized_path: Path
    bundle_path: Path
    row_count: int
    concurrencies: list[int]
    k_values: list[int]
    status: str
    importer: str = "inference_drive_load"
    kernels: KernelImportResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["campaign_dir"] = str(self.campaign_dir)
        d["cell_dir"] = str(self.cell_dir)
        d["normalized_path"] = str(self.normalized_path)
        d["bundle_path"] = str(self.bundle_path)
        if self.kernels is not None:
            d["kernels"] = self.kernels.to_dict()
        return d


# Reuse identity helpers from the bench-serve importer (the metadata shape
# in inference_perfbench_v1.json is identical across the two importers).
from tools.perf_tune_report.importers.inference_perf_bench import (
    _CellIdentity,
    _identity_from_meta_and_overrides,
    _load_bundle_metadata,
)


def _row_from_aggregate(
    identity: _CellIdentity,
    concurrency: int,
    metrics: _AggregatedMetrics,
    jsonl_path: Path,
    captured_at: str,
) -> AtlasCell:
    """Build one AtlasCell row from drive-load aggregates."""
    output_tps_per_gpu = (
        metrics.output_tps / identity.tensor_parallel
        if identity.tensor_parallel
        else metrics.output_tps
    )
    total_tps_per_gpu = (
        metrics.total_tps / identity.tensor_parallel
        if identity.tensor_parallel
        else metrics.total_tps
    )
    # Mean ISL/OSL (shape) from the per-request token sums over OK requests.
    mean_input_tokens = (
        metrics.total_input_tokens / metrics.n_ok if metrics.n_ok else None
    )
    mean_output_tokens = (
        metrics.total_output_tokens / metrics.n_ok if metrics.n_ok else None
    )

    notes_parts = [
        f"imported from {jsonl_path.parent.parent.parent.name}",
        f"drive_load.py JSONL ({metrics.n_ok}/{metrics.n_total} OK)",
    ]
    if metrics.n_fail > 0:
        fail_rate = metrics.n_fail / metrics.n_total
        notes_parts.append(f"fail_rate={fail_rate * 100:.1f}%")
    if identity.speculative_num_tokens and identity.speculative_num_tokens > 1:
        notes_parts.append(f"num_speculative_tokens={identity.speculative_num_tokens}")
    if identity.patched_vllm_enabled:
        notes_parts.append("patchedVllm.enabled=true")
    if identity.notes:
        notes_parts.append(identity.notes)

    extra: dict[str, Any] = {
        "imported_from_drive_load": str(jsonl_path.parent.parent.parent),
        # TTFT is available only when the run recorded ttft_s (drive_load.py
        # --stream-all / streaming shape). Flag which case this row is.
        "ttft_unavailable_from_drive_load": metrics.ttft_median_ms is None,
        "ttft_from_stream_all": metrics.ttft_median_ms is not None,
        "n_total": metrics.n_total,
        "n_ok": metrics.n_ok,
        "n_fail": metrics.n_fail,
        "effective_duration_s": metrics.effective_duration_s,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "total_tps": metrics.total_tps,
    }
    if identity.max_num_seqs is not None:
        extra["max_num_seqs"] = identity.max_num_seqs
    if identity.patched_vllm_enabled is not None:
        extra["patched_vllm_enabled"] = bool(identity.patched_vllm_enabled)

    return AtlasCell(
        cell_id=identity.cell_id,
        model=identity.model,
        hardware=identity.hardware,
        quant=identity.quant,
        tensor_parallel=identity.tensor_parallel,
        parallel_strategy=identity.parallel_strategy,
        mtp=identity.mtp,
        max_num_batched_tokens=identity.max_num_batched_tokens,
        concurrency=concurrency,
        status=STATUS_FULL if metrics.n_fail == 0 else STATUS_PARTIAL,
        ttft_avg_ms=metrics.ttft_median_ms,
        request_throughput_avg=metrics.req_per_s,
        output_tps_per_user=metrics.output_tps_per_user,
        output_tps_per_gpu=output_tps_per_gpu,
        total_tps_per_gpu=total_tps_per_gpu,
        mean_input_tokens=mean_input_tokens,
        mean_output_tokens=mean_output_tokens,
        prefix_cache_hit_rate=identity.prefix_cache_hit_rate,
        cache_mode=identity.cache_mode,
        backend=BACKEND_VLLM_SWEEP,
        raw_path=str(jsonl_path),
        captured_at=captured_at,
        notes=" | ".join(notes_parts),
        extra=extra,
    )


def import_drive_load_bundle(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
    concurrency_override: int | None = None,
) -> DriveLoadImportResult:
    """Convert one Kimi/drive-load bundle into a ``cells/<id>/normalized.json``.

    Args:
        bundle: ``*-deploy/experiments/artifacts/inference-perf-bench/<bundle>/``
            path. Must contain either ``bench-c<NNN>/raw/load.jsonl`` subdirs
            OR a single ``raw/load.jsonl`` plus ``--concurrency`` override.
        campaign_dir: target campaign directory (must exist).
        overrides: identity-field overrides (see inference_perf_bench).
        dry_run: parse + validate only.
        captured_at: ISO-8601 timestamp to stamp into rows.
        concurrency_override: required for single-c bundles
            (bundle/raw/load.jsonl with no bench-c<NNN>/ subdirs).

    Returns:
        ``DriveLoadImportResult`` summarizing what was imported.

    Raises:
        ValueError: bundle does not exist; no load.jsonl files found;
            required identity field missing.
    """
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"import_drive_load: bundle does not exist: {bundle}")

    jsonl_files = _enumerate_jsonl_files(
        bundle, concurrency_override=concurrency_override
    )
    if not jsonl_files:
        raise ValueError(
            f"import_drive_load: no load.jsonl files found in {bundle} "
            f"(looked for bench-c<NNN>/raw/load.jsonl and raw/load.jsonl)"
        )

    bundle_meta = _load_bundle_metadata(bundle)
    identity = _identity_from_meta_and_overrides(
        bundle, bundle_meta, overrides or {}
    )

    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[AtlasCell] = []
    skipped: list[Path] = []
    for jf in jsonl_files:
        metrics = _aggregate_jsonl(jf.path)
        if metrics is None:
            skipped.append(jf.path)
            continue
        rows.append(
            _row_from_aggregate(
                identity=identity,
                concurrency=jf.concurrency,
                metrics=metrics,
                jsonl_path=jf.path,
                captured_at=captured_at,
            )
        )

    if not rows:
        raise ValueError(
            f"import_drive_load: no valid load.jsonl results parsed from {bundle} "
            f"({len(skipped)} files were unparseable or had zero OK requests)"
        )

    status = STATUS_FULL if not skipped else STATUS_PARTIAL

    cell_dir = campaign_dir / "cells" / identity.cell_id
    concurrencies = sorted({r.concurrency for r in rows})
    k_values: list[int] = []  # drive_load.py does not currently sweep K
    kernels_result = import_zymtrace_kernels(bundle, cell_dir, dry_run=dry_run)

    if dry_run:
        return DriveLoadImportResult(
            campaign_dir=campaign_dir,
            cell_id=identity.cell_id,
            cell_dir=cell_dir,
            normalized_path=cell_dir / "normalized.json",
            bundle_path=bundle,
            row_count=len(rows),
            concurrencies=concurrencies,
            k_values=k_values,
            status=status,
            kernels=kernels_result,
        )

    cell_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = cell_dir / "normalized.json"
    normalized_path.write_text(
        json.dumps([r.to_dict() for r in rows], indent=2, sort_keys=True)
    )
    (cell_dir / "status.txt").write_text(status + "\n")
    (cell_dir / "backend.txt").write_text(BACKEND_VLLM_SWEEP + "\n")
    kernels_line = (
        f"- kernels.json:  {kernels_result.kernels_json_path}\n"
        if kernels_result.kernels_json_path is not None
        else f"- kernels.json:  (skipped: {kernels_result.skipped_reason})\n"
    )
    (cell_dir / "SOURCE.md").write_text(
        f"# {identity.cell_id}\n\n"
        f"- imported_from: {bundle}\n"
        f"- importer:      inference_drive_load (v1.21.0)\n"
        f"- captured_at:   {captured_at}\n"
        f"- concurrencies: {concurrencies}\n"
        f"- row_count:     {len(rows)}\n"
        f"- status:        {status}\n"
        f"- skipped_files: {len(skipped)}\n"
        + kernels_line
    )

    return DriveLoadImportResult(
        campaign_dir=campaign_dir,
        cell_id=identity.cell_id,
        cell_dir=cell_dir,
        normalized_path=normalized_path,
        bundle_path=bundle,
        row_count=len(rows),
        concurrencies=concurrencies,
        k_values=k_values,
        status=status,
        kernels=kernels_result,
    )


# ----------------------------------------------------------------------------
# Auto-detect dispatcher
# ----------------------------------------------------------------------------

def detect_bundle_pattern(bundle: Path) -> str:
    """Detect which importer to use for a given bundle directory.

    Returns one of:
    - ``"inference_perf_bench"`` if the bundle has any ``raw/sweep-c*.txt``
      or ``raw/sweep-K*-c*.txt`` files
    - ``"inference_drive_load"`` if the bundle has ``bench-c<NNN>/raw/load.jsonl``
      or ``raw/load.jsonl`` (and no sweep-*.txt files)
    - ``"unknown"`` otherwise (caller raises a descriptive error)

    Detection is intentionally fast (single dir read of bundle/raw/ + a
    minimal listdir for bench-c<NNN>/).
    """
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        return "unknown"

    raw_dir = bundle / "raw"
    if raw_dir.is_dir():
        for entry in raw_dir.iterdir():
            if entry.is_file() and (
                entry.name.startswith("sweep-c")
                or entry.name.startswith("sweep-K")
            ):
                return "inference_perf_bench"
        if (raw_dir / "load.jsonl").is_file():
            return "inference_drive_load"

    # Multi-c drive-load layout: bench-c<NNN>/raw/load.jsonl
    for sub in bundle.iterdir():
        if not sub.is_dir():
            continue
        if _BENCH_C_DIR.match(sub.name) and (sub / "raw" / "load.jsonl").is_file():
            return "inference_drive_load"

    return "unknown"
