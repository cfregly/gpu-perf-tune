"""Bundle importer for ``inference-perf-bench`` evidence dirs.

Reads a single
``*-deploy/experiments/artifacts/inference-perf-bench/<bundle>/`` directory
and writes a perf-report-compatible ``cells/<cell-id>/normalized.json`` into a
named campaign directory.

Source format conventions (vLLM ``bench serve`` text output, captured by
``perf-tune-glm51/03f-variant-runner.sh`` + ``perf-tune-dsv4/03f-variant-runner.sh``
+ analogous Kimi scripts):

- One file per concurrency at ``raw/sweep-c<N>.txt`` (single-K runs)
- OR one file per ``(K, concurrency)`` at ``raw/sweep-K<K>-c<N>.txt`` (DSv4 K-sweep runs)
- Each file contains the human-readable ``Serving Benchmark Result`` block with
  these grep-able lines:

    Successful requests:                     <int>
    Benchmark duration (s):                  <float>
    Request throughput (req/s):              <float>
    Output token throughput (tok/s):         <float>
    Total token throughput (tok/s):          <float>
    Median TTFT (ms):                        <float>
    Median TPOT (ms):                        <float>

Bundle-level metadata (model, hardware, quant, TP/EP, MTP, mbt) is sourced
from a sibling ``inference_perfbench_v1.json`` if present; otherwise the
caller MUST provide the values via CLI flags or the function's keyword args.

Output AtlasCell mapping:

- ``ttft_avg_ms``             = ``Median TTFT (ms)``       (median, not avg — see note below)
- ``request_throughput_avg``  = ``Request throughput (req/s)`` (or successful/duration if absent)
- ``output_tps_per_user``     = 1000 / ``Median TPOT (ms)``
- ``output_tps_per_gpu``      = ``Output token throughput (tok/s)`` / ``tensor_parallel``

Median-vs-average note: the schema field is named ``ttft_avg_ms`` for legacy
reasons (it predates the per-percentile capture). When ingesting from this
text format we populate it from ``Median TTFT`` because that's the most
robust number (insensitive to a single outlier request). Future schema bumps
may rename or split this — for now we document the substitution in the
emitted row's ``notes`` field.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import (
    BACKEND_SGLANG_SWEEP,
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


# Regexes are anchored on the leading label so we don't pick up
# percentile blocks or summaries from other parts of vllm's output.
_REGEX = {
    "n_reqs": re.compile(r"^Successful requests:\s+(\d+)\s*$", re.MULTILINE),
    "duration_s": re.compile(r"^Benchmark duration \(s\):\s+([\d.]+)\s*$", re.MULTILINE),
    "req_per_s": re.compile(r"^Request throughput \(req/s\):\s+([\d.]+)\s*$", re.MULTILINE),
    "output_tps": re.compile(
        r"^Output token throughput \(tok/s\):\s+([\d.]+)\s*$", re.MULTILINE
    ),
    "total_tps": re.compile(
        r"^Total token throughput \(tok/s\):\s+([\d.]+)\s*$", re.MULTILINE
    ),
    # Total input / generated tokens -> mean ISL/OSL per request (the shape
    # dimension pricing/capacity analysis needs). Derived against n_reqs below.
    "total_input_tokens": re.compile(
        r"^Total input tokens:\s+(\d+)\s*$", re.MULTILINE
    ),
    "total_generated_tokens": re.compile(
        r"^Total generated tokens:\s+(\d+)\s*$", re.MULTILINE
    ),
    "ttft_med_ms": re.compile(r"^Median TTFT \(ms\):\s+([\d.]+)\s*$", re.MULTILINE),
    "tpot_med_ms": re.compile(r"^Median TPOT \(ms\):\s+([\d.]+)\s*$", re.MULTILINE),
}

# ``sweep-c<N>.txt`` (single-K runs)
_SWEEP_SIMPLE = re.compile(r"^sweep-c(\d+)\.txt$")
# ``sweep-K<K>-c<N>.txt`` (multi-K runs; e.g. DSv4 K=1/2/3 sweeps)
_SWEEP_K = re.compile(r"^sweep-K(\d+)-c(\d+)\.txt$")


@dataclass(frozen=True)
class _SweepFileInfo:
    """Identifying info parsed from a sweep filename."""

    path: Path
    concurrency: int
    k: int  # 1 for sweep-c<N>.txt; explicit K for sweep-K<K>-c<N>.txt


def _enumerate_sweep_files(bundle: Path) -> list[_SweepFileInfo]:
    """Find all sweep-* files in ``bundle/raw/``. Returns sorted-by-(K, c)."""
    raw_dir = bundle / "raw"
    if not raw_dir.is_dir():
        return []
    files: list[_SweepFileInfo] = []
    for fp in raw_dir.iterdir():
        if not fp.is_file():
            continue
        m = _SWEEP_K.match(fp.name)
        if m:
            files.append(
                _SweepFileInfo(path=fp, concurrency=int(m.group(2)), k=int(m.group(1)))
            )
            continue
        m = _SWEEP_SIMPLE.match(fp.name)
        if m:
            files.append(_SweepFileInfo(path=fp, concurrency=int(m.group(1)), k=1))
    files.sort(key=lambda x: (x.k, x.concurrency))
    return files


def _parse_metrics(file_path: Path) -> dict[str, float | int] | None:
    """Extract metrics from a single sweep-c<N>.txt file.

    Returns ``None`` if no valid benchmark result is present (e.g. crashed
    sweep with no usable output).
    """
    text = file_path.read_text(errors="replace")
    out: dict[str, float | int] = {}
    for key, rx in _REGEX.items():
        m = rx.search(text)
        if m:
            val = m.group(1)
            out[key] = int(val) if key == "n_reqs" else float(val)
    # Require at least the bare minimum to call this a valid measurement:
    # we need either (req_per_s OR n_reqs+duration) AND (tpot OR output_tps).
    has_throughput = "req_per_s" in out or ("n_reqs" in out and "duration_s" in out)
    has_latency = "tpot_med_ms" in out or "output_tps" in out
    if not (has_throughput and has_latency):
        return None
    # Derive req_per_s if missing
    if "req_per_s" not in out and "n_reqs" in out and "duration_s" in out:
        out["req_per_s_derived"] = out["n_reqs"] / out["duration_s"]
    return out


@dataclass(frozen=True)
class ImportResult:
    """Summary returned by ``import_perf_bench_bundle``.

    Mirrors the shape returned by the other CLI verbs for consistent JSON
    output.
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
    # Optional zymtrace per-kernel breakdown import result. ``None`` when the
    # bundle did NOT declare zymtrace coverage via capture_sources.json
    # (correct no-op for non-zymtrace bundles). When present, indicates the
    # kernels.json file written next to normalized.json for the renderer
    # to pick up via the 3rd-page module.
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


def _load_bundle_metadata(bundle: Path) -> dict[str, Any]:
    """Try to read ``inference_perfbench_v1.json`` from the bundle root.

    This file may not exist in older / hand-built bundles; we fall back to
    CLI-supplied values in that case.
    """
    candidate = bundle / "inference_perfbench_v1.json"
    if candidate.is_file():
        try:
            return json.loads(candidate.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class _CellIdentity:
    """The identity fields the AtlasCell schema requires.

    These are the bundle-level invariants (same for every concurrency point
    in the bundle), separate from the per-concurrency metric fields.
    """

    cell_id: str
    model: str
    hardware: str
    quant: str
    tensor_parallel: int
    parallel_strategy: str
    mtp: bool
    max_num_batched_tokens: int

    # Serving engine (cross-engine A/B). "vllm" (default) | "sglang". Drives the
    # AtlasCell.backend tag (vllm-sweep vs sglang-sweep) + extra["engine"] so the
    # cross-engine view / compare-engines can group by engine.
    engine: str = "vllm"

    # Optional extra fields surfaced into ``AtlasCell.extra``
    max_num_seqs: int | None = None
    patched_vllm_enabled: bool | None = None
    speculative_num_tokens: int | None = None
    # Serving-variant knobs (2026-06-07): promoted to the typed AtlasCell fields so variant_key
    # distinguishes async / prefix-caching (MTP-K + max_num_seqs already carried above).
    async_scheduling: bool | None = None
    enable_prefix_caching: bool | None = None
    bench_backend: str = ""
    variant_extra: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # Analysis-carry-through (v1.42.0). Bundle-level, same for every concurrency.
    cache_mode: str = "unknown"  # warm | cold | unknown (declared label)
    prefix_cache_hit_rate: float | None = None  # 0..1, from bundle meta if present

    # Full-context descriptor (2026-06-07; AGENTS.md "Every performance number carries its
    # full context"). Bundle-level invariants; the methodology gate flags any left "unknown"
    # /None on a measured row under --strict.
    dataset: str = "unknown"
    cudagraph_mode: str = "unknown"
    gpu_memory_utilization: float | None = None
    kv_cache_dtype: str = "unknown"
    image: str = "unknown"
    data_parallel: int = 1
    pipeline_parallel: int = 1
    # Delivery ladder (2026-06-07; AGENTS.md "Experiment delivery ladder"): how the code
    # reached the cluster + the overlay sub-tier when delivery=overlay. Surfaced into extra.
    delivery: str = ""
    overlay_mode: str = ""
    patch_files: str = ""


def _identity_from_meta_and_overrides(
    bundle: Path,
    bundle_meta: dict[str, Any],
    overrides: dict[str, Any],
) -> _CellIdentity:
    """Merge bundle metadata + CLI overrides into a complete identity.

    Precedence: ``overrides`` (CLI) > ``bundle_meta`` (inference_perfbench_v1.json)
    > sensible defaults. Raises ``ValueError`` if a required field is still
    missing after the merge.
    """
    def pick(*keys: str, default: Any = None) -> Any:
        """Look up the first non-None match across overrides then bundle_meta.

        Each key may be passed multiple times to handle aliased field names
        between the CLI surface (``tensor_parallel``) and the bundle metadata
        schema (``tensor_parallel_size``).
        """
        for key in keys:
            if key in overrides and overrides[key] is not None:
                return overrides[key]
        for key in keys:
            if key in bundle_meta and bundle_meta[key] is not None:
                return bundle_meta[key]
        return default

    # Default cell_id = bundle directory name (matches operator convention).
    cell_id = overrides.get("cell_id") or bundle.name
    model = pick("model")
    if not model:
        raise ValueError(
            "import_perf_bench: --model is required (no model field in "
            f"{bundle}/inference_perfbench_v1.json and no --model override)"
        )

    # Hardware default = B200 (matches all current CW inference deploys).
    hw_field = pick("hardware", default="B200")
    # bundle_meta sometimes carries hardware as "B200 (single node, TP=8)"; strip parens.
    hardware = hw_field.split(" ")[0] if isinstance(hw_field, str) else hw_field

    quant = pick("quant") or _infer_quant_from_model_or_extra(model, bundle_meta)
    tensor_parallel = int(pick("tensor_parallel", "tensor_parallel_size", default=8))
    parallel_strategy = pick("parallel_strategy", default="TP")
    mtp_value = pick("mtp")
    if mtp_value is None:
        # Infer from speculative_decoding field (e.g. {"method":"mtp",...} or "mtp")
        spec = bundle_meta.get("speculative_decoding")
        if isinstance(spec, str) and "mtp" in spec.lower():
            mtp_value = True
        elif isinstance(spec, dict) and spec.get("method", "").endswith("mtp"):
            mtp_value = True
        else:
            mtp_value = False
    mbt = int(pick("max_num_batched_tokens", default=4096))

    cache_mode = pick("cache_mode", default="unknown")
    if cache_mode not in ("warm", "cold", "unknown"):
        cache_mode = "unknown"
    pchr = pick("prefix_cache_hit_rate", "gpu_prefix_cache_hit_rate")
    prefix_cache_hit_rate = float(pchr) if isinstance(pchr, (int, float)) else None

    # Full-context descriptor (2026-06-07). Picked from CLI overrides > bundle meta; left
    # "unknown"/None when neither supplies them (the methodology gate then flags it).
    dataset = pick("dataset", "workload_dataset", default="unknown") or "unknown"
    cudagraph_mode = pick("cudagraph_mode", default="unknown") or "unknown"
    if cudagraph_mode == "unknown" and pick("enforce_eager") is True:
        cudagraph_mode = "eager"
    gmu = pick("gpu_memory_utilization", "gpu_memory_util")
    gpu_memory_utilization = float(gmu) if isinstance(gmu, (int, float)) else None
    kv_cache_dtype = pick("kv_cache_dtype", default="unknown") or "unknown"
    image = pick("image", "vllm_version", "vllm_image", "vllm_commit", default="unknown") or "unknown"
    data_parallel = int(pick("data_parallel", "data_parallel_size", default=1) or 1)
    pipeline_parallel = int(pick("pipeline_parallel", "pipeline_parallel_size", default=1) or 1)
    _pf = pick("patch_files", default="")
    patch_files = ",".join(_pf) if isinstance(_pf, list) else str(_pf or "")

    return _CellIdentity(
        cell_id=cell_id,
        model=model,
        hardware=hardware,
        quant=quant,
        tensor_parallel=tensor_parallel,
        parallel_strategy=parallel_strategy,
        mtp=bool(mtp_value),
        max_num_batched_tokens=mbt,
        max_num_seqs=pick("max_num_seqs"),
        patched_vllm_enabled=pick("patched_vllm_enabled"),
        speculative_num_tokens=_infer_speculative_num_tokens(bundle_meta),
        async_scheduling=pick("async_scheduling"),
        enable_prefix_caching=pick("enable_prefix_caching", "prefix_caching"),
        bench_backend=pick("bench_backend", "backend_client") or "",
        notes=overrides.get("notes") or "",
        cache_mode=cache_mode,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        dataset=dataset,
        cudagraph_mode=cudagraph_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        image=image,
        data_parallel=data_parallel,
        pipeline_parallel=pipeline_parallel,
        delivery=pick("delivery", default="") or "",
        overlay_mode=pick("overlay_mode", default="") or "",
        patch_files=patch_files,
    )


def _infer_quant_from_model_or_extra(model: str, meta: dict[str, Any]) -> str:
    """Best-effort quant inference. Falls back to NVFP4 for the CW canonical."""
    ml = model.lower()
    if "fp8" in ml:
        return "FP8"
    if "nvfp4" in ml or "fp4" in ml:
        return "NVFP4"
    if "bf16" in ml:
        return "BF16"
    # Check extra fields (defensively handle missing/None vllm_image)
    vllm_image = meta.get("vllm_image", "")
    if isinstance(vllm_image, str) and "fp4" in vllm_image.lower():
        return "NVFP4"
    return "NVFP4"  # workspace canonical


def _infer_speculative_num_tokens(meta: dict[str, Any]) -> int | None:
    """Pull num_speculative_tokens out of the various shapes that field may take."""
    spec = meta.get("speculative_decoding")
    if isinstance(spec, dict):
        n = spec.get("num_speculative_tokens")
        if isinstance(n, int):
            return n
    return None


def _row_from_metrics(
    identity: _CellIdentity,
    concurrency: int,
    metrics: dict[str, float | int],
    sweep_path: Path,
    k: int,
    captured_at: str,
) -> AtlasCell:
    """Build one AtlasCell row from parsed metrics."""
    # Throughput aggregations:
    req_per_s = metrics.get("req_per_s") or metrics.get("req_per_s_derived")
    tpot_ms = metrics.get("tpot_med_ms")
    output_tps_total = metrics.get("output_tps")

    total_tps_total = metrics.get("total_tps")

    output_tps_per_user = (1000.0 / tpot_ms) if tpot_ms else None
    output_tps_per_gpu = (
        (output_tps_total / identity.tensor_parallel) if output_tps_total else None
    )
    total_tps_per_gpu = (
        (total_tps_total / identity.tensor_parallel) if total_tps_total else None
    )
    ttft = metrics.get("ttft_med_ms")

    # Mean ISL/OSL (shape): total input/generated tokens / successful requests.
    n_reqs = metrics.get("n_reqs")
    total_in = metrics.get("total_input_tokens")
    total_gen = metrics.get("total_generated_tokens")
    mean_input_tokens = (total_in / n_reqs) if (total_in and n_reqs) else None
    mean_output_tokens = (total_gen / n_reqs) if (total_gen and n_reqs) else None

    # Typed num_speculative_tokens for THIS row (captured before the variant_extra loop below
    # rebinds the local `k`): the row's K when sweeping K>1, else the bundle identity, else
    # 1-if-MTP. This is what makes variant_key distinguish K=2 vs K=3.
    num_speculative_tokens = (
        k if (k and k > 1)
        else (identity.speculative_num_tokens
              if identity.speculative_num_tokens is not None
              else (1 if identity.mtp else None))
    )

    notes_parts = [f"imported from {sweep_path.parent.parent.name}"]
    if k != 1:
        notes_parts.append(f"K={k}")
    if identity.speculative_num_tokens and identity.speculative_num_tokens > 1:
        notes_parts.append(f"num_speculative_tokens={identity.speculative_num_tokens}")
    if identity.patched_vllm_enabled:
        notes_parts.append("patchedVllm.enabled=true")
    if identity.notes:
        notes_parts.append(identity.notes)

    extra: dict[str, Any] = {
        "imported_from_perf_bench": str(sweep_path.parent.parent),
        "ttft_median_ms_substituted_for_avg": True,
        "engine": identity.engine,
    }
    if k != 1:
        extra["spec_decode_k"] = k
    if identity.max_num_seqs is not None:
        extra["max_num_seqs"] = identity.max_num_seqs
    if identity.patched_vllm_enabled is not None:
        extra["patched_vllm_enabled"] = bool(identity.patched_vllm_enabled)
    if identity.delivery:
        extra["delivery"] = identity.delivery
    if identity.overlay_mode:
        extra["overlay_mode"] = identity.overlay_mode
    if identity.patch_files:
        extra["patch_files"] = identity.patch_files
    if identity.variant_extra:
        for k, v in identity.variant_extra.items():
            if k not in extra:
                extra[k] = v

    # cell_id-with-K-suffix: keep different K runs distinguishable in the atlas.
    cell_id = identity.cell_id if k == 1 else f"{identity.cell_id}-K{k}"

    # Loud warning when this row is marked STATUS_FULL but lacks the
    # ttft + request-throughput that the scatter plots need: such a row
    # silently produces a blank point. Name the missing field(s) + the fix
    # so the gap is impossible to miss in the import log.
    if ttft is None or req_per_s is None:
        missing = []
        if ttft is None:
            missing.append("ttft_avg_ms (from 'Median TTFT (ms)')")
        if req_per_s is None:
            missing.append("request_throughput_avg (from 'Request throughput (req/s)')")
        print(
            f"WARNING: imported cell {cell_id!r} (c={concurrency}) is STATUS_FULL but "
            f"NOT plot-ready -- missing {', '.join(missing)}. This row will not "
            f"produce a scatter point. How to fix: ensure the bench output for "
            f"{sweep_path} prints the missing line(s), then re-import + re-aggregate.",
            file=sys.stderr,
        )

    return AtlasCell(
        cell_id=cell_id,
        model=identity.model,
        hardware=identity.hardware,
        quant=identity.quant,
        tensor_parallel=identity.tensor_parallel,
        parallel_strategy=identity.parallel_strategy,
        mtp=identity.mtp,
        max_num_batched_tokens=identity.max_num_batched_tokens,
        num_speculative_tokens=num_speculative_tokens,
        async_scheduling=identity.async_scheduling,
        max_num_seqs=identity.max_num_seqs,
        enable_prefix_caching=identity.enable_prefix_caching,
        bench_backend=identity.bench_backend,
        concurrency=concurrency,
        status=STATUS_FULL,
        ttft_avg_ms=ttft,
        request_throughput_avg=req_per_s,
        output_tps_per_user=output_tps_per_user,
        output_tps_per_gpu=output_tps_per_gpu,
        total_tps_per_gpu=total_tps_per_gpu,
        tpot_median_ms=tpot_ms,
        itl_avg_ms=metrics.get("itl_med_ms"),
        mean_input_tokens=mean_input_tokens,
        mean_output_tokens=mean_output_tokens,
        prefix_cache_hit_rate=identity.prefix_cache_hit_rate,
        cache_mode=identity.cache_mode,
        dataset=identity.dataset,
        cudagraph_mode=identity.cudagraph_mode,
        gpu_memory_utilization=identity.gpu_memory_utilization,
        kv_cache_dtype=identity.kv_cache_dtype,
        image=identity.image,
        data_parallel=identity.data_parallel,
        pipeline_parallel=identity.pipeline_parallel,
        backend=BACKEND_SGLANG_SWEEP if identity.engine == "sglang" else BACKEND_VLLM_SWEEP,
        raw_path=str(sweep_path),
        captured_at=captured_at,
        notes=" | ".join(notes_parts),
        extra=extra,
    )


def _check_plot_ready(
    rows: list[AtlasCell], require_plot_ready: bool, *, context: str = ""
) -> None:
    """Fail EARLY (at import) when ``require_plot_ready`` and any STATUS_FULL row
    lacks the ttft + request-throughput a throughput scatter point needs.

    The strict ``report_render`` / ``publish_to_lake`` gate already refuses a
    campaign with 0 plot-ready points at the END; this raises at import so an
    incomplete capture (e.g. a grep-dropped bench) is caught before the
    render/publish cycle and can never silently build a strict throughput
    campaign. Off by default (back-compat); the CLI ``--require-plot-ready``
    flag turns it on (and should be set for any throughput-focus campaign).
    """
    if not require_plot_ready:
        return
    bad = [
        f"{r.cell_id} (c={r.concurrency})"
        for r in rows
        if r.status == STATUS_FULL
        and (r.ttft_avg_ms is None or r.request_throughput_avg is None)
    ]
    if bad:
        raise ValueError(
            "import_perf_bench --require-plot-ready: "
            f"{len(bad)} STATUS_FULL cell(s) are NOT plot-ready (missing "
            "'Median TTFT (ms)' and/or 'Request throughput (req/s)' in the bench "
            f"output): {', '.join(bad)}. A strict throughput campaign cannot be "
            "built from incomplete capture. How to fix: re-capture with the FULL "
            "`vllm bench serve` output (use profiling/capture-bench.sh; never "
            "grep-drop those lines), then re-import."
            + (f" [{context}]" if context else "")
        )


def import_perf_bench_bundle(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
    require_plot_ready: bool = False,
) -> ImportResult:
    """Convert one bundle into a ``cells/<cell-id>/normalized.json`` entry.

    Args:
        bundle: ``*-deploy/experiments/artifacts/inference-perf-bench/<bundle>/``
            path. Must contain ``raw/sweep-c<N>.txt`` files.
        campaign_dir: target campaign directory (must exist; created by
            ``perf_tune_report_campaign_init``).
        overrides: optional dict of identity-field overrides. Keys mirror the
            CLI flags (``cell_id``, ``model``, ``hardware``, ``quant``,
            ``tensor_parallel``, ``parallel_strategy``, ``mtp``,
            ``max_num_batched_tokens``, ``max_num_seqs``,
            ``patched_vllm_enabled``, ``notes``).
        dry_run: if True, parse + validate but do NOT write any files. Used by
            tests + the CLI's ``--dry-run`` flag.
        captured_at: ISO-8601 timestamp string to stamp into emitted rows.
            Defaults to ``datetime.utcnow().isoformat()``.

    Raises:
        ValueError: bundle does not exist; no sweep files found; required
            identity field missing after merging metadata + overrides.
    """
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"import_perf_bench: bundle does not exist: {bundle}")

    sweep_files = _enumerate_sweep_files(bundle)
    if not sweep_files:
        raise ValueError(
            f"import_perf_bench: no sweep-c*.txt files found in {bundle}/raw/"
        )

    bundle_meta = _load_bundle_metadata(bundle)
    identity = _identity_from_meta_and_overrides(
        bundle, bundle_meta, overrides or {}
    )

    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[AtlasCell] = []
    skipped: list[Path] = []
    for sf in sweep_files:
        metrics = _parse_metrics(sf.path)
        if metrics is None:
            skipped.append(sf.path)
            continue
        rows.append(
            _row_from_metrics(
                identity=identity,
                concurrency=sf.concurrency,
                metrics=metrics,
                sweep_path=sf.path,
                k=sf.k,
                captured_at=captured_at,
            )
        )

    if not rows:
        raise ValueError(
            f"import_perf_bench: no valid sweep results parsed from {bundle}/raw/ "
            f"({len(skipped)} files were unparseable)"
        )

    _check_plot_ready(rows, require_plot_ready, context=str(bundle))

    # Choose status: full if every sweep file produced a row; partial if some
    # skipped; failed only if we have no rows at all (already raised above).
    status = STATUS_FULL if not skipped else STATUS_PARTIAL

    # Determine the output cell_id (some K-sweeps emit multiple cell_ids;
    # use the base identity.cell_id for the directory name, the rows
    # themselves may have K-suffixed cell_ids).
    cell_dir = campaign_dir / "cells" / identity.cell_id

    concurrencies = sorted({r.concurrency for r in rows})
    k_values = sorted({sf.k for sf in sweep_files if sf.path not in skipped})

    # Zymtrace per-kernel ingestion. Declared-coverage contract: if the
    # bundle's capture_sources.json doesn't declare zymtrace, this returns
    # a "skipped" result and no kernels.json is written (correct for non-
    # zymtrace bundles). If it DOES declare zymtrace, a missing/empty/
    # malformed TSV raises ZymtraceTSVMissing or ZymtraceTSVMalformed and
    # aborts the whole bundle import -- the silent-degradation pattern
    # Phase 5 SOURCE.md fell into is not allowed back in.
    kernels_result = import_zymtrace_kernels(bundle, cell_dir, dry_run=dry_run)

    if dry_run:
        return ImportResult(
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
        f"- captured_at:   {captured_at}\n"
        f"- concurrencies: {concurrencies}\n"
        f"- k_values:      {k_values}\n"
        f"- row_count:     {len(rows)}\n"
        f"- status:        {status}\n"
        f"- skipped_files: {len(skipped)}\n"
        + kernels_line
    )

    return ImportResult(
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
