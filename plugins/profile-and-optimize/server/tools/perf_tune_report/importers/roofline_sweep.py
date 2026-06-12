"""Importer for the always-on prefill+decode roofline sweep bundle.

Ingests the output of ``perftune-specdec/profiling/roofline-sweep.sh``:

    <bundle>/decode_sweep.jsonl    # one cell.py JSON line per decode concurrency
    <bundle>/prefill_sweep.jsonl   # one cell.py JSON line per prefill ISL
    <bundle>/roofline_sweep_manifest.json  # optional metadata

Each JSONL line (emitted by ``profiling/roofline/cell.py``) carries MEASURED
bench metrics (``bench``) + in-pod DCGM PROF active fractions (``dcgm_steady``:
``sm_active_mean`` / ``tensor_active_mean`` / ``dram_active_mean`` /
``nvlink_tx_Bps_mean`` ...). This is the per-(phase, concurrency/ISL) DCGM
utilization that the workload-level dcgm_correlate path cannot produce.

Output (per the perf_tune_report cell convention):

- ``cells/<cell_id>-decode/normalized.json`` + ``-prefill/normalized.json`` --
  ``AtlasCell`` rows; the DCGM utilization rides in ``extra["dcgm_util"]`` so it
  flows to ``atlas_v1.extra_json`` in the lake with no schema change.
- ``cells/<cell_id>-decode/roofline_sweep.json`` -- the phase-tagged operating
  points the new ``renderer/prefill_decode_roofline.py`` page consumes.

The roofline math (arithmetic-intensity x-axis, achieved-FLOPS y-axis, the
NVFP4/HBM ceilings) is applied in the renderer from these measured inputs +
``sol-ceilings.yaml`` -- this importer only normalizes + persists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report import roofline_math
from tools.perf_tune_report.schema import (
    BACKEND_VLLM_SWEEP,
    STATUS_FULL,
    AtlasCell,
)

DECODE_FILE = "decode_sweep.jsonl"
PREFILL_FILE = "prefill_sweep.jsonl"
MANIFEST_FILE = "roofline_sweep_manifest.json"
# A model config.json captured in-pod by roofline-sweep.sh (so the analytical
# roofline math is self-contained without a registry hit at render time).
MODEL_CONFIG_FILE = "model_config.json"


def _resolve_shape(
    bundle: Path, identity: dict[str, Any], overrides: dict[str, Any]
) -> "roofline_math.ModelShape | None":
    """Resolve the analytical ModelShape used to embed FLOP/byte + byte-grounded
    fields. Priority: explicit --model-config path > a model_config.json captured
    in the bundle > the built-in registry (by served model name)."""
    cfg_path = overrides.get("model_config_path")
    candidates = []
    if cfg_path:
        candidates.append(Path(cfg_path).expanduser())
    candidates.append(bundle / MODEL_CONFIG_FILE)
    for p in candidates:
        try:
            if p.is_file():
                cfg = json.loads(p.read_text())
                return roofline_math.from_hf_config(cfg, name=identity["model"])
        except (json.JSONDecodeError, OSError):
            continue
    return roofline_math.shape_for_model(identity["model"])


@dataclass(frozen=True)
class RooflineSweepImportResult:
    campaign_dir: Path
    cell_id: str
    bundle_path: Path
    decode_points: int
    prefill_points: int
    cell_dirs: list[str]
    status: str
    # decode cells whose num_prompts < 2*c -> steady-state window too short ->
    # output_throughput undercounts high-concurrency throughput (see
    # _steady_window_warnings + docs/METHODOLOGY.md trap 4). Empty = clean.
    steady_window_warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["campaign_dir"] = str(self.campaign_dir)
        d["bundle_path"] = str(self.bundle_path)
        d["cell_dirs"] = [str(x) for x in self.cell_dirs]
        return d


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _dcgm_util(cell: dict[str, Any]) -> dict[str, float | int | None]:
    g = cell.get("dcgm_steady", {}) or {}
    return {
        "sm_active": g.get("sm_active_mean"),
        "tensor_active": g.get("tensor_active_mean"),
        "dram_active": g.get("dram_active_mean"),
        "fp16_active": g.get("fp16_active_mean"),
        "nvlink_tx_Bps": g.get("nvlink_tx_Bps_mean"),
        "nvlink_rx_Bps": g.get("nvlink_rx_Bps_mean"),
        "dmon_samples": g.get("dmon_samples"),
    }


def _point_analytics(
    cell: dict[str, Any],
    phase: str,
    shape: "roofline_math.ModelShape | None",
    quant: str,
    tp: int,
    kv_dtype: str,
) -> dict[str, float | None]:
    """Analytical roofline coordinates for one operating point (None when the
    model shape is unavailable -> renderer falls back to the DCGM proxy).

    Ceiling-free by design: stores the arithmetic intensity (pure analytical),
    the achieved compute / GPU (analytical FLOP/token x MEASURED tok/s), and the
    delivered HBM bytes/s / GPU. The %-of-peak is computed downstream where the
    sol-ceilings.yaml peaks are loaded (renderer + lake_writer)."""
    if shape is None:
        return {"arithmetic_intensity": None, "achieved_tflops_per_gpu": None,
                "hbm_delivered_Bps_per_gpu": None}
    b = cell.get("bench", {}) or {}
    if phase == "decode":
        c = cell.get("c") or 1
        rate = b.get("output_throughput")
        if not rate:
            return {"arithmetic_intensity": None, "achieved_tflops_per_gpu": None,
                    "hbm_delivered_Bps_per_gpu": None}
        ctx = int((cell.get("isl") or 256) + (cell.get("osl") or 512) // 2)
        ai = shape.decode_arithmetic_intensity(c, ctx, quant, kv_dtype)
        union = (min(shape.n_routed_experts, shape.n_experts_per_tok * c)
                 if shape.is_moe else 0)
        bytes_per_tok = shape.active_weight_bytes(union, quant) / c \
            + shape.kv_bytes_per_token(ctx, kv_dtype)
    else:  # prefill
        isl = cell.get("isl")
        inp, dur = b.get("total_input_tokens"), b.get("duration")
        rate = (inp / dur) if (inp and dur) else None
        if not rate or not isl:
            return {"arithmetic_intensity": None, "achieved_tflops_per_gpu": None,
                    "hbm_delivered_Bps_per_gpu": None}
        ai = shape.prefill_arithmetic_intensity(isl, quant)
        experts = shape.n_routed_experts if shape.is_moe else 0
        bytes_per_tok = shape.active_weight_bytes(experts, quant) / max(int(isl), 1)
    return {
        "arithmetic_intensity": ai,
        "achieved_tflops_per_gpu": shape.flop_per_token * rate / tp / 1e12,
        "hbm_delivered_Bps_per_gpu": bytes_per_tok * rate / tp,
    }


def _atlas_row(
    cell: dict[str, Any],
    phase: str,
    cell_id: str,
    identity: dict[str, Any],
    captured_at: str,
    raw_path: str,
    shape: "roofline_math.ModelShape | None" = None,
) -> AtlasCell | None:
    b = cell.get("bench", {}) or {}
    if not b:
        return None
    tp = int(identity["tensor_parallel"])
    tpot = b.get("median_tpot_ms")
    out_tps = b.get("output_throughput")
    total_tps = b.get("total_token_throughput")
    util = _dcgm_util(cell)
    analytics = _point_analytics(cell, phase, shape, identity["quant"], tp,
                                 identity.get("kv_dtype", "fp8"))
    _c = cell.get("c") or 0
    _np = cell.get("num_prompts") or 0
    # decode steady-window guard: num_prompts must be >= 2*c so the bench's
    # output_throughput (= tokens / full duration) reflects steady state, not
    # the ramp+drain window. num=c+4 undercounts high-c throughput ~1.6-1.8x
    # (docs/METHODOLOGY.md trap 4). Flagged here; surfaced at import.
    _undercount = bool(phase == "decode" and _c >= 16 and _np and _np < 2 * _c)
    extra = {
        "phase": phase,
        "isl": cell.get("isl"),
        "osl": cell.get("osl"),
        "num_prompts": cell.get("num_prompts"),
        "seed": cell.get("seed"),
        "steady_window_undercount": _undercount,
        "dcgm_util": util,
        "dcgm_source": "in-pod dcgmi PROF (active fractions 0..1)",
        "roofline": analytics,
    }
    for _k in ("delivery", "overlay_mode", "patch_files"):
        _v = identity.get(_k)
        if _v:
            extra[_k] = _v
    return AtlasCell(
        cell_id=cell_id,
        model=identity["model"],
        hardware=identity["hardware"],
        quant=identity["quant"],
        tensor_parallel=tp,
        parallel_strategy=identity["parallel_strategy"],
        mtp=bool(identity["mtp"]),
        max_num_batched_tokens=int(identity["max_num_batched_tokens"]),
        concurrency=int(cell.get("c") or 0),
        status=STATUS_FULL,
        ttft_avg_ms=b.get("median_ttft_ms"),
        request_throughput_avg=b.get("request_throughput"),
        output_tps_per_user=(1000.0 / tpot) if tpot else None,
        output_tps_per_gpu=(out_tps / tp) if out_tps else None,
        total_tps_per_gpu=(total_tps / tp) if total_tps else None,
        tpot_median_ms=tpot,
        itl_avg_ms=b.get("median_itl_ms"),
        mean_input_tokens=float(cell["isl"]) if cell.get("isl") is not None else None,
        mean_output_tokens=float(cell["osl"]) if cell.get("osl") is not None else None,
        cache_mode=identity.get("cache_mode", "unknown"),
        dataset=identity.get("dataset", "random"),
        cudagraph_mode=identity.get("cudagraph_mode", "unknown"),
        gpu_memory_utilization=identity.get("gpu_memory_utilization"),
        kv_cache_dtype=identity.get("kv_dtype", "unknown"),
        image=identity.get("image", "unknown"),
        data_parallel=int(identity.get("data_parallel", 1) or 1),
        pipeline_parallel=int(identity.get("pipeline_parallel", 1) or 1),
        backend=BACKEND_VLLM_SWEEP,
        raw_path=raw_path,
        captured_at=captured_at,
        notes=f"roofline sweep phase={phase}",
        extra=extra,
    )


def _roofline_artifact(
    decode: list[dict[str, Any]],
    prefill: list[dict[str, Any]],
    identity: dict[str, Any],
    shape: "roofline_math.ModelShape | None" = None,
) -> dict[str, Any]:
    """The phase-tagged operating points the renderer page consumes. Carries the
    embedded ``analytical_shape`` (so the renderer + lake are self-contained,
    independent of the registry) and per-point analytical coordinates."""
    tp = int(identity["tensor_parallel"])
    quant = identity["quant"]
    kvd = identity.get("kv_dtype", "fp8")

    def point(cell: dict[str, Any], phase: str) -> dict[str, Any]:
        b = cell.get("bench", {}) or {}
        u = _dcgm_util(cell)
        a = _point_analytics(cell, phase, shape, quant, tp, kvd)
        return {
            "phase": phase,
            "c": cell.get("c"),
            "isl": cell.get("isl"),
            "osl": cell.get("osl"),
            "median_tpot_ms": b.get("median_tpot_ms"),
            "median_ttft_ms": b.get("median_ttft_ms"),
            "output_throughput": b.get("output_throughput"),
            "total_input_tokens": b.get("total_input_tokens"),
            "duration": b.get("duration"),
            "sm_active": u["sm_active"],
            "tensor_active": u["tensor_active"],
            "dram_active": u["dram_active"],
            "nvlink_tx_Bps": u["nvlink_tx_Bps"],
            # analytical (ceiling-free) roofline coordinates
            "arithmetic_intensity": a["arithmetic_intensity"],
            "achieved_tflops_per_gpu": a["achieved_tflops_per_gpu"],
            "hbm_delivered_Bps_per_gpu": a["hbm_delivered_Bps_per_gpu"],
        }
    art: dict[str, Any] = {
        "schema": "roofline_sweep_points_v1",
        "hardware": identity["hardware"],
        "tensor_parallel": tp,
        "quant": quant,
        "kv_dtype": kvd,
        "model": identity["model"],
        "decode": [point(c, "decode") for c in decode],
        "prefill": [point(c, "prefill") for c in prefill],
    }
    if shape is not None:
        art["analytical_shape"] = roofline_math.shape_to_dict(shape)
        art["analytical_summary"] = shape.to_summary(quant, kvd)
    return art


def _resolve_identity(bundle: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    meta = {}
    mf = bundle / MANIFEST_FILE
    if mf.is_file():
        try:
            meta = json.loads(mf.read_text())
        except json.JSONDecodeError:
            meta = {}

    def pick(*keys, default=None):
        for k in keys:
            if overrides.get(k) is not None:
                return overrides[k]
        for k in keys:
            if meta.get(k) is not None:
                return meta[k]
        return default

    model = pick("model", default="zai-org/GLM-5.1")
    hw = pick("hardware", default="GB300")
    hardware = hw.split(" ")[0] if isinstance(hw, str) else hw
    quant = pick("quant", default="NVFP4")
    # Full-context descriptors (2026-06-07) so roofline cells can pass
    # publish_to_lake --strict. dataset defaults to "random" (roofline-sweep.sh's
    # cell.py always drives --dataset-name random); cudagraph_mode follows the
    # eager flag like inference_perf_bench; the rest come from override/manifest.
    cudagraph_mode = pick("cudagraph_mode", default="unknown")
    if cudagraph_mode == "unknown" and pick("enforce_eager") is True:
        cudagraph_mode = "eager"
    gmu = pick("gpu_memory_utilization")
    return {
        "cell_id": overrides.get("cell_id") or bundle.name,
        "model": model,
        "hardware": hardware,
        "quant": quant,
        "kv_dtype": pick("kv_dtype", "kv_cache_dtype", default="fp8"),
        "tensor_parallel": int(pick("tensor_parallel", default=4)),
        "parallel_strategy": pick("parallel_strategy", default="TP"),
        "mtp": bool(pick("mtp", default=False)),
        "max_num_batched_tokens": int(pick("max_num_batched_tokens", default=12288)),
        "cache_mode": pick("cache_mode", default="unknown"),
        "dataset": pick("dataset", default="random"),
        "cudagraph_mode": cudagraph_mode,
        "gpu_memory_utilization": float(gmu) if isinstance(gmu, (int, float)) else None,
        "image": pick("image", default="unknown"),
        "delivery": pick("delivery", default=""),
        "overlay_mode": pick("overlay_mode", default=""),
        "patch_files": pick("patch_files", default=""),
        "data_parallel": int(pick("data_parallel", "data_parallel_size", default=1) or 1),
        "pipeline_parallel": int(pick("pipeline_parallel", "pipeline_parallel_size", default=1) or 1),
    }


def _steady_window_warnings(decode: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode cells whose num_prompts < 2*c -> steady-state window too short ->
    output_throughput undercounts high-concurrency throughput ~1.6-1.8x. This is
    the runtime guard for docs/METHODOLOGY.md trap 4 (the 2026-06-07
    roofline-sweep.sh num=c+4 bug). Caught for ANY driver, so a future
    undercounted sweep is surfaced at import, never silently published."""
    warns: list[dict[str, Any]] = []
    for cell in decode:
        c = cell.get("c") or 0
        npr = cell.get("num_prompts") or 0
        if c >= 16 and npr and npr < 2 * c:
            warns.append({"c": c, "num_prompts": npr, "min_required": 2 * c,
                          "tag": cell.get("tag")})
    return warns


def import_roofline_sweep_bundle(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
) -> RooflineSweepImportResult:
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"import_roofline_sweep: bundle does not exist: {bundle}")
    decode = _read_jsonl(bundle / DECODE_FILE)
    prefill = _read_jsonl(bundle / PREFILL_FILE)
    steady_warns = _steady_window_warnings(decode)
    if steady_warns:
        import sys as _sys
        cells = ", ".join(f"c={w['c']}(num={w['num_prompts']}<{w['min_required']})"
                          for w in steady_warns)
        print(
            f"WARN: steady-window undercount in {len(steady_warns)} decode cell(s) "
            f"[{cells}] -- num_prompts < 2*c, so output_throughput is ramp/drain-"
            f"dominated and UNDERCOUNTS high-concurrency throughput ~1.6-1.8x "
            f"(docs/METHODOLOGY.md trap 4). Re-run with num_prompts >= 2*c "
            f"(roofline-sweep.sh now defaults this).",
            file=_sys.stderr,
        )
    if not decode and not prefill:
        raise ValueError(
            f"import_roofline_sweep: no points in {bundle}/{DECODE_FILE} or {PREFILL_FILE}"
        )
    overrides = overrides or {}
    identity = _resolve_identity(bundle, overrides)
    shape = _resolve_shape(bundle, identity, overrides)
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    base = identity["cell_id"]
    cell_dirs: list[str] = []
    plans = [("decode", decode), ("prefill", prefill)]
    for phase, cells in plans:
        if not cells:
            continue
        rows = [
            r for r in (
                _atlas_row(c, phase, f"{base}-{phase}", identity, captured_at,
                           str(bundle / f"{phase}_sweep.jsonl"), shape)
                for c in cells
            ) if r is not None
        ]
        if not rows:
            continue
        cell_dir = campaign_dir / "cells" / f"{base}-{phase}"
        cell_dirs.append(str(cell_dir))
        if dry_run:
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "normalized.json").write_text(
            json.dumps([r.to_dict() for r in rows], indent=2, sort_keys=True)
        )
        (cell_dir / "status.txt").write_text(STATUS_FULL + "\n")
        (cell_dir / "backend.txt").write_text(BACKEND_VLLM_SWEEP + "\n")
        # the renderer-page input artifact (written on the decode cell, carrying both phases)
        if phase == "decode" or not decode:
            (cell_dir / "roofline_sweep.json").write_text(
                json.dumps(_roofline_artifact(decode, prefill, identity, shape), indent=2)
            )

    return RooflineSweepImportResult(
        campaign_dir=campaign_dir,
        cell_id=base,
        bundle_path=bundle,
        decode_points=len(decode),
        prefill_points=len(prefill),
        cell_dirs=cell_dirs,
        status=STATUS_FULL,
        steady_window_warnings=steady_warns,
    )
