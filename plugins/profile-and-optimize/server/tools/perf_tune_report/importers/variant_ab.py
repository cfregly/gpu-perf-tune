"""Importer for the model-optimize variant-A/B layout (run-variant-ab.sh output).

Sibling to ``inference_perf_bench.py`` (single-cell vLLM sweep) and
``lws_summary.py`` (multi-variant summary.json). Added so
``perftunereport import_perf_bench --bundle <ab-dir>`` ingests the
``run-variant-ab.sh`` A/B output DIRECTLY -- no per-campaign glue.

Source layout (one subdir per A/B arm == one cell):

    <bundle>/<arm>/c<C>-t<T>.txt   # raw `vllm bench serve` text, per (concurrency, trial)
    <bundle>/<arm>/result.json     # {"arm","tp","isl","osl","warm","by_c":{...}}  (optional)
    <bundle>/<arm>.json            # copy of result.json (optional)

Each arm has >=1 trial per concurrency. This importer parses the raw per-trial
text with the inference_perf_bench parser (so it gets Median TTFT + Request
throughput + Output/Total token throughput + Median TPOT -- i.e. PLOT-READY
rows), then AVERAGES the metrics across trials per concurrency, and emits one
``cells/<arm>/normalized.json`` per arm. ``atlas_aggregate`` then picks them up.

Identity (model / hardware / quant / max_num_batched_tokens) comes from CLI
overrides -- the arm ``result.json`` carries only ``tp`` / ``isl`` / ``osl`` /
``warm`` (per-token math is identical across arms; only the lever differs).
``mtp`` is inferred from the arm name when not overridden (``*-mtp*`` -> True).
"""

from __future__ import annotations

import dataclasses
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import (
    BACKEND_SGLANG_SWEEP,
    BACKEND_VLLM_SWEEP,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)
from tools.perf_tune_report.importers.inference_perf_bench import (
    _CellIdentity,
    _check_plot_ready,
    _parse_metrics,
    _row_from_metrics,
)
from tools.perf_tune_report.importers.zymtrace_kernels import import_zymtrace_kernels

# A raw per-trial file: ``c<concurrency>-t<trial>.txt``.
_TRIAL_RE = re.compile(r"^c(\d+)-t(\d+)\.txt$")
_VARIANT_EXTRA_KEYS = (
    "arm",
    "attention_backend",
    "cuda_graph_max_bs",
    "engine",
    "env",
    "extra_args",
    "extra_env",
    "fimoe",
    "flags",
    "max_num_seqs",
    "mem_fraction_static",
    "prefix_cache",
    "router",
    "sglang_mem_fraction_static",
    "topology",
)


def _arm_dirs(bundle: Path) -> list[Path]:
    """Immediate subdirs that contain at least one ``c<C>-t<T>.txt`` file."""
    out: list[Path] = []
    for d in sorted(bundle.iterdir()):
        if not d.is_dir():
            continue
        try:
            if any(_TRIAL_RE.match(f.name) for f in d.iterdir() if f.is_file()):
                out.append(d)
        except OSError:
            continue
    return out


def detect_variant_ab(bundle: Path) -> bool:
    """True iff ``bundle`` has the run-variant-ab.sh ``<arm>/c<C>-t<T>.txt`` layout."""
    try:
        bundle = bundle.expanduser()
        return bundle.is_dir() and len(_arm_dirs(bundle)) > 0
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class VariantAbImportResult:
    """Multi-cell import summary. Duck-types the single-cell ImportResult fields
    the CLI prints (``cell_id`` / ``cell_dir`` / ``normalized_path`` /
    ``k_values``) via properties, so the existing import_perf_bench CLI handler
    renders it without modification."""

    campaign_dir: Path
    bundle_path: Path
    cells: list[str]
    row_count: int
    concurrencies: list[int]
    status: str
    importer: str = "variant_ab"

    @property
    def cell_id(self) -> str:
        return f"{self.bundle_path.name} [{len(self.cells)} arms: {','.join(self.cells)}]"

    @property
    def cell_dir(self) -> Path:
        return self.campaign_dir / "cells"

    @property
    def normalized_path(self) -> Path:
        return self.campaign_dir / "cells"

    @property
    def k_values(self) -> list[int]:
        return [1]

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["campaign_dir"] = str(self.campaign_dir)
        d["bundle_path"] = str(self.bundle_path)
        return d


def _avg_metrics(trial_files: list[Path]) -> dict[str, float | int] | None:
    """Parse each trial file and average each metric across the valid trials."""
    parsed = [m for f in sorted(trial_files) if (m := _parse_metrics(f)) is not None]
    if not parsed:
        return None
    keys = set().union(*(d.keys() for d in parsed))
    avg: dict[str, float | int] = {}
    for k in keys:
        vals = [d[k] for d in parsed if k in d]
        if vals:
            avg[k] = statistics.mean(vals)
    return avg


def _arm_identity(
    arm: Path, overrides: dict[str, Any]
) -> _CellIdentity:
    res: dict[str, Any] = {}
    rj = arm / "result.json"
    if rj.is_file():
        try:
            res = json.loads(rj.read_text())
        except json.JSONDecodeError:
            res = {}
    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in overrides and overrides[key] is not None:
                return overrides[key]
        for key in keys:
            if key in res and res[key] is not None:
                return res[key]
        return default

    model = pick("model")
    if not model:
        raise ValueError(
            "import_perf_bench (variant_ab): --model is required "
            "(arm result.json carries no model)"
        )
    hw = pick("hardware", default="B200")
    hardware = hw.split(" ")[0] if isinstance(hw, str) else hw
    tp = int(pick("tensor_parallel", "tp", default=8))
    warm = res.get("warm")
    cache_mode = (
        "warm" if warm is True
        else "cold" if warm is False
        else (pick("cache_mode", default="unknown") or "unknown")
    )
    if cache_mode not in ("warm", "cold", "unknown"):
        cache_mode = "unknown"
    mtp = (
        bool(overrides["mtp"]) if overrides.get("mtp") is not None
        else bool(res["mtp"]) if res.get("mtp") is not None
        else "mtp" in arm.name.lower()
    )
    # Engine (cross-engine A/B): result.json "engine" (written by run-variant-ab.sh)
    # wins; else CLI override; else infer from the arm name (the "-s-" / "sgl" naming
    # convention); else default vllm.
    engine = (
        res.get("engine")
        or overrides.get("engine")
        or ("sglang" if ("sgl" in arm.name.lower() or "-s-" in arm.name.lower()) else "vllm")
    )
    engine = engine if engine in ("vllm", "sglang") else "vllm"
    cudagraph_mode = pick("cudagraph_mode", default="unknown") or "unknown"
    if cudagraph_mode == "unknown" and pick("enforce_eager") is True:
        cudagraph_mode = "eager"
    gmu = pick("gpu_memory_utilization", "gpu_memory_util")
    pchr = pick("prefix_cache_hit_rate", "gpu_prefix_cache_hit_rate")
    variant_extra = {
        k: res[k]
        for k in _VARIANT_EXTRA_KEYS
        if k in res and res[k] not in (None, "", "unknown")
    }
    extra_block = res.get("extra")
    if isinstance(extra_block, dict):
        for k, v in extra_block.items():
            if v not in (None, "", "unknown"):
                variant_extra.setdefault(k, v)
    return _CellIdentity(
        cell_id=arm.name,
        model=model,
        hardware=hardware,
        quant=pick("quant", default="NVFP4"),
        tensor_parallel=tp,
        parallel_strategy=pick("parallel_strategy", default="TP"),
        mtp=mtp,
        max_num_batched_tokens=int(pick("max_num_batched_tokens", default=4096)),
        engine=engine,
        max_num_seqs=pick("max_num_seqs"),
        patched_vllm_enabled=pick("patched_vllm_enabled"),
        speculative_num_tokens=pick("num_speculative_tokens", "speculative_num_tokens"),
        variant_extra=variant_extra,
        cache_mode=cache_mode,
        prefix_cache_hit_rate=float(pchr) if isinstance(pchr, (int, float)) else None,
        dataset=pick("dataset", "workload_dataset", default="unknown") or "unknown",
        cudagraph_mode=cudagraph_mode,
        gpu_memory_utilization=float(gmu) if isinstance(gmu, (int, float)) else None,
        kv_cache_dtype=pick("kv_cache_dtype", default="unknown") or "unknown",
        image=pick("image", "vllm_image", "sglang_image", "vllm_commit", default="unknown") or "unknown",
        data_parallel=int(pick("data_parallel", "data_parallel_size", default=1) or 1),
        pipeline_parallel=int(pick("pipeline_parallel", "pipeline_parallel_size", default=1) or 1),
        delivery=pick("delivery", default="") or "",
        overlay_mode=pick("overlay_mode", default="") or "",
        patch_files=(lambda x: ",".join(x) if isinstance(x, list) else str(x or ""))(pick("patch_files", default="")),
        notes=overrides.get("notes") or "",
    )


def import_variant_ab_bundle(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
    require_plot_ready: bool = False,
) -> VariantAbImportResult:
    """Convert a run-variant-ab.sh A/B bundle into one ``cells/<arm>/normalized.json``
    per arm. Reuses the inference_perf_bench text parser + row builder.
    """
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"import_perf_bench (variant_ab): bundle does not exist: {bundle}")
    arms = _arm_dirs(bundle)
    if not arms:
        raise ValueError(
            f"import_perf_bench (variant_ab): no <arm>/c<C>-t<T>.txt dirs in {bundle}"
        )
    overrides = overrides or {}
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cells: list[str] = []
    total_rows = 0
    all_concs: set[int] = set()
    any_skipped = False

    for arm in arms:
        identity = _arm_identity(arm, overrides)
        by_c: dict[int, list[Path]] = {}
        for f in arm.iterdir():
            m = _TRIAL_RE.match(f.name)
            if m and f.is_file():
                by_c.setdefault(int(m.group(1)), []).append(f)
        rows: list[AtlasCell] = []
        for c in sorted(by_c):
            metrics = _avg_metrics(by_c[c])
            if metrics is None:
                any_skipped = True
                continue
            rows.append(
                _row_from_metrics(
                    identity=identity,
                    concurrency=c,
                    metrics=metrics,
                    sweep_path=sorted(by_c[c])[0],
                    k=1,
                    captured_at=captured_at,
                )
            )
        if not rows:
            any_skipped = True
            continue
        _check_plot_ready(rows, require_plot_ready, context=f"{bundle} arm={arm.name}")
        cells.append(arm.name)
        total_rows += len(rows)
        all_concs.update(r.concurrency for r in rows)
        cell_dir = campaign_dir / "cells" / arm.name
        # Per-arm zymtrace SoL ingestion (declared-coverage contract, mirrors
        # inference_perf_bench): when run-variant-ab.sh's inline SoL capture wrote
        # <arm>/capture_sources.json + <arm>/zymtrace/, a cells/<arm>/kernels.json is
        # emitted (renderer page 4 / L1). If the manifest DECLARES zymtrace but a TSV
        # is missing/empty/malformed, this raises and aborts the whole import -- the
        # silent-degradation pattern is not allowed back in. No manifest -> correct
        # no-op (single-engine / pre-SoL bundles still import cleanly). Validated even
        # on dry_run (parse-only; no write).
        kernels_result = import_zymtrace_kernels(arm, cell_dir, dry_run=dry_run)
        if dry_run:
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "normalized.json").write_text(
            json.dumps([r.to_dict() for r in rows], indent=2, sort_keys=True) + "\n"
        )
        (cell_dir / "status.txt").write_text(STATUS_FULL + "\n")
        backend = BACKEND_SGLANG_SWEEP if identity.engine == "sglang" else BACKEND_VLLM_SWEEP
        (cell_dir / "backend.txt").write_text(backend + "\n")
        kernels_line = (
            f"- kernels.json:  {kernels_result.kernels_json_path}\n"
            if kernels_result.kernels_json_path is not None
            else f"- kernels.json:  (skipped: {kernels_result.skipped_reason})\n"
        )
        (cell_dir / "SOURCE.md").write_text(
            f"# {arm.name}\n\n"
            f"- imported_from: {bundle} (variant_ab; trial-averaged)\n"
            f"- engine:        {identity.engine} (backend={backend})\n"
            f"- captured_at:   {captured_at}\n"
            f"- concurrencies: {sorted(by_c)}\n"
            f"- trials/conc:   {{c: len(files)}} averaged\n"
            f"- row_count:     {len(rows)}\n"
            + kernels_line
        )

    if not cells:
        raise ValueError(
            f"import_perf_bench (variant_ab): no arm produced rows in {bundle}"
        )
    return VariantAbImportResult(
        campaign_dir=campaign_dir,
        bundle_path=bundle,
        cells=cells,
        row_count=total_rows,
        concurrencies=sorted(all_concs),
        status=STATUS_PARTIAL if any_skipped else STATUS_FULL,
    )
