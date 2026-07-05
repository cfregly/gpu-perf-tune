"""Bundle importer for ``perf-report`` campaigns built from a multi-variant
``summary.json`` (the GLM-5.1 LWS-baseline-vs-champions layout).

Sibling to ``inference_perf_bench.py`` and ``inference_drive_load.py``.
Added in profile-and-optimize v1.23.1.

Source format
-------------

Some perf-report campaigns are produced by an upstream variant-runner
(e.g. the GLM-LWS comparison) that emits a single multi-variant
``summary.json`` at the campaign root, not a per-cell bundle:

.. code-block:: json

    {
      "variants": [
        {
          "short": "LWS-baseline",
          "label": "LWS-baseline (mns=40, mbt=32768, kv=fp8)",
          "knob": "mns=40, mbt=32768, kv=fp8",
          "metrics_per_concurrency": [
            {
              "path": ".../sweep-c1.txt",
              "concurrency": 1,
              "duration_s": 30.5,
              "req_per_s": 0.14,
              "output_tps": 71.43,
              "ttft_median_ms": 198.58,
              "tpot_median_ms": 13.64
            },
            ...
          ]
        },
        ...
      ]
    }

This importer reads that ``summary.json`` from the campaign root and
emits one ``AtlasCell`` per ``(variant, concurrency)`` point directly
into ``<campaign_dir>/atlas.jsonl``. It bypasses the
``cells/<id>/normalized.json -> atlas_aggregate`` pipeline because
the summary.json is already pre-aggregated; running each variant
through the cell-level pipeline would duplicate the aggregation for
no gain.

Operator-facing CLI dispatch lives in ``import_bundle_auto``; an
additional dispatch arm detects ``bundle/summary.json`` with the
``variants[*].metrics_per_concurrency[]`` shape and routes here.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import (
    BACKEND_VLLM_SWEEP,
    STATUS_FULL,
    AtlasCell,
)


_KNOB_MBT_RE = re.compile(r"mbt=(\d+)")


@dataclass(frozen=True)
class LwsSummaryImportResult:
    """Summary returned by ``import_lws_summary_bundle``.

    Same shape as the inference-perf-bench / inference_drive_load result
    classes for consistency in the CLI JSON envelope.
    """

    campaign_dir: Path
    summary_path: Path
    atlas_path: Path
    variant_count: int
    row_count: int
    concurrencies: list[int]
    status: str
    importer: str = "lws_summary"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["campaign_dir"] = str(self.campaign_dir)
        d["summary_path"] = str(self.summary_path)
        d["atlas_path"] = str(self.atlas_path)
        return d


def _parse_mbt(knob: str | None) -> int | None:
    """Pull max-num-batched-tokens out of a knob string.

    ``"mns=40, mbt=32768, kv=fp8"`` -> ``32768``. Returns ``None`` if
    no ``mbt=<N>`` token is present.
    """
    if not knob:
        return None
    m = _KNOB_MBT_RE.search(knob)
    return int(m.group(1)) if m else None


def detect_lws_summary(bundle: Path) -> bool:
    """Return True if ``bundle/summary.json`` matches the LWS shape.

    The shape check is liberal: any JSON file at ``bundle/summary.json``
    that has a non-empty ``variants[]`` list with at least one variant
    carrying ``metrics_per_concurrency[]`` qualifies.
    """
    summary_path = bundle / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        data = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        return False
    first = variants[0]
    if not isinstance(first, dict):
        return False
    metrics = first.get("metrics_per_concurrency")
    return isinstance(metrics, list)


def import_lws_summary_bundle(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
) -> LwsSummaryImportResult:
    """Read ``<bundle>/summary.json`` and write ``<campaign_dir>/atlas.jsonl``.

    Args:
        bundle: directory containing ``summary.json``. For the GLM-LWS
            layout this is identical to the campaign directory; the
            argument is named ``bundle`` for consistency with sibling
            importers.
        campaign_dir: target directory for ``atlas.jsonl``. Created if
            absent (operator may pass the same path as ``bundle``).
        overrides: optional ``AtlasCell`` field overrides applied to
            every emitted row. Recognised keys: ``model``, ``hardware``,
            ``quant``, ``tensor_parallel``, ``parallel_strategy``,
            ``mtp``, ``backend``, ``captured_at``, ``cell_id_prefix``.
            Unknown keys are ignored.
        dry_run: if True, parse + validate but do NOT write
            ``atlas.jsonl``.
        captured_at: optional UTC timestamp string to stamp on every
            emitted row's ``captured_at`` field. Falls back to the
            ``overrides["captured_at"]`` value or to ``""``.

    Returns:
        ``LwsSummaryImportResult`` with provenance + counts.

    Raises:
        ValueError: bundle absent, summary.json malformed, or no
            variants emit a non-zero row count.
    """
    bundle = bundle.expanduser().resolve()
    campaign_dir = campaign_dir.expanduser().resolve()
    overrides = overrides or {}

    if not bundle.is_dir():
        raise ValueError(
            f"import_lws_summary: bundle directory does not exist: {bundle}"
        )
    summary_path = bundle / "summary.json"
    if not summary_path.is_file():
        raise ValueError(
            f"import_lws_summary: summary.json not found at {summary_path}"
        )

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"import_lws_summary: summary.json malformed at {summary_path}: {e}"
        ) from e

    variants = summary.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(
            f"import_lws_summary: summary.json at {summary_path} has no variants[]"
        )

    # Resolve overrides + defaults.
    model = overrides.get("model", "zai-org/GLM-5.1")
    hardware = overrides.get("hardware", "B200")
    quant = overrides.get("quant", "NVFP4")
    tensor_parallel = int(overrides.get("tensor_parallel", 8))
    parallel_strategy = overrides.get("parallel_strategy", "TP")
    mtp = bool(overrides.get("mtp", False))
    backend = overrides.get("backend", BACKEND_VLLM_SWEEP)
    cell_id_prefix = overrides.get("cell_id_prefix", "")
    stamp = captured_at or overrides.get("captured_at", "")
    # Full-context descriptor (2026-06-09): apply the importer overrides so lws-summary
    # cells carry the same descriptor the perf-bench path does (the methodology_problems
    # --strict gate, CLAUDE.md 'Every performance number carries its full context').
    cache_mode = overrides.get("cache_mode", "unknown")
    dataset = overrides.get("dataset", "unknown")
    cudagraph_mode = overrides.get("cudagraph_mode", "unknown")
    kv_cache_dtype = overrides.get("kv_cache_dtype", "unknown")
    image = overrides.get("image", "unknown")
    gpu_memory_utilization = overrides.get("gpu_memory_utilization")

    rows: list[AtlasCell] = []
    concurrencies: set[int] = set()

    for variant in variants:
        if not isinstance(variant, dict):
            continue
        short = variant.get("short") or variant.get("label") or "variant"
        knob = variant.get("knob", "")
        label = variant.get("label", "")
        per_c = variant.get("metrics_per_concurrency") or []
        if not isinstance(per_c, list):
            continue

        cell_id = f"{cell_id_prefix}{short}" if cell_id_prefix else short
        mbt = _parse_mbt(knob) or 0
        # Per-variant dataset (multi-workload: each variant is a different dataset);
        # falls back to the global override when the variant does not name one.
        v_dataset = variant.get("dataset") or dataset

        for m in per_c:
            if not isinstance(m, dict):
                continue
            c_raw = m.get("concurrency", m.get("c"))
            try:
                c = int(c_raw)
            except (TypeError, ValueError):
                continue
            concurrencies.add(c)

            tpot_ms = m.get("tpot_median_ms")
            try:
                tpot_ms_f = float(tpot_ms) if tpot_ms is not None else 0.0
            except (TypeError, ValueError):
                tpot_ms_f = 0.0
            output_tps_per_user = (1000.0 / tpot_ms_f) if tpot_ms_f > 0 else 0.0

            ttft = m.get("ttft_median_ms") or m.get("ttft_p99_ms") or 0
            req_per_s = m.get("req_per_s") or 0
            output_tps = m.get("output_tps") or 0

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
                    concurrency=c,
                    status=STATUS_FULL,
                    ttft_avg_ms=float(ttft),
                    request_throughput_avg=float(req_per_s),
                    output_tps_per_user=output_tps_per_user,
                    output_tps_per_gpu=float(output_tps) / tensor_parallel,
                    cache_mode=cache_mode,
                    dataset=v_dataset,
                    cudagraph_mode=cudagraph_mode,
                    kv_cache_dtype=kv_cache_dtype,
                    image=image,
                    gpu_memory_utilization=gpu_memory_utilization,
                    mean_input_tokens=m.get("mean_input_tokens"),
                    mean_output_tokens=m.get("mean_output_tokens"),
                    backend=backend,
                    raw_path=str(m.get("path", "")),
                    captured_at=stamp,
                    notes=knob,
                    extra={"variant_label": label},
                )
            )

    if not rows:
        raise ValueError(
            f"import_lws_summary: no rows emitted from {summary_path} "
            f"(check that variants[*].metrics_per_concurrency[] is populated)"
        )

    atlas_path = campaign_dir / "atlas.jsonl"
    if not dry_run:
        campaign_dir.mkdir(parents=True, exist_ok=True)
        with atlas_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(dataclasses.asdict(row)) + "\n")

    return LwsSummaryImportResult(
        campaign_dir=campaign_dir,
        summary_path=summary_path,
        atlas_path=atlas_path,
        variant_count=len(variants),
        row_count=len(rows),
        concurrencies=sorted(concurrencies),
        status="full",
    )
