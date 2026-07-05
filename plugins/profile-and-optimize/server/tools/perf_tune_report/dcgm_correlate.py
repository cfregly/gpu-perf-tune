"""DCGM byte-grounded workload SoL correlator.

Phase 2 of the SoL framing workstream (per workspace CLAUDE.md
"Speed-of-light framing" 3-level rigor hierarchy: sample-share -> ncu
per-kernel -> DCGM workload-level). This module is the third level.

For a given inference-perf-bench bundle that captured its sweep window
(``inference_perfbench_v1.json.bench.captured_at`` + ``duration_effective_s``)
on a known cluster, this tool:

1. Reads the sweep window + GPU set from the bundle
2. Reads the DCGM Prometheus metric anchors from sol-ceilings.yaml
3. Probes whether the DCGM_FI_PROF_* group is exported on this cluster
4. Queries Prometheus over the window for each enabled peak's
   corresponding DCGM metric
5. Computes per-resource ``%SoL = measured / (peak * duration * n_gpus)``
6. Emits ``dcgm_correlation.json`` per cell, schema_version=1, with
   measured bytes/FLOPS + computed %SoL + provenance

The tool is dependency-injected with a ``PrometheusClient`` protocol so
unit tests can run against canned time-series without live infra. The
production caller passes a real client backed by ``the Prometheus MCP-mcp``
(``query_prometheus`` tool).

Architecture
------------

```
DcgmCorrelateInputs
  -> probe_prof_group     -> returns "prof" | "counter" | "absent"
  -> build_queries        -> list[DcgmQuery]
  -> client.query_range   -> list[TimeSeries] per query
  -> aggregate_resource   -> per-resource measured bytes/FLOPS + %SoL
  -> DcgmCorrelationResult
```

The dry-run path stops after build_queries and prints the PromQL
without executing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


_DCGM_CORRELATION_OUTPUT = "dcgm_correlation.json"


# ---------------------------------------------------------------------------
# Protocols + data classes
# ---------------------------------------------------------------------------


@dataclass
class TimeSeries:
    """One Prometheus time series returned for one (metric, labels) combo.

    Values are (unix_timestamp_s, float_value) pairs in ascending time
    order. Labels include at minimum the metric name and any
    grouping-by labels (typically `gpu` + `pod`).
    """

    metric: str
    labels: dict[str, str]
    samples: list[tuple[float, float]]


class PrometheusClient(Protocol):
    """Minimal client interface so tests can stub the Prometheus MCP-mcp.

    Production implementations wrap ``query_prometheus_range`` on the
    the Prometheus MCP-mcp server.
    """

    def query_instant(self, promql: str, ts: datetime) -> list[TimeSeries]:
        """One-shot point query (used for label/cardinality probes)."""

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_s: float,
    ) -> list[TimeSeries]:
        """Range query returning one TimeSeries per result label combo."""


@dataclass
class DcgmCorrelateInputs:
    """Operator-supplied inputs for one correlation run."""

    bundle_path: Path
    cell_dir: Path
    sweep_start: datetime
    sweep_end: datetime
    hw_key: str = "b200_sm100"
    pod_label_selector: str | None = None  # e.g. "app=basic-inference"
    namespace: str = "inference"
    expected_n_gpus: int | None = None  # if known, e.g. TP=8 -> 8

    @property
    def duration_s(self) -> float:
        return (self.sweep_end - self.sweep_start).total_seconds()


@dataclass
class DcgmQuery:
    """One pre-built PromQL query for one peak's DCGM proxy."""

    peak_key: str          # e.g. "hbm3e_tbps"
    metric: str            # e.g. "DCGM_FI_PROF_DRAM_ACTIVE"
    promql: str            # the actual query string
    unit: str              # "ratio" | "bytes_per_scrape" | "bytes"
    is_fallback: bool      # True if this is the DCGM_FI_DEV_* fallback
    metrics_combined: list[str] = field(default_factory=list)  # for TX+RX-style


@dataclass
class ResourceResult:
    """One row of the dcgm_correlation.json output."""

    peak_key: str
    metric: str
    is_fallback: bool
    n_gpus: int
    measured_bytes_total: float | None
    measured_bytes_per_s: float | None
    measured_tflops_avg: float | None
    peak_per_gpu: float
    peak_per_gpu_units: str
    peak_aggregate: float    # peak_per_gpu * n_gpus (in same units as peak)
    sol_pct: float | None    # 0..100; None if measurement unavailable
    notes: list[str] = field(default_factory=list)


@dataclass
class PerCategoryAttribution:
    """One row of the per_category_attribution block.

    Built by ``cross_attribute_zymtrace``: each zymtrace category
    (NCCL, MoE, FMHA, etc.) gets a measured byte-traffic share of the
    DCGM workload totals, attributed by the category's time-share.

    The "effective" fields are normalised to the category's time window
    (not the full sweep) so they directly compare to peaks. Example: if
    NCCL ran for 30% of the sweep and accounted for 30% of DCGM-measured
    DRAM bytes, effective_bw_during_category_window is the same as the
    workload average; if NCCL ran 10% but moved 50% of bytes, its
    effective_bw is 5x the workload average and %SoL_BW jumps
    accordingly.
    """

    category: str
    time_share_pct: float
    attributed_bytes_total: float | None
    attributed_flops_total: float | None
    effective_bw_during_category_window: float | None  # bytes/s
    effective_tflops_during_category_window: float | None
    sol_pct_bw: float | None  # 0..100, vs hbm3e_tbps OR nvlink5_tbps depending on category
    sol_pct_compute: float | None  # 0..100, vs nvfp4_dense_pflops OR bf16_dense_pflops
    bound: str | None  # "bandwidth" | "compute" | None (per category_ceiling_map)
    ceiling_metric: str | None  # the YAML key used for sol_pct_bw OR sol_pct_compute


@dataclass
class DcgmCorrelationResult:
    """Top-level output written to dcgm_correlation.json."""

    schema_version: int
    captured_sources: list[str]
    hw_key: str
    sweep_start_utc: str
    sweep_end_utc: str
    duration_s: float
    n_gpus: int
    dcgm_group_level: str        # "prof" | "counter" | "absent"
    scrape_interval_s: float | None
    short_sweep_warning: bool
    resources: list[ResourceResult]
    queries: list[dict[str, Any]]  # provenance: the actual PromQL fired
    dry_run: bool
    # Mean per-GPU power draw (watts) over the bench window, from
    # DCGM_FI_DEV_POWER_USAGE (added v1.42.0). Enables tokens-per-watt in the
    # economics/cost_v1 table. None when power was not captured.
    power_watts_per_gpu: float | None = None
    # Provenance for a later re-query of windowed DCGM metrics (added v1.51.0).
    # `nodes` is the distinct host/node identifier(s) the DCGM series carried
    # (DCGM_FI_DEV_POWER_USAGE is per-node, so without the node a window-only
    # capture cannot be re-queried for tokens-per-watt). `namespace` +
    # `pod_label_selector` record the pod scope. All best-effort; empty/None
    # when the exporter did not expose a host label.
    nodes: list[str] = field(default_factory=list)
    namespace: str | None = None
    pod_label_selector: str | None = None
    # Phase B (2026-05-27): Level-2 zymtrace x DCGM cross-attribution.
    # Populated when ``correlate(kernels_json_path=...)`` was supplied;
    # None / empty list otherwise. Each row attributes a slice of the
    # workload-level DCGM bytes/FLOPS to one zymtrace category via the
    # category's time-share.
    per_category_attribution: list[PerCategoryAttribution] = field(default_factory=list)
    kernels_json_path: str | None = None  # provenance: source of zymtrace cross-attribution

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resources"] = [asdict(r) for r in self.resources]
        d["per_category_attribution"] = [asdict(c) for c in self.per_category_attribution]
        return d


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def _build_label_filter(inputs: DcgmCorrelateInputs) -> str:
    """Build the {ns="...", pod=~"..."} Prometheus label filter."""
    parts: list[str] = [f'namespace="{inputs.namespace}"']
    if inputs.pod_label_selector:
        # Convert k=v style to a pod=~"..." regex (rudimentary; production
        # uses kube-state-metrics joins, but this captures the common case).
        if "=" in inputs.pod_label_selector:
            k, v = inputs.pod_label_selector.split("=", 1)
            if k == "app":
                # Pod names typically start with the app label value
                parts.append(f'pod=~"{v}-.*"')
    return "{" + ",".join(parts) + "}"


def build_queries(
    inputs: DcgmCorrelateInputs,
    ceilings: dict[str, Any],
    dcgm_group_level: str,
) -> list[DcgmQuery]:
    """Construct the PromQL queries for each peak that has a DCGM mapping.

    Args:
        inputs: correlator inputs (window + cluster + pod selector).
        ceilings: parsed sol-ceilings.yaml dict.
        dcgm_group_level: "prof" (use DCGM_FI_PROF_*), "counter" (use
            DCGM_FI_DEV_* fallback), or "absent" (return empty list).
    """
    if dcgm_group_level == "absent":
        return []

    hw_data = ceilings.get(inputs.hw_key, {})
    if not hw_data:
        return []

    label_filter = _build_label_filter(inputs)
    queries: list[DcgmQuery] = []

    for peak_key, entry in hw_data.items():
        if not isinstance(entry, dict):
            continue

        # Prefer dcgm_metric (single ratio) or dcgm_metrics_bytes (paired counters);
        # fall back to dcgm_fallback_metric if PROF group not available.
        use_prof = dcgm_group_level == "prof"

        if use_prof and "dcgm_metrics_bytes" in entry:
            # Paired byte counters (TX + RX). Sum them.
            metrics = entry["dcgm_metrics_bytes"]
            sum_terms = " + ".join(f'rate({m}{label_filter}[1m])' for m in metrics)
            queries.append(DcgmQuery(
                peak_key=peak_key,
                metric=f"sum({','.join(metrics)})",
                promql=f"sum by (gpu) ({sum_terms})",
                unit="bytes_per_s",  # rate(counter_bytes[1m]) yields bytes/s
                is_fallback=False,
                metrics_combined=metrics,
            ))
        elif use_prof and "dcgm_metric" in entry:
            metric = entry["dcgm_metric"]
            unit = entry.get("dcgm_unit", "ratio")
            queries.append(DcgmQuery(
                peak_key=peak_key,
                metric=metric,
                promql=f"avg by (gpu) ({metric}{label_filter})",
                unit=unit,
                is_fallback=False,
            ))
        elif not use_prof and "dcgm_fallback_metric" in entry:
            metric = entry["dcgm_fallback_metric"]
            queries.append(DcgmQuery(
                peak_key=peak_key,
                metric=metric,
                promql=f"avg by (gpu) ({metric}{label_filter})",
                unit="counter",
                is_fallback=True,
            ))

    return queries


# ---------------------------------------------------------------------------
# Probing + aggregation
# ---------------------------------------------------------------------------


def probe_prof_group(
    client: PrometheusClient,
    inputs: DcgmCorrelateInputs,
    ceilings: dict[str, Any],
) -> str:
    """Decide whether DCGM_FI_PROF_* is exported on this cluster.

    Returns one of:
    - "prof": at least one PROF probe metric returned series in last hour
    - "counter": no PROF metrics, but counter-only DCGM_FI_DEV_* present
    - "absent": no DCGM metrics at all
    """
    cfg = ceilings.get("dcgm_config") or {}
    probe_metrics = cfg.get("prof_group_probe_metrics") or [
        "DCGM_FI_PROF_DRAM_ACTIVE",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    ]
    label_filter = _build_label_filter(inputs)
    for metric in probe_metrics:
        series = client.query_instant(
            f"{metric}{label_filter}",
            inputs.sweep_start,
        )
        if series:
            return "prof"

    # Probe counter fallback. Use DCGM_FI_DEV_GPU_UTIL as a baseline existence check.
    counter_series = client.query_instant(
        f"DCGM_FI_DEV_GPU_UTIL{label_filter}",
        inputs.sweep_start,
    )
    if counter_series:
        return "counter"
    return "absent"


def _aggregate_ratio_resource(
    series: list[TimeSeries],
    duration_s: float,
    peak_per_gpu: float,
    peak_units: str,
    peak_key: str,
    metric: str,
) -> ResourceResult:
    """Aggregate a ratio-style DCGM time series (e.g. DRAM_ACTIVE, PIPE_TENSOR_ACTIVE).

    For a ratio metric R(t) in [0,1], integrate to get a time-weighted
    average busy fraction across the window, then multiply by peak to
    get achieved throughput.
    """
    gpu_ratios: dict[str, float] = {}  # gpu_uuid_or_id -> time-weighted mean ratio
    for ts in series:
        gpu = ts.labels.get("gpu") or ts.labels.get("device") or "unknown"
        if not ts.samples:
            continue
        # Time-weighted mean across samples.
        if len(ts.samples) == 1:
            gpu_ratios[gpu] = ts.samples[0][1]
            continue
        total_weighted = 0.0
        total_weight = 0.0
        for i in range(1, len(ts.samples)):
            t0, v0 = ts.samples[i - 1]
            t1, v1 = ts.samples[i]
            dt = t1 - t0
            total_weighted += (v0 + v1) / 2.0 * dt
            total_weight += dt
        gpu_ratios[gpu] = total_weighted / total_weight if total_weight > 0 else 0.0

    n_gpus = len(gpu_ratios)
    if n_gpus == 0:
        return ResourceResult(
            peak_key=peak_key,
            metric=metric,
            is_fallback=False,
            n_gpus=0,
            measured_bytes_total=None,
            measured_bytes_per_s=None,
            measured_tflops_avg=None,
            peak_per_gpu=peak_per_gpu,
            peak_per_gpu_units=peak_units,
            peak_aggregate=peak_per_gpu,
            sol_pct=None,
            notes=["no series returned for this metric"],
        )

    avg_ratio = sum(gpu_ratios.values()) / n_gpus
    # avg_ratio is the fraction of peak utilised. Multiplying by peak
    # gives achieved throughput in the same units as peak (TB/s for
    # bandwidth peaks, PFLOPS for compute peaks).
    achieved_per_gpu = avg_ratio * peak_per_gpu
    measured_bytes_total = None
    measured_bytes_per_s = None
    measured_tflops_avg = None
    if peak_units == "TB/s":
        measured_bytes_per_s = achieved_per_gpu * 1e12  # TB/s -> bytes/s
        measured_bytes_total = measured_bytes_per_s * duration_s * n_gpus
    elif peak_units == "PFLOPS":
        measured_tflops_avg = achieved_per_gpu * 1000.0  # PFLOPS -> TFLOPS
    elif peak_units == "TFLOPS":
        measured_tflops_avg = achieved_per_gpu

    return ResourceResult(
        peak_key=peak_key,
        metric=metric,
        is_fallback=False,
        n_gpus=n_gpus,
        measured_bytes_total=measured_bytes_total,
        measured_bytes_per_s=measured_bytes_per_s,
        measured_tflops_avg=measured_tflops_avg,
        peak_per_gpu=peak_per_gpu,
        peak_per_gpu_units=peak_units,
        peak_aggregate=peak_per_gpu * n_gpus,
        sol_pct=avg_ratio * 100.0,
    )


def _aggregate_bytes_per_s_resource(
    series: list[TimeSeries],
    duration_s: float,
    peak_per_gpu_tbps: float,
    peak_key: str,
    metrics_combined: list[str],
) -> ResourceResult:
    """Aggregate a bytes/s series (e.g. NVLink TX+RX rate sum).

    The PromQL is ``rate(metric[1m])``, returning bytes/s. We compute
    the time-weighted average bytes/s per GPU then multiply by duration
    to get total bytes.
    """
    gpu_bps: dict[str, float] = {}
    for ts in series:
        gpu = ts.labels.get("gpu") or ts.labels.get("device") or "unknown"
        if not ts.samples:
            continue
        total_weighted = 0.0
        total_weight = 0.0
        for i in range(1, len(ts.samples)):
            t0, v0 = ts.samples[i - 1]
            t1, v1 = ts.samples[i]
            dt = t1 - t0
            total_weighted += (v0 + v1) / 2.0 * dt
            total_weight += dt
        if total_weight == 0 and ts.samples:
            gpu_bps[gpu] = ts.samples[0][1]
        elif total_weight > 0:
            gpu_bps[gpu] = total_weighted / total_weight

    n_gpus = len(gpu_bps)
    metric_name = " + ".join(metrics_combined) if metrics_combined else "?"
    if n_gpus == 0:
        return ResourceResult(
            peak_key=peak_key,
            metric=metric_name,
            is_fallback=False,
            n_gpus=0,
            measured_bytes_total=None,
            measured_bytes_per_s=None,
            measured_tflops_avg=None,
            peak_per_gpu=peak_per_gpu_tbps,
            peak_per_gpu_units="TB/s",
            peak_aggregate=peak_per_gpu_tbps,
            sol_pct=None,
            notes=["no series returned for this metric"],
        )

    avg_bps = sum(gpu_bps.values()) / n_gpus
    measured_bytes_total = avg_bps * duration_s * n_gpus
    peak_bps = peak_per_gpu_tbps * 1e12
    sol_pct = (avg_bps / peak_bps) * 100.0 if peak_bps > 0 else None

    return ResourceResult(
        peak_key=peak_key,
        metric=metric_name,
        is_fallback=False,
        n_gpus=n_gpus,
        measured_bytes_total=measured_bytes_total,
        measured_bytes_per_s=avg_bps,
        measured_tflops_avg=None,
        peak_per_gpu=peak_per_gpu_tbps,
        peak_per_gpu_units="TB/s",
        peak_aggregate=peak_per_gpu_tbps * n_gpus,
        sol_pct=sol_pct,
    )


def _extract_workload_totals(
    resources: list[ResourceResult],
) -> dict[str, dict[str, float]]:
    """Pull bytes/FLOPS totals from the workload-level resource rows.

    Returns ``{peak_key: {"bytes_total": x, "flops_total": y, "peak_per_gpu": p, "n_gpus": n}}``
    for the peaks that the cross-attribution helper consumes. Only
    resources with non-None measurements appear in the result.
    """
    out: dict[str, dict[str, float]] = {}
    for r in resources:
        entry: dict[str, float] = {
            "peak_per_gpu": r.peak_per_gpu,
            "n_gpus": float(r.n_gpus),
        }
        if r.measured_bytes_total is not None:
            entry["bytes_total"] = r.measured_bytes_total
        if r.measured_tflops_avg is not None:
            # Convert avg-TFLOPS-per-GPU to total-FLOPS-during-sweep
            # for the per-category cross-attribution math. We don't
            # know the duration here, so the caller passes it in.
            entry["tflops_avg_per_gpu"] = r.measured_tflops_avg
        if entry:
            out[r.peak_key] = entry
    return out


def cross_attribute_zymtrace(
    result: DcgmCorrelationResult,
    kernels_json: dict[str, Any],
    ceilings: dict[str, Any],
) -> list[PerCategoryAttribution]:
    """Attribute DCGM workload totals to zymtrace categories.

    Computes per-category byte / FLOP traffic via time-share weighting:

        category_bytes = total_dcgm_bytes * (category_samples / total_samples)
        category_flops = total_dcgm_flops * (category_samples / total_samples)

    Then derives effective per-second throughput during the category's
    time slice (not the full sweep) and compares to the category's
    natural peak from ``category_ceiling_map``.

    Args:
        result: the workload-level ``DcgmCorrelationResult`` to be
            extended.
        kernels_json: parsed zymtrace ``kernels.json`` payload with
            ``per_category`` sample counts.
        ceilings: parsed sol-ceilings.yaml dict (provides
            ``category_ceiling_map`` + per-hardware peaks).

    Returns:
        list of ``PerCategoryAttribution`` rows (empty if zymtrace
        per_category block is missing or zero).
    """
    per_cat = kernels_json.get("per_category") or {}
    total_samples = sum(int(v) for v in per_cat.values()) if per_cat else 0
    if total_samples <= 0:
        return []

    hw_data = ceilings.get(result.hw_key, {})
    cat_map = ceilings.get("category_ceiling_map") or {}
    workload_totals = _extract_workload_totals(result.resources)

    # The bandwidth-style total bytes is the canonical HBM byte count
    # for memory-bound categories; NVLink5 for NCCL specifically. We
    # prefer HBM as the default workload-byte source; NCCL gets NVLink.
    hbm_total_bytes = workload_totals.get("hbm3e_tbps", {}).get("bytes_total")
    nvl_total_bytes = workload_totals.get("nvlink5_tbps", {}).get("bytes_total")
    # Compute (tensor pipe) avg per-GPU TFLOPS over the sweep:
    compute_avg_tflops_per_gpu = (
        workload_totals.get("nvfp4_dense_pflops", {}).get("tflops_avg_per_gpu")
        or workload_totals.get("fp8_dense_pflops", {}).get("tflops_avg_per_gpu")
        or workload_totals.get("bf16_dense_pflops", {}).get("tflops_avg_per_gpu")
    )
    n_gpus = result.n_gpus or 1
    duration_s = result.duration_s
    # Aggregate FLOPS across the sweep across all GPUs.
    compute_total_flops = (
        (compute_avg_tflops_per_gpu * 1e12 * duration_s * n_gpus)
        if compute_avg_tflops_per_gpu is not None
        else None
    )

    out: list[PerCategoryAttribution] = []
    for category, samples in per_cat.items():
        samples = int(samples)
        if samples <= 0:
            continue
        time_share = samples / total_samples
        time_share_pct = time_share * 100.0
        cat_info = cat_map.get(category) or {}
        bound = cat_info.get("bound")
        ceiling_metric_key = cat_info.get("metric")
        ceiling_entry = hw_data.get(ceiling_metric_key) if ceiling_metric_key else None
        ceiling_value = ceiling_entry.get("value") if isinstance(ceiling_entry, dict) else None
        ceiling_units = ceiling_entry.get("units") if isinstance(ceiling_entry, dict) else None

        # Pick the workload byte/FLOP source per category bound.
        if bound == "bandwidth":
            # NCCL goes against NVLink; everything else against HBM.
            if category == "NCCL" and nvl_total_bytes is not None:
                workload_bytes = nvl_total_bytes
            else:
                workload_bytes = hbm_total_bytes
            workload_flops = None
        elif bound == "compute":
            workload_bytes = hbm_total_bytes  # compute kernels still read HBM
            workload_flops = compute_total_flops
        else:
            workload_bytes = hbm_total_bytes
            workload_flops = None

        attributed_bytes = (workload_bytes * time_share) if workload_bytes is not None else None
        attributed_flops = (workload_flops * time_share) if workload_flops is not None else None

        # Effective during-category throughput: divide by category time
        # window (time_share * duration_s) rather than full duration.
        cat_window_s = time_share * duration_s
        if cat_window_s > 0 and attributed_bytes is not None:
            effective_bw = attributed_bytes / cat_window_s / n_gpus  # bytes/s/GPU
        else:
            effective_bw = None
        if cat_window_s > 0 and attributed_flops is not None:
            effective_tflops = attributed_flops / cat_window_s / n_gpus / 1e12
        else:
            effective_tflops = None

        # %SoL math: compare effective rate vs peak per-GPU
        sol_pct_bw = None
        sol_pct_compute = None
        if bound == "bandwidth" and ceiling_value is not None and effective_bw is not None:
            # ceiling_value in TB/s -> bytes/s
            peak_bps = ceiling_value * 1e12
            sol_pct_bw = (effective_bw / peak_bps) * 100.0 if peak_bps > 0 else None
        if bound == "compute" and ceiling_value is not None and effective_tflops is not None:
            # ceiling_value in PFLOPS -> TFLOPS
            peak_tflops = ceiling_value * 1000.0
            sol_pct_compute = (effective_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else None

        out.append(PerCategoryAttribution(
            category=category,
            time_share_pct=time_share_pct,
            attributed_bytes_total=attributed_bytes,
            attributed_flops_total=attributed_flops,
            effective_bw_during_category_window=effective_bw,
            effective_tflops_during_category_window=effective_tflops,
            sol_pct_bw=sol_pct_bw,
            sol_pct_compute=sol_pct_compute,
            bound=bound,
            ceiling_metric=ceiling_metric_key,
        ))
    return out


_POWER_METRIC = "DCGM_FI_DEV_POWER_USAGE"

# Label keys a DCGM series may carry the node under, in priority order.
# DCGM-exporter exposes the node as `Hostname`; some scrape configs relabel it
# to `node` / `exported_node` / `instance`.
_NODE_LABEL_KEYS = ("Hostname", "node", "exported_node", "instance")


def _collect_nodes(series: list[TimeSeries]) -> set[str]:
    """Distinct host/node identifiers from a DCGM series' labels (best-effort).

    Returns an empty set when no recognizable host label is present (so the
    `nodes` provenance simply stays empty rather than failing the correlate)."""
    out: set[str] = set()
    for s in series or []:
        for key in _NODE_LABEL_KEYS:
            val = s.labels.get(key)
            if isinstance(val, str) and val.strip():
                out.add(val.strip())
                break
    return out


def _query_mean_power_watts_per_gpu(
    client: PrometheusClient,
    inputs: DcgmCorrelateInputs,
    *,
    step_s: float,
) -> tuple[float | None, str]:
    """Mean per-GPU power draw (watts) over the bench window from
    ``DCGM_FI_DEV_POWER_USAGE``. Returns ``(power_or_None, promql)``.

    Best-effort: returns ``None`` when the metric is unavailable or the query
    fails, so tokens-per-watt degrades to null rather than blocking."""
    label_filter = _build_label_filter(inputs)
    promql = f"avg by (gpu) ({_POWER_METRIC}{label_filter})"
    try:
        series = client.query_range(
            promql, inputs.sweep_start, inputs.sweep_end, step_s=step_s
        )
    except Exception:  # noqa: BLE001 - power is best-effort; never break correlate
        return None, promql
    vals = [v for s in (series or []) for (_ts, v) in s.samples if v is not None and v > 0]
    if not vals:
        return None, promql
    return sum(vals) / len(vals), promql


def correlate(
    inputs: DcgmCorrelateInputs,
    ceilings: dict[str, Any],
    client: PrometheusClient,
    *,
    dry_run: bool = False,
    kernels_json_path: Path | None = None,
) -> DcgmCorrelationResult:
    """Run the DCGM correlation pipeline.

    Args:
        inputs: window + cluster + pod selector + hw_key.
        ceilings: parsed sol-ceilings.yaml dict.
        client: Prometheus client (production: the Prometheus MCP-mcp wrapper;
            tests: in-memory stub).
        dry_run: if True, build queries but don't execute them; return a
            result with the PromQL strings only (no data).
        kernels_json_path: optional path to a zymtrace ``kernels.json``
            payload (per-cell). When provided, the result's
            ``per_category_attribution`` block is populated via
            ``cross_attribute_zymtrace``. When None, only the
            workload-level Level-3 correlation is produced.
    """
    cfg = ceilings.get("dcgm_config") or {}
    min_sweep_s = float(cfg.get("min_sweep_seconds", 60))
    expected_scrape_s = cfg.get("expected_scrape_interval_s")

    duration_s = inputs.duration_s
    short_sweep_warning = duration_s < min_sweep_s

    if dry_run:
        # Skip probe; assume PROF for the planned queries.
        dcgm_group_level = "prof"
    else:
        dcgm_group_level = probe_prof_group(client, inputs, ceilings)

    queries = build_queries(inputs, ceilings, dcgm_group_level)

    hw_data = ceilings.get(inputs.hw_key, {})
    resources: list[ResourceResult] = []
    query_records: list[dict[str, Any]] = []
    node_set: set[str] = set()

    for q in queries:
        query_records.append({
            "peak_key": q.peak_key,
            "metric": q.metric,
            "promql": q.promql,
            "unit": q.unit,
            "is_fallback": q.is_fallback,
        })
        if dry_run:
            continue
        series = client.query_range(
            q.promql,
            inputs.sweep_start,
            inputs.sweep_end,
            step_s=float(expected_scrape_s or 15),
        )
        node_set.update(_collect_nodes(series))
        peak_entry = hw_data.get(q.peak_key, {})
        peak_value = peak_entry.get("value", 0.0)
        peak_units = peak_entry.get("units", "")

        if q.unit == "bytes_per_s":
            res = _aggregate_bytes_per_s_resource(
                series,
                duration_s,
                peak_per_gpu_tbps=peak_value,
                peak_key=q.peak_key,
                metrics_combined=q.metrics_combined,
            )
        elif q.unit == "ratio":
            res = _aggregate_ratio_resource(
                series,
                duration_s,
                peak_per_gpu=peak_value,
                peak_units=peak_units,
                peak_key=q.peak_key,
                metric=q.metric,
            )
        else:
            # Counter fallback: emit a placeholder result with a warning;
            # operator can interpret raw counter values manually.
            res = ResourceResult(
                peak_key=q.peak_key,
                metric=q.metric,
                is_fallback=q.is_fallback,
                n_gpus=0,
                measured_bytes_total=None,
                measured_bytes_per_s=None,
                measured_tflops_avg=None,
                peak_per_gpu=peak_value,
                peak_per_gpu_units=peak_units,
                peak_aggregate=peak_value,
                sol_pct=None,
                notes=[
                    f"counter-fallback metric {q.metric}: byte-rate "
                    "derivation not implemented; refer to raw payload"
                ],
            )

        if short_sweep_warning:
            res.notes.append(
                f"short sweep ({duration_s:.0f}s < {min_sweep_s:.0f}s): "
                "DCGM scrape granularity makes integration coarse"
            )

        resources.append(res)

    n_gpus = max((r.n_gpus for r in resources), default=(inputs.expected_n_gpus or 0))

    # Mean per-GPU power over the window (enables tokens-per-watt in cost_v1).
    # Best-effort: null when the metric is absent or the query fails.
    power_watts_per_gpu: float | None = None
    if not dry_run:
        power_watts_per_gpu, power_promql = _query_mean_power_watts_per_gpu(
            client, inputs, step_s=float(expected_scrape_s or 15)
        )
        query_records.append({
            "peak_key": "power_watts_per_gpu",
            "metric": _POWER_METRIC,
            "promql": power_promql,
            "unit": "watts",
            "is_fallback": False,
        })

    result = DcgmCorrelationResult(
        schema_version=1,
        captured_sources=["dcgm"],
        hw_key=inputs.hw_key,
        sweep_start_utc=inputs.sweep_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sweep_end_utc=inputs.sweep_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_s=duration_s,
        n_gpus=n_gpus,
        dcgm_group_level=dcgm_group_level,
        scrape_interval_s=expected_scrape_s,
        short_sweep_warning=short_sweep_warning,
        resources=resources,
        queries=query_records,
        dry_run=dry_run,
        power_watts_per_gpu=power_watts_per_gpu,
        nodes=sorted(node_set),
        namespace=inputs.namespace,
        pod_label_selector=inputs.pod_label_selector,
    )

    # Phase B (2026-05-27): Level-2 zymtrace x DCGM cross-attribution.
    # When the caller passed a kernels.json path, layer per-category
    # attribution on top of the workload-level result.
    if kernels_json_path is not None and not dry_run:
        kernels_json_path = Path(kernels_json_path).expanduser().resolve()
        if kernels_json_path.is_file():
            try:
                kernels_payload = json.loads(kernels_json_path.read_text())
            except json.JSONDecodeError:
                kernels_payload = None
            if kernels_payload is not None:
                result.per_category_attribution = cross_attribute_zymtrace(
                    result, kernels_payload, ceilings
                )
                result.kernels_json_path = str(kernels_json_path)
                if "zymtrace" not in result.captured_sources:
                    result.captured_sources = list(result.captured_sources) + ["zymtrace"]

    return result


def write_correlation(
    result: DcgmCorrelationResult,
    cell_dir: Path,
) -> Path:
    """Write the result to ``<cell_dir>/dcgm_correlation.json``."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    out_path = cell_dir / _DCGM_CORRELATION_OUTPUT
    out_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return out_path


# ---------------------------------------------------------------------------
# Frozen-correlation path (v1.23.1)
#
# Some operators capture DCGM measurements interactively via the
# the Prometheus MCP-mcp `query_prometheus` tool then need to fold those
# numbers into a `dcgm_correlation.json` for the renderer. The live
# `correlate()` path needs a Prometheus client that the agent context
# may not have; the frozen path takes a YAML carrying the already-
# captured measurements + ceiling references and produces the same
# `DcgmCorrelationResult` shape.
#
# Schema (`dcgm_frozen_v1`):
#
#     schema_version: 1
#     hw_key: b200_sm100
#     n_gpus: 8
#     sweep_start_utc: "2026-05-27T13:29:00Z"
#     sweep_end_utc: "2026-05-27T13:52:00Z"
#     captured_at: "2026-05-28T..."
#     captured_by: "<operator>@<workstation>"
#     data_tier: prof   # or counter
#     scrape_interval_s: 30.0
#     notes: |
#       Free-form provenance notes (cluster, deploy label selector,
#       what the measurements describe, partial-success caveats, etc.).
#     resources:
#       - peak_key: hbm3e_tbps
#         metric: DCGM_FI_PROF_DRAM_ACTIVE
#         unit: ratio                 # measured value below is a ratio (0..1)
#         measured_avg: 0.155         # time-weighted mean across the window
#         promql: 'avg by (gpu) (DCGM_FI_PROF_DRAM_ACTIVE{...})'
#       - peak_key: nvlink5_tbps
#         metric: "sum(NVLINK_TX_BYTES, NVLINK_RX_BYTES)"
#         unit: bytes_per_s           # measured value below is bytes/s per GPU
#         measured_avg: 400000000.0   # 400 MB/s per GPU mean rate
#         promql: '...'
#       - peak_key: nvfp4_dense_pflops
#         metric: DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
#         unit: ratio
#         measured_avg: 0.018
#         promql: '...'
# ---------------------------------------------------------------------------


class FrozenYamlMalformed(Exception):
    """Raised when a frozen YAML is present but missing required keys."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"frozen YAML malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


def _resource_from_frozen(
    entry: dict[str, Any],
    n_gpus: int,
    duration_s: float,
    hw_data: dict[str, Any],
) -> ResourceResult:
    """Materialise one ResourceResult from a frozen YAML resource entry.

    The unit-specific computation mirrors the inline math in the
    pre-v1.23.1 workshop scripts:

    - ``ratio``: measured_avg is fraction-of-peak. For bandwidth peaks
      the bytes/s = measured_avg * peak * 1e12 (TB/s -> bytes/s);
      for compute peaks the achieved TFLOPS = measured_avg * peak * 1000
      (PFLOPS -> TFLOPS). %SoL = measured_avg * 100.
    - ``bytes_per_s``: measured_avg is bytes/s/GPU. Bandwidth-only;
      %SoL = (measured_avg / (peak * 1e12)) * 100.
    """
    peak_key = entry["peak_key"]
    metric = entry["metric"]
    unit = entry["unit"]
    measured_avg = float(entry["measured_avg"])

    peak_entry = hw_data.get(peak_key, {}) or {}
    peak_value = float(peak_entry.get("value", 0.0))
    peak_units = peak_entry.get("units", "")

    notes: list[str] = []
    if entry.get("promql"):
        notes.append(f"promql: {entry['promql']}")

    if unit == "ratio":
        # Determine whether this peak is bandwidth (bytes) or compute (FLOPS)
        # by looking at the units string in the ceilings YAML.
        is_compute = "FLOPS" in peak_units.upper()
        if is_compute:
            # PFLOPS peak * ratio = achieved PFLOPS per GPU; convert to TFLOPS
            achieved_tflops = measured_avg * peak_value * 1000.0  # PFLOPS -> TFLOPS
            return ResourceResult(
                peak_key=peak_key,
                metric=metric,
                is_fallback=False,
                n_gpus=n_gpus,
                measured_bytes_total=None,
                measured_bytes_per_s=None,
                measured_tflops_avg=achieved_tflops,
                peak_per_gpu=peak_value,
                peak_per_gpu_units=peak_units,
                peak_aggregate=peak_value * n_gpus,
                sol_pct=measured_avg * 100.0,
                notes=notes,
            )
        # Bandwidth ratio (e.g. DRAM_ACTIVE)
        bytes_per_s_per_gpu = measured_avg * peak_value * 1e12  # TB/s -> bytes/s
        bytes_total = bytes_per_s_per_gpu * duration_s * n_gpus
        return ResourceResult(
            peak_key=peak_key,
            metric=metric,
            is_fallback=False,
            n_gpus=n_gpus,
            measured_bytes_total=bytes_total,
            measured_bytes_per_s=bytes_per_s_per_gpu,
            measured_tflops_avg=None,
            peak_per_gpu=peak_value,
            peak_per_gpu_units=peak_units,
            peak_aggregate=peak_value * n_gpus,
            sol_pct=measured_avg * 100.0,
            notes=notes,
        )
    if unit == "bytes_per_s":
        # measured_avg is bytes/s/GPU.
        bytes_total = measured_avg * duration_s * n_gpus
        peak_bps = peak_value * 1e12  # peak (TB/s) -> bytes/s/GPU
        sol_pct = (measured_avg / peak_bps) * 100.0 if peak_bps > 0 else 0.0
        return ResourceResult(
            peak_key=peak_key,
            metric=metric,
            is_fallback=False,
            n_gpus=n_gpus,
            measured_bytes_total=bytes_total,
            measured_bytes_per_s=measured_avg,
            measured_tflops_avg=None,
            peak_per_gpu=peak_value,
            peak_per_gpu_units=peak_units,
            peak_aggregate=peak_value * n_gpus,
            sol_pct=sol_pct,
            notes=notes,
        )
    raise ValueError(
        f"_resource_from_frozen: unsupported unit {unit!r} on peak_key={peak_key} "
        f"(expected 'ratio' or 'bytes_per_s')"
    )


def correlate_from_frozen(
    frozen_yaml: Path,
    ceilings: dict[str, Any],
    *,
    cell_dir: Path | None = None,
    kernels_json_path: Path | None = None,
) -> DcgmCorrelationResult:
    """Build a ``DcgmCorrelationResult`` from a frozen YAML.

    Used when DCGM measurements were captured interactively (e.g. via
    the the Prometheus MCP-mcp ``query_prometheus`` tool) and need to be
    folded into a ``dcgm_correlation.json`` without re-running the
    live `correlate()` pipeline.

    Args:
        frozen_yaml: path to a ``dcgm_frozen_v1`` YAML.
        ceilings: parsed ``sol-ceilings.yaml`` dict.
        cell_dir: optional cell directory; if provided AND
            ``kernels_json_path`` is None, the function looks for
            ``<cell_dir>/kernels.json`` and uses it for the
            zymtrace cross-attribution (level-2). If neither is
            provided, the result has ``per_category_attribution=[]``.
        kernels_json_path: explicit override for the kernels.json
            path (takes precedence over the implicit cell_dir lookup).

    Raises:
        FrozenYamlMalformed: required keys missing.
        FileNotFoundError: yaml not present.
    """
    import yaml as _yaml  # lazy import

    if not frozen_yaml.is_file():
        raise FileNotFoundError(frozen_yaml)

    try:
        data = _yaml.safe_load(frozen_yaml.read_text())
    except _yaml.YAMLError as e:
        raise FrozenYamlMalformed(frozen_yaml, reason=f"YAML parse error: {e}") from e

    if not isinstance(data, dict):
        raise FrozenYamlMalformed(frozen_yaml, reason="top-level not a mapping")

    for required in ("hw_key", "n_gpus", "sweep_start_utc", "sweep_end_utc", "resources"):
        if required not in data:
            raise FrozenYamlMalformed(
                frozen_yaml, reason=f"missing required key '{required}'"
            )

    hw_key = data["hw_key"]
    if hw_key not in ceilings:
        raise FrozenYamlMalformed(
            frozen_yaml,
            reason=f"hw_key '{hw_key}' not present in ceilings YAML",
        )
    hw_data = ceilings[hw_key]

    sweep_start = datetime.fromisoformat(
        str(data["sweep_start_utc"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    sweep_end = datetime.fromisoformat(
        str(data["sweep_end_utc"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    duration_s = (sweep_end - sweep_start).total_seconds()
    if duration_s <= 0:
        raise FrozenYamlMalformed(
            frozen_yaml, reason=f"non-positive duration_s={duration_s}"
        )

    n_gpus = int(data["n_gpus"])

    resources_in = data["resources"]
    if not isinstance(resources_in, list) or not resources_in:
        raise FrozenYamlMalformed(
            frozen_yaml, reason="resources must be a non-empty list"
        )

    resources: list[ResourceResult] = []
    queries: list[dict[str, Any]] = []
    for entry in resources_in:
        if not isinstance(entry, dict):
            raise FrozenYamlMalformed(
                frozen_yaml, reason=f"resource entry not a mapping: {entry!r}"
            )
        for required in ("peak_key", "metric", "unit", "measured_avg"):
            if required not in entry:
                raise FrozenYamlMalformed(
                    frozen_yaml,
                    reason=f"resource entry missing '{required}': {entry!r}",
                )
        resources.append(_resource_from_frozen(entry, n_gpus, duration_s, hw_data))
        queries.append({
            "peak_key": entry["peak_key"],
            "metric": entry["metric"],
            "promql": str(entry.get("promql", "")),
            "unit": entry["unit"],
            "is_fallback": False,
        })

    # Optional mean per-GPU power (watts) from the frozen YAML -> tokens-per-watt.
    power_in = data.get("power_watts_per_gpu")
    power_watts_per_gpu = (
        float(power_in) if isinstance(power_in, (int, float)) and power_in > 0 else None
    )

    # Optional re-query provenance (v1.51.0): node(s) + pod scope.
    nodes_in = data.get("nodes")
    nodes = (
        [str(n).strip() for n in nodes_in if str(n).strip()]
        if isinstance(nodes_in, list)
        else []
    )
    namespace_in = data.get("namespace")
    namespace = str(namespace_in) if isinstance(namespace_in, str) and namespace_in.strip() else None
    selector_in = data.get("pod_label_selector")
    pod_label_selector = (
        str(selector_in) if isinstance(selector_in, str) and selector_in.strip() else None
    )

    result = DcgmCorrelationResult(
        schema_version=1,
        captured_sources=["dcgm"] + (
            ["zymtrace"] if (cell_dir or kernels_json_path) else []
        ),
        hw_key=hw_key,
        sweep_start_utc=sweep_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sweep_end_utc=sweep_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_s=duration_s,
        n_gpus=n_gpus,
        dcgm_group_level=str(data.get("data_tier", "prof")),
        scrape_interval_s=float(data.get("scrape_interval_s", 30.0)),
        short_sweep_warning=False,
        resources=resources,
        queries=queries,
        dry_run=False,
        power_watts_per_gpu=power_watts_per_gpu,
        nodes=nodes,
        namespace=namespace,
        pod_label_selector=pod_label_selector,
    )

    # Optional zymtrace cross-attribution.
    if kernels_json_path is None and cell_dir is not None:
        candidate = cell_dir / "kernels.json"
        if candidate.is_file():
            kernels_json_path = candidate

    if kernels_json_path is not None and kernels_json_path.is_file():
        kernels_payload = json.loads(kernels_json_path.read_text())
        result.per_category_attribution = cross_attribute_zymtrace(
            result, kernels_payload, ceilings
        )
        result.kernels_json_path = str(kernels_json_path)

    return result


# ---------------------------------------------------------------------------
# Helper for tests + CLI: read sweep window from inference_perfbench_v1.json
# ---------------------------------------------------------------------------


def read_sweep_window_from_bundle(bundle_path: Path) -> tuple[datetime, datetime]:
    """Read sweep start + end from ``<bundle>/inference_perfbench_v1.json``.

    Returns ``(sweep_start, sweep_end)`` as timezone-aware UTC datetimes.
    Raises FileNotFoundError if the file is absent; KeyError if the
    expected keys are missing.
    """
    ipb_path = bundle_path / "inference_perfbench_v1.json"
    if not ipb_path.is_file():
        raise FileNotFoundError(ipb_path)
    ipb = json.loads(ipb_path.read_text())
    captured_at = ipb["bench"]["captured_at"] if "captured_at" in ipb.get("bench", {}) else None
    if not captured_at:
        # Some bundles use top-level captured_at. Fallback.
        captured_at = ipb.get("captured_at") or ipb.get("bench", {}).get("captured_at_utc")
    duration_s = float(ipb["bench"]["duration_effective_s"])
    # Parse ISO-8601; tolerate trailing Z.
    cs = captured_at.replace("Z", "+00:00")
    start = datetime.fromisoformat(cs).astimezone(timezone.utc)
    # Production captured_at may be either start or end; the rest of the
    # codebase uses it as a timestamp at capture write-time = effectively
    # end-of-sweep. Convention here: treat captured_at as start; user
    # who knows otherwise can pass explicit start/end to DcgmCorrelateInputs.
    end = datetime.fromtimestamp(start.timestamp() + duration_s, tz=timezone.utc)
    return start, end
