"""Tests for the DCGM byte-grounded workload SoL correlator.

Stubs the Prometheus client with a fake that returns canned series
keyed by query string, so no live the Prometheus MCP access is required.

Scenarios:

A. probe_prof_group: prof / counter / absent
B. build_queries: PROF-mode produces sane PromQL per peak; counter-mode falls back
C. correlate happy path: ratio + bytes/s aggregation; %SoL within expected band
D. dry_run: builds queries without executing; result.dry_run == True
E. short sweep warning: when sweep < dcgm_config.min_sweep_seconds, notes fire
F. read_sweep_window_from_bundle: happy path + missing file
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.dcgm_correlate import (
    DcgmCorrelateInputs,
    PrometheusClient,
    TimeSeries,
    build_queries,
    correlate,
    correlate_from_frozen,
    cross_attribute_zymtrace,
    probe_prof_group,
    read_sweep_window_from_bundle,
    write_correlation,
)


# ---------------------------------------------------------------------------
# Fake Prometheus client
# ---------------------------------------------------------------------------


@dataclass
class _FakePromClient:
    """Records every query + returns canned series keyed by metric substring.

    Tests configure ``self.instant_responses[metric_substring] = [TimeSeries]``
    and ``self.range_responses[metric_substring] = [TimeSeries]`` before
    invoking the correlator.
    """

    instant_responses: dict[str, list[TimeSeries]] = field(default_factory=dict)
    range_responses: dict[str, list[TimeSeries]] = field(default_factory=dict)
    instant_calls: list[tuple[str, datetime]] = field(default_factory=list)
    range_calls: list[tuple[str, datetime, datetime, float]] = field(default_factory=list)

    def _match(self, table: dict[str, list[TimeSeries]], promql: str) -> list[TimeSeries]:
        for needle, series in table.items():
            if needle in promql:
                return series
        return []

    def query_instant(self, promql: str, ts: datetime) -> list[TimeSeries]:
        self.instant_calls.append((promql, ts))
        return self._match(self.instant_responses, promql)

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_s: float,
    ) -> list[TimeSeries]:
        self.range_calls.append((promql, start, end, step_s))
        return self._match(self.range_responses, promql)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_CEILINGS: dict = {
    "b200_sm100": {
        "hw_name": "B200 test",
        "nvfp4_dense_pflops": {
            "value": 9.0,
            "units": "PFLOPS",
            "dcgm_metric": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
            "dcgm_unit": "ratio",
        },
        "bf16_dense_pflops": {
            "value": 2.25,
            "units": "PFLOPS",
            "dcgm_metric": "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
            "dcgm_unit": "ratio",
        },
        "hbm3e_tbps": {
            "value": 8.0,
            "units": "TB/s",
            "dcgm_metric": "DCGM_FI_PROF_DRAM_ACTIVE",
            "dcgm_unit": "ratio",
            "dcgm_fallback_metric": "DCGM_FI_DEV_FB_USED",
        },
        "nvlink5_tbps": {
            "value": 1.8,
            "units": "TB/s",
            "dcgm_metrics_bytes": [
                "DCGM_FI_PROF_NVLINK_TX_BYTES",
                "DCGM_FI_PROF_NVLINK_RX_BYTES",
            ],
            "dcgm_unit": "bytes_per_scrape",
            "dcgm_fallback_metric": "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
        },
    },
    "category_ceiling_map": {
        "NCCL":         {"metric": "nvlink5_tbps",       "bound": "bandwidth"},
        "MoE":          {"metric": "nvfp4_dense_pflops", "bound": "compute"},
        "FMHA":         {"metric": "hbm3e_tbps",         "bound": "bandwidth"},
        "BMM-NVFP4":    {"metric": "nvfp4_dense_pflops", "bound": "compute"},
        "Triton-fused": {"metric": "bf16_dense_pflops",  "bound": "compute"},
        "cuBLAS":       {"metric": "bf16_dense_pflops",  "bound": "compute"},
        "Elementwise":  {"metric": "hbm3e_tbps",         "bound": "bandwidth"},
        "Other":        {"metric": "hbm3e_tbps",         "bound": "bandwidth"},
    },
    "dcgm_config": {
        "default_labels": ["gpu", "device"],
        "min_sweep_seconds": 60,
        "expected_scrape_interval_s": 15,
        "prof_group_probe_metrics": [
            "DCGM_FI_PROF_DRAM_ACTIVE",
            "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        ],
    },
}


def _inputs(tmp_path: Path, *, start: datetime, duration_s: float) -> DcgmCorrelateInputs:
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    cell = tmp_path / "campaign" / "cells" / "c1"
    return DcgmCorrelateInputs(
        bundle_path=bundle,
        cell_dir=cell,
        sweep_start=start,
        sweep_end=start + timedelta(seconds=duration_s),
        hw_key="b200_sm100",
        pod_label_selector="app=basic-inference",
        namespace="inference",
        expected_n_gpus=8,
    )


def _series(metric: str, gpu: str, samples: list[tuple[float, float]]) -> TimeSeries:
    return TimeSeries(metric=metric, labels={"gpu": gpu}, samples=samples)


_START = datetime(2026, 5, 27, 13, 30, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# A. probe_prof_group
# ---------------------------------------------------------------------------


def test_probe_prof_group_returns_prof(tmp_path):
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(0.0, 0.5)])]},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    assert probe_prof_group(client, inp, _CEILINGS) == "prof"


def test_probe_prof_group_returns_counter(tmp_path):
    client = _FakePromClient(
        instant_responses={
            # No PROF metrics, only counter-tier
            "DCGM_FI_DEV_GPU_UTIL": [_series("DCGM_FI_DEV_GPU_UTIL", "0", [(0.0, 70.0)])],
        },
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    assert probe_prof_group(client, inp, _CEILINGS) == "counter"


def test_probe_prof_group_returns_absent(tmp_path):
    client = _FakePromClient()  # nothing responds
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    assert probe_prof_group(client, inp, _CEILINGS) == "absent"


# ---------------------------------------------------------------------------
# B. build_queries
# ---------------------------------------------------------------------------


def test_build_queries_prof_emits_one_per_peak(tmp_path):
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    queries = build_queries(inp, _CEILINGS, dcgm_group_level="prof")
    peak_keys = {q.peak_key for q in queries}
    assert "hbm3e_tbps" in peak_keys
    assert "nvlink5_tbps" in peak_keys
    assert "nvfp4_dense_pflops" in peak_keys

    nvl = next(q for q in queries if q.peak_key == "nvlink5_tbps")
    assert "DCGM_FI_PROF_NVLINK_TX_BYTES" in nvl.promql
    assert "DCGM_FI_PROF_NVLINK_RX_BYTES" in nvl.promql
    assert nvl.unit == "bytes_per_s"

    hbm = next(q for q in queries if q.peak_key == "hbm3e_tbps")
    assert hbm.metric == "DCGM_FI_PROF_DRAM_ACTIVE"
    assert 'namespace="inference"' in hbm.promql
    assert 'pod=~"basic-inference-.*"' in hbm.promql


def test_build_queries_counter_uses_fallbacks(tmp_path):
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    queries = build_queries(inp, _CEILINGS, dcgm_group_level="counter")
    # Only entries with dcgm_fallback_metric should appear
    peak_keys = {q.peak_key for q in queries}
    assert "hbm3e_tbps" in peak_keys  # has dcgm_fallback_metric
    assert "nvlink5_tbps" in peak_keys  # has dcgm_fallback_metric
    assert "nvfp4_dense_pflops" not in peak_keys  # no fallback defined
    for q in queries:
        assert q.is_fallback


def test_build_queries_absent_returns_empty(tmp_path):
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    assert build_queries(inp, _CEILINGS, dcgm_group_level="absent") == []


# ---------------------------------------------------------------------------
# C. correlate happy path
# ---------------------------------------------------------------------------


def test_correlate_ratio_resource_happy_path(tmp_path):
    """For DRAM_ACTIVE at 0.5 ratio sustained for 120s on 8 GPUs:
    measured BW = 0.5 * 8 TB/s = 4 TB/s per GPU = 32 TB/s aggregate
    %SoL = 50%
    """
    # DCGM_FI_PROF_DRAM_ACTIVE: 8 GPUs each at 0.5 across the sweep.
    dram_series = [
        _series("DCGM_FI_PROF_DRAM_ACTIVE", str(i), [
            (_START.timestamp(), 0.5),
            (_START.timestamp() + 60, 0.5),
            (_START.timestamp() + 120, 0.5),
        ])
        for i in range(8)
    ]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram_series[:1]},  # probe
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram_series},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client)

    assert result.dcgm_group_level == "prof"
    assert result.duration_s == 120
    hbm = next(r for r in result.resources if r.peak_key == "hbm3e_tbps")
    assert hbm.n_gpus == 8
    assert hbm.sol_pct == pytest.approx(50.0, rel=1e-3)
    assert hbm.measured_bytes_per_s == pytest.approx(0.5 * 8e12, rel=1e-3)  # 4 TB/s per GPU
    # 4 TB/s per GPU * 120s * 8 GPUs
    assert hbm.measured_bytes_total == pytest.approx(0.5 * 8e12 * 120 * 8, rel=1e-3)


def test_correlate_bytes_per_s_resource_happy_path(tmp_path):
    """NVLink TX+RX rates summed to 0.9 TB/s per GPU on 8 GPUs.
    Aggregate sum = ~0.9 TB/s per GPU; %SoL = 50% of 1.8 TB/s peak.
    """
    target_bps = 0.9e12  # 0.9 TB/s per GPU
    nvl_series = [
        _series("nvlink_combined", str(i), [
            (_START.timestamp() + t, target_bps) for t in range(0, 121, 15)
        ])
        for i in range(8)
    ]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(0.0, 0.5)])]},
        range_responses={"DCGM_FI_PROF_NVLINK": nvl_series},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client)
    nvl = next(r for r in result.resources if r.peak_key == "nvlink5_tbps")
    assert nvl.n_gpus == 8
    assert nvl.sol_pct == pytest.approx(50.0, rel=1e-2)
    # 0.9 TB/s per GPU * 120s * 8 GPUs
    assert nvl.measured_bytes_total == pytest.approx(target_bps * 120 * 8, rel=1e-2)


# ---------------------------------------------------------------------------
# D. dry_run
# ---------------------------------------------------------------------------


def test_correlate_dry_run_emits_queries_without_executing(tmp_path):
    client = _FakePromClient()  # no canned data
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client, dry_run=True)

    assert result.dry_run is True
    assert client.range_calls == []  # no range queries fired
    assert len(result.queries) >= 3  # at least HBM, NVLink, Tensor
    assert result.resources == []  # no data aggregated


# ---------------------------------------------------------------------------
# E. short sweep warning
# ---------------------------------------------------------------------------


def test_correlate_short_sweep_warns(tmp_path):
    """30-second sweep is below the 60s min_sweep_seconds threshold."""
    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [
        (_START.timestamp(), 0.3),
        (_START.timestamp() + 30, 0.3),
    ])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=30)

    result = correlate(inp, _CEILINGS, client)
    assert result.short_sweep_warning is True
    hbm = next(r for r in result.resources if r.peak_key == "hbm3e_tbps")
    assert any("short sweep" in n for n in hbm.notes)


# ---------------------------------------------------------------------------
# F. read_sweep_window_from_bundle
# ---------------------------------------------------------------------------


def test_read_sweep_window_happy_path(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "inference_perfbench_v1.json").write_text(json.dumps({
        "bench": {
            "captured_at": "2026-05-26T23:18:47Z",
            "duration_effective_s": 400.5,
        },
    }))
    start, end = read_sweep_window_from_bundle(bundle)
    assert start.tzinfo is not None
    assert (end - start).total_seconds() == pytest.approx(400.5)


def test_read_sweep_window_missing_file(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError):
        read_sweep_window_from_bundle(bundle)


# ---------------------------------------------------------------------------
# G. write_correlation
# ---------------------------------------------------------------------------


def test_write_correlation_writes_dcgm_correlation_json(tmp_path):
    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(_START.timestamp(), 0.5)])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    result = correlate(inp, _CEILINGS, client)

    out = write_correlation(result, inp.cell_dir)
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["schema_version"] == 1
    assert "dcgm" in data["captured_sources"]
    assert "resources" in data
    assert "queries" in data


# ---------------------------------------------------------------------------
# H. cross_attribute_zymtrace (Phase B Level-2 cross-attribution)
# ---------------------------------------------------------------------------


_KERNELS_PAYLOAD = {
    "schema_version": 1,
    "captured_sources": ["zymtrace"],
    "top_kernels": [],
    "per_gpu": [],
    "per_category": {
        # 30% NCCL, 20% MoE, 25% FMHA, 15% BMM-NVFP4, 10% cuBLAS
        "NCCL": 30,
        "MoE": 20,
        "FMHA": 25,
        "BMM-NVFP4": 15,
        "cuBLAS": 10,
    },
    "top_python_during_cuda": [],
}


def _make_workload_result(tmp_path: Path) -> "object":
    """Helper: produce a populated DcgmCorrelationResult by running correlate
    against canned series for HBM (0.5 ratio) + NVLink (0.9 TB/s/GPU) +
    Tensor pipe (0.2 ratio = 1800 TFLOPS/GPU avg) across 120s on 8 GPUs."""
    dram = [
        _series("DCGM_FI_PROF_DRAM_ACTIVE", str(i),
                [(_START.timestamp(), 0.5), (_START.timestamp() + 60, 0.5),
                 (_START.timestamp() + 120, 0.5)])
        for i in range(8)
    ]
    nvl = [
        _series("nvl_combined", str(i),
                [(_START.timestamp() + t, 0.9e12) for t in range(0, 121, 15)])
        for i in range(8)
    ]
    tensor = [
        _series("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", str(i),
                [(_START.timestamp(), 0.2), (_START.timestamp() + 60, 0.2),
                 (_START.timestamp() + 120, 0.2)])
        for i in range(8)
    ]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram[:1]},
        range_responses={
            "DCGM_FI_PROF_DRAM_ACTIVE": dram,
            "DCGM_FI_PROF_NVLINK": nvl,
            "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": tensor,
        },
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    return correlate(inp, _CEILINGS, client)


def test_cross_attribute_zymtrace_happy_path(tmp_path):
    """Per-category bytes attribute by time-share; %SoL math holds."""
    result = _make_workload_result(tmp_path)
    rows = cross_attribute_zymtrace(result, _KERNELS_PAYLOAD, _CEILINGS)

    cats = {r.category for r in rows}
    assert cats == {"NCCL", "MoE", "FMHA", "BMM-NVFP4", "cuBLAS"}

    nccl = next(r for r in rows if r.category == "NCCL")
    assert nccl.bound == "bandwidth"
    assert nccl.ceiling_metric == "nvlink5_tbps"
    assert nccl.time_share_pct == pytest.approx(30.0, rel=1e-3)
    # NCCL goes against NVLink (0.9 TB/s/GPU * 120s * 8 GPUs avg = 864 GB total in window).
    # NCCL attributed share = 30% of that.
    # During NCCL window (30% of 120s = 36s on 8 GPUs):
    #   effective_bw_during_category_window = attributed_bytes / 36s / 8 GPUs
    #                                      = 0.30 * 0.9e12*120*8 / 36 / 8 = 0.9e12 bytes/s/GPU
    # That equals the workload-avg NVLink BW (since time-share weight cancels).
    # %SoL_BW = 0.9e12 / 1.8e12 = 50%.
    assert nccl.sol_pct_bw == pytest.approx(50.0, rel=1e-2)

    bmm = next(r for r in rows if r.category == "BMM-NVFP4")
    assert bmm.bound == "compute"
    assert bmm.ceiling_metric == "nvfp4_dense_pflops"
    # Compute: avg 0.2 ratio of NVFP4 9 PFLOPS = 1800 TFLOPS/GPU.
    # During BMM window the effective_tflops resolves to the same 1800 TFLOPS (time-share cancels).
    # %SoL_compute = 1800/9000 = 20%.
    assert bmm.sol_pct_compute == pytest.approx(20.0, rel=1e-2)


def test_cross_attribute_zymtrace_empty_per_category(tmp_path):
    """kernels.json with empty per_category -> empty attribution list."""
    result = _make_workload_result(tmp_path)
    empty = dict(_KERNELS_PAYLOAD)
    empty["per_category"] = {}
    rows = cross_attribute_zymtrace(result, empty, _CEILINGS)
    assert rows == []


def test_cross_attribute_zymtrace_handles_unmapped_category(tmp_path):
    """Categories not in category_ceiling_map still emit a row with bound=None."""
    result = _make_workload_result(tmp_path)
    kernels = {
        "per_category": {"NCCL": 50, "ExoticUnknownCategory": 50},
    }
    rows = cross_attribute_zymtrace(result, kernels, _CEILINGS)
    cats = {r.category for r in rows}
    assert cats == {"NCCL", "ExoticUnknownCategory"}
    unknown = next(r for r in rows if r.category == "ExoticUnknownCategory")
    assert unknown.bound is None
    assert unknown.ceiling_metric is None


def test_correlate_with_kernels_json_path_populates_attribution(tmp_path):
    """correlate(kernels_json_path=...) writes per_category_attribution + zymtrace source."""
    # Write a kernels.json
    kj = tmp_path / "kernels.json"
    kj.write_text(json.dumps(_KERNELS_PAYLOAD))

    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(_START.timestamp(), 0.5),
                                                       (_START.timestamp() + 120, 0.5)])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client, kernels_json_path=kj)

    assert "zymtrace" in result.captured_sources
    assert result.kernels_json_path == str(kj.resolve())
    assert len(result.per_category_attribution) >= 1
    # Categories from the kernels.json
    cats = {a.category for a in result.per_category_attribution}
    assert "NCCL" in cats


def test_correlate_without_kernels_json_path_no_attribution(tmp_path):
    """correlate() without kernels_json_path -> empty per_category_attribution."""
    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(_START.timestamp(), 0.5)])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client)
    assert result.per_category_attribution == []
    assert result.kernels_json_path is None


def test_correlate_captures_power_watts_per_gpu(tmp_path):
    """Live correlate() reads DCGM_FI_DEV_POWER_USAGE -> mean per-GPU watts."""
    power = [_series("DCGM_FI_DEV_POWER_USAGE", "0",
                     [(_START.timestamp(), 650.0), (_START.timestamp() + 15, 700.0),
                      (_START.timestamp() + 30, 750.0)])]
    client = _FakePromClient(range_responses={"DCGM_FI_DEV_POWER_USAGE": power})
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client)
    assert result.power_watts_per_gpu == pytest.approx(700.0)  # mean(650,700,750)
    assert any(q["metric"] == "DCGM_FI_DEV_POWER_USAGE" for q in result.queries)


def test_correlate_power_null_when_absent(tmp_path):
    """No power series -> power_watts_per_gpu stays None (best-effort)."""
    client = _FakePromClient()
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    result = correlate(inp, _CEILINGS, client)
    assert result.power_watts_per_gpu is None


def test_correlate_captures_nodes_and_pod_scope(tmp_path):
    """Live correlate() records the node(s) from the DCGM series labels +
    the namespace/pod_label_selector from inputs (re-query provenance)."""
    dram = [
        TimeSeries(
            metric="DCGM_FI_PROF_DRAM_ACTIVE",
            labels={"gpu": "0", "Hostname": "g55dc2e"},
            samples=[(_START.timestamp(), 0.5)],
        ),
        TimeSeries(
            metric="DCGM_FI_PROF_DRAM_ACTIVE",
            labels={"gpu": "1", "Hostname": "g55dc2e"},
            samples=[(_START.timestamp(), 0.5)],
        ),
    ]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client)
    assert result.nodes == ["g55dc2e"]  # distinct + sorted
    assert result.namespace == "inference"
    assert result.pod_label_selector == "app=basic-inference"
    # Serialized into dcgm_correlation.json.
    d = result.to_dict()
    assert d["nodes"] == ["g55dc2e"]
    assert d["pod_label_selector"] == "app=basic-inference"


def test_correlate_nodes_empty_when_no_host_label(tmp_path):
    """No recognizable host label -> nodes stays empty (best-effort, no raise)."""
    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(_START.timestamp(), 0.5)])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    result = correlate(inp, _CEILINGS, client)
    assert result.nodes == []


def test_correlate_from_frozen_preserves_nodes(tmp_path):
    """A frozen YAML carrying nodes/namespace/pod_label_selector round-trips
    into the result so offline re-runs keep the re-query provenance."""
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump({
        "schema_version": 1,
        "hw_key": "b200_sm100",
        "n_gpus": 8,
        "sweep_start_utc": "2026-05-27T13:30:00Z",
        "sweep_end_utc": "2026-05-27T13:32:00Z",
        "data_tier": "prof",
        "scrape_interval_s": 30.0,
        "nodes": ["g55dc2e", "g04ade0"],
        "namespace": "<slurm-namespace>",
        "pod_label_selector": "exported_pod=glm51-baseline-e2e-lat-x",
        "resources": [
            {
                "peak_key": "hbm3e_tbps",
                "metric": "DCGM_FI_PROF_DRAM_ACTIVE",
                "unit": "ratio",
                "measured_avg": 0.05,
            }
        ],
    }))
    result = correlate_from_frozen(frozen, _CEILINGS)
    assert result.nodes == ["g55dc2e", "g04ade0"]
    assert result.namespace == "<slurm-namespace>"
    assert result.pod_label_selector == "exported_pod=glm51-baseline-e2e-lat-x"


def test_correlate_with_missing_kernels_json_path_graceful(tmp_path):
    """correlate() with a non-existent kernels.json -> graceful (no attribution, no raise)."""
    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(_START.timestamp(), 0.5)])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client, kernels_json_path=tmp_path / "does_not_exist.json")
    assert result.per_category_attribution == []
    assert result.kernels_json_path is None


def test_correlate_dry_run_with_kernels_json_skips_attribution(tmp_path):
    """dry_run=True skips the cross-attribution step (no real DCGM data anyway)."""
    kj = tmp_path / "kernels.json"
    kj.write_text(json.dumps(_KERNELS_PAYLOAD))
    client = _FakePromClient()
    inp = _inputs(tmp_path, start=_START, duration_s=120)

    result = correlate(inp, _CEILINGS, client, dry_run=True, kernels_json_path=kj)
    assert result.dry_run is True
    assert result.per_category_attribution == []


def test_dcgm_correlation_json_includes_attribution_block(tmp_path):
    """write_correlation persists the per_category_attribution list."""
    kj = tmp_path / "kernels.json"
    kj.write_text(json.dumps(_KERNELS_PAYLOAD))

    dram = [_series("DCGM_FI_PROF_DRAM_ACTIVE", "0", [(_START.timestamp(), 0.5),
                                                       (_START.timestamp() + 120, 0.5)])]
    nvl = [_series("nvl", "0", [(_START.timestamp() + t, 0.9e12) for t in range(0, 121, 15)])]
    client = _FakePromClient(
        instant_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram},
        range_responses={"DCGM_FI_PROF_DRAM_ACTIVE": dram,
                          "DCGM_FI_PROF_NVLINK": nvl},
    )
    inp = _inputs(tmp_path, start=_START, duration_s=120)
    result = correlate(inp, _CEILINGS, client, kernels_json_path=kj)

    out = write_correlation(result, inp.cell_dir)
    data = json.loads(out.read_text())
    assert "per_category_attribution" in data
    assert isinstance(data["per_category_attribution"], list)
    assert len(data["per_category_attribution"]) >= 1
    assert "kernels_json_path" in data
