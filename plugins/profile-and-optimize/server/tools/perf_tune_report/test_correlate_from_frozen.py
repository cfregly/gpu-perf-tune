"""Tests for ``correlate_from_frozen()`` (v1.23.1).

Validates the frozen-YAML -> ``DcgmCorrelationResult`` path used when
DCGM measurements are captured interactively (e.g. via the
the Prometheus MCP-mcp ``query_prometheus`` tool) and need to be folded
into a ``dcgm_correlation.json`` without re-running the live
``correlate()`` pipeline.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
import yaml as _yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.dcgm_correlate import (
    FrozenYamlMalformed,
    correlate_from_frozen,
    write_correlation,
)


_MIN_CEILINGS: dict = {
    "b200_sm100": {
        "hw_name": "NVIDIA B200 (test)",
        "nvfp4_dense_pflops": {"value": 9.0, "units": "PFLOPS", "source": "test"},
        "fp8_dense_pflops": {"value": 4.5, "units": "PFLOPS", "source": "test"},
        "bf16_dense_pflops": {"value": 2.25, "units": "PFLOPS", "source": "test"},
        "hbm3e_tbps": {"value": 8.0, "units": "TB/s", "source": "test"},
        "nvlink5_tbps": {"value": 1.8, "units": "TB/s", "source": "test"},
    },
    "category_ceiling_map": {
        "NCCL": {"bound": "bandwidth", "metric": "nvlink5_tbps"},
        "MoE": {"bound": "compute", "metric": "nvfp4_dense_pflops"},
        "FMHA": {"bound": "bandwidth", "metric": "hbm3e_tbps"},
        "BMM-NVFP4": {"bound": "compute", "metric": "nvfp4_dense_pflops"},
        "Triton-fused": {"bound": "compute", "metric": "bf16_dense_pflops"},
        "cuBLAS": {"bound": "compute", "metric": "bf16_dense_pflops"},
        "Elementwise": {"bound": "bandwidth", "metric": "hbm3e_tbps"},
        "Other": {"bound": "bandwidth", "metric": "hbm3e_tbps"},
    },
}


def _write_frozen(path: Path, *, resources: list[dict] | None = None, hw_key: str = "b200_sm100") -> Path:
    """Write a minimally-valid frozen YAML and return its path."""
    if resources is None:
        resources = [
            {
                "peak_key": "hbm3e_tbps",
                "metric": "DCGM_FI_PROF_DRAM_ACTIVE",
                "unit": "ratio",
                "measured_avg": 0.155,
                "promql": "avg by (gpu) (DCGM_FI_PROF_DRAM_ACTIVE{...})",
            },
            {
                "peak_key": "nvfp4_dense_pflops",
                "metric": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
                "unit": "ratio",
                "measured_avg": 0.018,
            },
            {
                "peak_key": "nvlink5_tbps",
                "metric": "sum(NVLINK_TX_BYTES, NVLINK_RX_BYTES)",
                "unit": "bytes_per_s",
                "measured_avg": 400e6,
            },
        ]
    payload = {
        "schema_version": 1,
        "hw_key": hw_key,
        "n_gpus": 8,
        "sweep_start_utc": "2026-05-27T13:29:00Z",
        "sweep_end_utc": "2026-05-27T13:52:00Z",
        "captured_at": "2026-05-28T00:00:00Z",
        "captured_by": "test@workstation",
        "data_tier": "prof",
        "scrape_interval_s": 30.0,
        "notes": "test fixture",
        "resources": resources,
    }
    path.write_text(_yaml.dump(payload))
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_correlate_from_frozen_emits_three_resources(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS)
    assert result.schema_version == 1
    assert result.hw_key == "b200_sm100"
    assert result.n_gpus == 8
    assert result.duration_s == 1380.0  # 23 minutes
    assert result.dcgm_group_level == "prof"
    assert len(result.resources) == 3
    peak_keys = {r.peak_key for r in result.resources}
    assert peak_keys == {"hbm3e_tbps", "nvfp4_dense_pflops", "nvlink5_tbps"}


def test_correlate_from_frozen_ratio_bandwidth_math(tmp_path: Path):
    """DRAM_ACTIVE=0.155 vs hbm3e_tbps=8.0 -> 15.5% SoL."""
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS)
    hbm = next(r for r in result.resources if r.peak_key == "hbm3e_tbps")
    assert hbm.sol_pct == pytest.approx(15.5)
    # bytes/s = 0.155 * 8.0 * 1e12 = 1.24e12
    assert hbm.measured_bytes_per_s == pytest.approx(1.24e12)
    # bytes_total = 1.24e12 * 1380s * 8 gpus = 1.36896e16
    assert hbm.measured_bytes_total == pytest.approx(1.24e12 * 1380.0 * 8)
    assert hbm.measured_tflops_avg is None


def test_correlate_from_frozen_ratio_compute_math(tmp_path: Path):
    """TENSOR_ACTIVE=0.018 vs nvfp4_dense_pflops=9.0 PFLOPS -> 1.8% SoL, 162 TFLOPS."""
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS)
    nvfp4 = next(r for r in result.resources if r.peak_key == "nvfp4_dense_pflops")
    assert nvfp4.sol_pct == pytest.approx(1.8)
    # achieved_tflops = 0.018 * 9.0 * 1000 = 162
    assert nvfp4.measured_tflops_avg == pytest.approx(162.0)
    assert nvfp4.measured_bytes_per_s is None
    assert nvfp4.measured_bytes_total is None


def test_correlate_from_frozen_bytes_per_s_math(tmp_path: Path):
    """NVLINK 400 MB/s/GPU vs nvlink5_tbps=1.8 TB/s -> 0.022% SoL."""
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS)
    nvl = next(r for r in result.resources if r.peak_key == "nvlink5_tbps")
    # 400e6 / (1.8 * 1e12) = 2.222e-4 -> 0.0222%
    assert nvl.sol_pct == pytest.approx((400e6 / (1.8 * 1e12)) * 100.0)
    assert nvl.measured_bytes_per_s == pytest.approx(400e6)
    assert nvl.measured_tflops_avg is None


def test_correlate_from_frozen_promql_recorded_in_queries(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS)
    promqls = {q["peak_key"]: q["promql"] for q in result.queries}
    assert "DCGM_FI_PROF_DRAM_ACTIVE" in promqls["hbm3e_tbps"]
    # Resource without promql -> empty string (no crash)
    assert promqls["nvfp4_dense_pflops"] == ""


def test_correlate_from_frozen_writes_dcgm_correlation_json_round_trip(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS)
    out = write_correlation(result, tmp_path)
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["hw_key"] == "b200_sm100"
    assert payload["n_gpus"] == 8
    assert payload["dcgm_group_level"] == "prof"
    assert len(payload["resources"]) == 3


# ---------------------------------------------------------------------------
# Cross-attribution with zymtrace kernels.json
# ---------------------------------------------------------------------------


def test_correlate_from_frozen_no_kernels_json_means_empty_attribution(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS, cell_dir=tmp_path)
    assert result.per_category_attribution == []
    # captured_sources should NOT include zymtrace because no kernels.json
    assert "zymtrace" in result.captured_sources  # cell_dir was supplied
    # but per_category_attribution remains empty since kernels.json wasn't found
    assert result.kernels_json_path is None


def test_correlate_from_frozen_with_kernels_json_populates_attribution(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml")
    # Build a minimal kernels.json that cross_attribute_zymtrace can consume.
    # Schema: per_category is a dict {category_name -> sample_count}.
    kernels_payload = {
        "schema_version": 1,
        "per_category": {
            "Elementwise": 500,
            "FMHA": 300,
            "Other": 200,
        },
    }
    kpath = tmp_path / "kernels.json"
    kpath.write_text(json.dumps(kernels_payload))
    result = correlate_from_frozen(yaml_path, _MIN_CEILINGS, kernels_json_path=kpath)
    assert result.kernels_json_path == str(kpath)
    assert len(result.per_category_attribution) == 3
    cats = {a.category for a in result.per_category_attribution}
    assert cats == {"Elementwise", "FMHA", "Other"}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_correlate_from_frozen_raises_on_missing_yaml(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        correlate_from_frozen(tmp_path / "missing.yaml", _MIN_CEILINGS)


def test_correlate_from_frozen_raises_on_malformed_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("{not: [valid")
    with pytest.raises(FrozenYamlMalformed, match="YAML parse error"):
        correlate_from_frozen(bad, _MIN_CEILINGS)


def test_correlate_from_frozen_raises_on_missing_required_key(tmp_path: Path):
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(_yaml.dump({"hw_key": "b200_sm100"}))
    with pytest.raises(FrozenYamlMalformed, match="missing required key"):
        correlate_from_frozen(incomplete, _MIN_CEILINGS)


def test_correlate_from_frozen_raises_on_unknown_hw_key(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml", hw_key="ada_lovelace_99")
    with pytest.raises(FrozenYamlMalformed, match="not present in ceilings"):
        correlate_from_frozen(yaml_path, _MIN_CEILINGS)


def test_correlate_from_frozen_raises_on_empty_resources(tmp_path: Path):
    yaml_path = _write_frozen(tmp_path / "frozen.yaml", resources=[])
    with pytest.raises(FrozenYamlMalformed, match="resources must be a non-empty list"):
        correlate_from_frozen(yaml_path, _MIN_CEILINGS)


def test_correlate_from_frozen_raises_on_unsupported_unit(tmp_path: Path):
    yaml_path = _write_frozen(
        tmp_path / "frozen.yaml",
        resources=[
            {
                "peak_key": "hbm3e_tbps",
                "metric": "DCGM_FI_DEV_GPU_TEMP",
                "unit": "celsius",
                "measured_avg": 65.0,
            },
        ],
    )
    with pytest.raises(ValueError, match="unsupported unit"):
        correlate_from_frozen(yaml_path, _MIN_CEILINGS)
