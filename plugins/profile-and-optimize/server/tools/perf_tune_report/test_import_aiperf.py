"""Unit tests for the AIPerf-export bundle importer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_tune_report.importers.aiperf_export import (
    AiperfImportResult,
    detect_aiperf_bundle,
    import_aiperf_bundle,
    _parse_aiperf_csv,
)
from tools.perf_tune_report.importers import import_bundle_auto

# Minimal AIPerf profile_export_aiperf.csv (two sections), matching the real
# 2026-05-31 campaign export shape.
_CSV_FULL = """Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
Time to First Token (ms),165.73,56.5,2506,11601,57,60,61,76,153,170,189,196,985,287
Inter Token Latency (ms),12.57,11.8,13.0,879,11.8,11.8,12.4,12.5,12.6,12.7,12.9,12.9,13.0,0.26
Output Token Throughput Per User (tokens/sec/user),79.61,77.1,84.8,5573,77,77,78,79,80,80,81,84,85,1.67

Metric,Value
Benchmark Duration (sec),402.13
Output Token Throughput (tokens/sec),42.19
Overall Usage Prompt Cache Read % (%),93.59
Request Count,70.00
Request Throughput (requests/sec),0.17
"""

# A "partial" cell: only 21 of an expected 284 requests completed (the MTP-K2
# c=8 abort case).
_CSV_PARTIAL = _CSV_FULL.replace("Request Count,70.00", "Request Count,21.00")


def _make_variant(tmp_path: Path) -> Path:
    v = tmp_path / "mtpk3"
    (v / "c1").mkdir(parents=True)
    (v / "c8").mkdir(parents=True)
    (v / "c1" / "profile_export_aiperf.csv").write_text(_CSV_FULL)
    (v / "c8" / "profile_export_aiperf.csv").write_text(_CSV_PARTIAL)
    return v


def test_detect(tmp_path: Path) -> None:
    v = _make_variant(tmp_path)
    assert detect_aiperf_bundle(v) is True
    assert detect_aiperf_bundle(tmp_path / "nope") is False


def test_parse_csv(tmp_path: Path) -> None:
    v = _make_variant(tmp_path)
    m = _parse_aiperf_csv(v / "c1" / "profile_export_aiperf.csv")
    assert m is not None
    assert m["ttft_avg_ms"] == pytest.approx(165.73)
    assert m["output_tps_per_user"] == pytest.approx(79.61)
    assert m["request_throughput_avg"] == pytest.approx(0.17)
    assert m["output_tps_total"] == pytest.approx(42.19)
    assert m["request_count"] == pytest.approx(70.0)


def test_import_and_status(tmp_path: Path) -> None:
    v = _make_variant(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    res = import_aiperf_bundle(
        v, campaign, overrides={"model": "zai-org/GLM-5.1", "mtp": True}
    )
    assert isinstance(res, AiperfImportResult)
    assert res.cell_id == "mtpk3"
    assert res.concurrencies == [1, 8]
    # c=8 has 21/284 reqs -> partial; overall status partial.
    assert res.partial_cells == [8]
    assert res.status == "partial"
    rows = json.loads(res.normalized_path.read_text())
    by_c = {r["concurrency"]: r for r in rows}
    assert by_c[1]["status"] == "full"
    assert by_c[8]["status"] == "partial"
    assert by_c[1]["backend"] == "aiperf"
    assert by_c[1]["ttft_avg_ms"] == pytest.approx(165.73)
    assert by_c[1]["output_tps_per_gpu"] == pytest.approx(42.19 / 8)
    assert by_c[1]["mtp"] is True


def test_auto_dispatch(tmp_path: Path) -> None:
    v = _make_variant(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    res = import_bundle_auto(v, campaign, overrides={"model": "zai-org/GLM-5.1"})
    assert isinstance(res, AiperfImportResult)
    assert res.row_count == 2


def test_requires_model(tmp_path: Path) -> None:
    v = _make_variant(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with pytest.raises(ValueError, match="model is required"):
        import_aiperf_bundle(v, campaign, overrides={})


# v1.42.0 carry-through: AIPerf prompt-cache -> prefix_cache_hit_rate, ISL/OSL,
# total throughput, declared cache_mode.
_CSV_SHAPE = """Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
Time to First Token (ms),165.73,56.5,2506,11601,57,60,61,76,153,170,189,196,985,287
Input Sequence Length (tokens),3200,3000,3400,224000,3000,3000,3100,3150,3200,3250,3300,3350,3400,80
Output Sequence Length (tokens),512,500,520,35840,500,500,505,510,512,515,518,519,520,5

Metric,Value
Benchmark Duration (sec),402.13
Output Token Throughput (tokens/sec),42.19
Total Token Throughput (tokens/sec),305.0
Overall Usage Prompt Cache Read % (%),93.59
Request Count,70.00
Request Throughput (requests/sec),0.17
"""


def test_aiperf_isl_osl_total_prefix_cache_and_cache_mode(tmp_path: Path) -> None:
    v = tmp_path / "shape"
    (v / "c1").mkdir(parents=True)
    (v / "c1" / "profile_export_aiperf.csv").write_text(_CSV_SHAPE)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    res = import_aiperf_bundle(
        v, campaign,
        overrides={"model": "zai-org/GLM-5.1", "tensor_parallel": 8, "cache_mode": "warm"},
    )
    row = json.loads(res.normalized_path.read_text())[0]
    assert row["mean_input_tokens"] == pytest.approx(3200.0)
    assert row["mean_output_tokens"] == pytest.approx(512.0)
    assert row["total_tps_per_gpu"] == pytest.approx(305.0 / 8)
    # 93.59% prompt-cache read -> 0.9359 prefix_cache_hit_rate
    assert row["prefix_cache_hit_rate"] == pytest.approx(0.9359)
    assert row["cache_mode"] == "warm"
