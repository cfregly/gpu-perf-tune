"""Unit + smoke tests for the perf_tune_report publish_to_lake verb.

Coverage:

- ``test_parse_campaign_utc_*``    -- UTC suffix extraction from campaign dir name
- ``test_build_atlas_table_*``     -- pyarrow schema stability + row mapping
- ``test_build_campaign_row_*``    -- single-row provenance table from SOURCE.md
- ``test_extra_dict_*``            -- JSON encoding keeps Iceberg schema stable
- ``test_s3_key_*``                -- canonical Hive-style key layout
- ``test_publish_dry_run_*``       -- dry-run never calls boto3.put_object
- ``test_upload_if_exists_*``      -- fail / skip / overwrite semantics
- ``test_cli_*``                   -- CLI ack-gating + JSON envelope shape
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.lake_writer import (
    ATLAS_TABLE_NAME,
    S3_PREFIX,
    CAMPAIGN_TABLE_NAME,
    IF_EXISTS_FAIL,
    IF_EXISTS_OVERWRITE,
    IF_EXISTS_SKIP,
    SOL_TABLE_NAME,
    TPM_TABLE_NAME,
    COST_TABLE_NAME,
    QUALITY_TABLE_NAME,
    CHAMPION_TABLE_NAME,
    ROOFLINE_TABLE_NAME,
    S3Config,
    CampaignIncompleteError,
    build_atlas_table,
    build_campaign_row,
    build_sol_table,
    build_tpm_table,
    build_cost_table,
    build_quality_table,
    parse_campaign_utc,
    parse_source_md,
    publish,
    resolve_s3_config,
    s3_key_for,
    upload_to_s3,
)
from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.schema import AtlasCell


pytest.importorskip("pyarrow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atlas_row(**overrides: Any) -> AtlasCell:
    base = dict(
        cell_id="cellA",
        model="GLM-5.1-NVFP4",
        hardware="B200",
        quant="NVFP4",
        tensor_parallel=8,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=12288,
        concurrency=1,
        status="full",
        ttft_avg_ms=182.04,
        request_throughput_avg=0.14,
        output_tps_per_user=75.24,
        output_tps_per_gpu=9.15,
        # Full-context descriptor (2026-06-07): complete by default so a row built here
        # is a fully-described measured row; tests that exercise the descriptor gate
        # override a single field to its sentinel ("unknown"/None). ISL/OSL added 2026-06-08
        # so the default complete row also clears the vllm-bench ISL/OSL shape gate.
        mean_input_tokens=4096.0,
        mean_output_tokens=512.0,
        dataset="random",
        cudagraph_mode="full",
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8_e4m3",
        image="vllm-test:v0",
        data_parallel=1,
        pipeline_parallel=1,
        backend="vllm-sweep",
        raw_path="raw/sweep-c1.txt",
        captured_at="2026-05-25T06:18Z..08:14Z",
        notes="",
        extra={"max_num_seqs": 16},
    )
    base.update(overrides)
    return AtlasCell(**base)


def _stage_campaign(tmp_path: Path, *, campaign_slug: str = "test-20260525T081650Z") -> Path:
    """Lay down a minimal campaign directory with atlas.jsonl + SOURCE.md + config.yaml."""
    campaign_dir = tmp_path / campaign_slug
    campaign_dir.mkdir(parents=True)
    rows = [
        _make_atlas_row(concurrency=1),
        _make_atlas_row(concurrency=2, ttft_avg_ms=264.31, output_tps_per_gpu=18.81),
        _make_atlas_row(cell_id="cellB", concurrency=1, extra={"block_size": 128}),
    ]
    atlas_path = campaign_dir / "atlas.jsonl"
    with atlas_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    (campaign_dir / "SOURCE.md").write_text(
        "# Campaign\n\n"
        "- captured_at: 20260525T081650Z\n"
        "- config: /tmp/test.yaml\n"
        "- operator: security@example.com\n"
        "- cluster_context: <cluster>\n"
    )
    (campaign_dir / "config.yaml").write_text(
        "name: test\ncells: []\nnext_lever: 'batching -> dispatch-depth (HYPOTHESIS)'\n"
    )
    # A rendered, complete campaign by default so the publish completeness
    # gate passes; tests that exercise the gate stage their own status.
    (campaign_dir / "report_status.json").write_text(
        json.dumps(
            {
                "sol_complete": True,
                "plot_ready_points": 3,
                "non_plot_ready_full_cells": 0,
                "rendered_pages": ["scatter (page 1)", "heatmap (page 2)"],
                "omitted_pages": [],
            }
        )
    )
    return campaign_dir


# ---------------------------------------------------------------------------
# Provenance parsing
# ---------------------------------------------------------------------------


def test_parse_campaign_utc_strips_slug_prefix():
    assert parse_campaign_utc("glm51-phase6-20260525T081650Z") == datetime(
        2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc
    )


def test_parse_campaign_utc_handles_utc_prefix():
    # Canonical campaign_init layout is "<UTC>-<slug>" (timestamp first).
    assert parse_campaign_utc("20260529T175706Z-glm51-deepep-sweep") == datetime(
        2026, 5, 29, 17, 57, 6, tzinfo=timezone.utc
    )


def test_parse_campaign_utc_rejects_missing_suffix():
    with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
        parse_campaign_utc("not-a-real-campaign")


def test_parse_source_md_extracts_operator_and_cluster(tmp_path: Path):
    src = tmp_path / "SOURCE.md"
    src.write_text(
        "# Header\n\n"
        "- operator: security@example.com\n"
        "- cluster_context: <cluster>\n"
        "- noise without colon\n"
        "  - nested item: ignored at top level\n"
    )
    md = parse_source_md(src)
    assert md["operator"] == "security@example.com"
    assert md["cluster_context"] == "<cluster>"


def test_parse_source_md_handles_missing_file(tmp_path: Path):
    assert parse_source_md(tmp_path / "does-not-exist.md") == {}


# ---------------------------------------------------------------------------
# Parquet schema construction
# ---------------------------------------------------------------------------


def test_build_atlas_table_schema_stable_across_extra_keys():
    """Heterogeneous `extra` keys must not change the parquet schema."""
    rows = [
        _make_atlas_row(extra={"max_num_seqs": 16}),
        _make_atlas_row(extra={"block_size": 128, "max_num_seqs": 32}),
        _make_atlas_row(extra={}),
    ]
    table = build_atlas_table(rows, campaign_id="campaign-20260525T081650Z")
    names = [f.name for f in table.schema]
    assert names == [
        "campaign_id", "cell_id", "model", "hardware", "quant",
        "tensor_parallel", "parallel_strategy", "mtp",
        "max_num_batched_tokens", "concurrency", "status",
        "ttft_avg_ms", "request_throughput_avg",
        "output_tps_per_user", "output_tps_per_gpu",
        "total_tps_per_gpu",
        "tpot_median_ms", "itl_avg_ms",
        "mean_input_tokens", "mean_output_tokens",
            "prefix_cache_hit_rate", "cache_mode",
            "dataset", "cudagraph_mode", "gpu_memory_utilization",
            "kv_cache_dtype", "image", "data_parallel", "pipeline_parallel",
            "num_speculative_tokens", "async_scheduling", "max_num_seqs",
            "enable_prefix_caching", "bench_backend", "variant_key",
            "backend", "serving_engine",
            "router_policy", "prefix_reuse", "per_replica_cache_hit", "acceptance_length",
            "spec_accept_rate",
            "kv_cache_tokens", "ep_mode", "dcgm_sm_active", "dcgm_dram_active", "dcgm_tensor_active",
            "raw_path", "captured_at", "notes", "extra_json",
        ]
    assert table.num_rows == 3
    # Serving-variant descriptor columns (2026-06-07) are present + variant_key populated.
    assert "variant_key" in names and "num_speculative_tokens" in names
    vt = build_atlas_table(
        [_make_atlas_row(num_speculative_tokens=3, async_scheduling=True,
                         max_num_seqs=192, enable_prefix_caching=True, bench_backend="vllm")],
        campaign_id="c-20260525T081650Z",
    )
    assert vt.column("num_speculative_tokens").to_pylist() == [3]
    assert vt.column("async_scheduling").to_pylist() == [True]
    assert vt.column("max_num_seqs").to_pylist() == [192]
    assert vt.column("bench_backend").to_pylist() == ["vllm"]
    # variant_key is the capture_signature hash (non-empty, deterministic).
    vk = vt.column("variant_key").to_pylist()[0]
    assert vk and len(vk) == 64
    # Every row has a stringified JSON in extra_json.
    extras = table.column("extra_json").to_pylist()
    assert json.loads(extras[0]) == {"max_num_seqs": 16}
    assert json.loads(extras[1]) == {"block_size": 128, "max_num_seqs": 32}
    assert json.loads(extras[2]) == {}


def test_build_atlas_table_serving_engine_derived_from_backend():
    """serving_engine is normalized from backend: vllm-sweep->vllm,
    sglang-sweep->sglang, trtllm->trtllm, aiperf->'' (load-gen client)."""
    rows = [
        _make_atlas_row(backend="vllm-sweep"),
        _make_atlas_row(backend="sglang-sweep"),
        _make_atlas_row(backend="trtllm"),
        _make_atlas_row(backend="aiperf"),
    ]
    table = build_atlas_table(rows, campaign_id="x-20260525T081650Z")
    assert table.column("serving_engine").to_pylist() == ["vllm", "sglang", "trtllm", ""]
    # Explicit serving_engine is respected (not overwritten by the backend map).
    explicit = build_atlas_table(
        [_make_atlas_row(backend="aiperf", serving_engine="vllm")],
        campaign_id="x-20260525T081650Z",
    )
    assert explicit.column("serving_engine").to_pylist() == ["vllm"]


def test_build_atlas_table_preserves_nullable_metrics():
    rows = [_make_atlas_row(status="failed", ttft_avg_ms=None, request_throughput_avg=None)]
    table = build_atlas_table(rows, campaign_id="x-20260525T081650Z")
    assert table.column("ttft_avg_ms").to_pylist() == [None]
    assert table.column("request_throughput_avg").to_pylist() == [None]


def test_build_atlas_table_handles_none_max_num_batched_tokens():
    # A non-serving run (e.g. focus=accuracy / EAGLE3 acceptance) can carry
    # max_num_batched_tokens=None; build_atlas_table must not crash -- it records
    # 0 (the existing missing-shape sentinel) rather than raising on int(None).
    rows = [_make_atlas_row(max_num_batched_tokens=None)]
    table = build_atlas_table(rows, campaign_id="x-20260525T081650Z")
    assert table.column("max_num_batched_tokens").to_pylist() == [0]


def test_build_atlas_table_lifts_num_speculative_tokens_from_extra_and_note():
    """num_speculative_tokens is lifted from extra / note when the typed field is
    unset, so MTP-K arms recorded only in extra (historical / model-optimize A/B)
    do NOT collapse on the atlas grain. The typed field, when present, wins."""
    # Typed field wins over an extra value.
    typed = build_atlas_table(
        [_make_atlas_row(num_speculative_tokens=2, mtp=True, extra={"num_speculative_tokens": 9})],
        campaign_id="c-20260525T081650Z",
    )
    assert typed.column("num_speculative_tokens").to_pylist() == [2]
    # Lift from extra.num_speculative_tokens / extra.spec_decode_k.
    lifted = build_atlas_table(
        [
            _make_atlas_row(num_speculative_tokens=None, mtp=True, extra={"num_speculative_tokens": 2}),
            _make_atlas_row(num_speculative_tokens=None, mtp=True, extra={"spec_decode_k": 3}),
        ],
        campaign_id="c-20260525T081650Z",
    )
    assert lifted.column("num_speculative_tokens").to_pylist() == [2, 3]
    # Lift from a mtp-K<n> / mtp-num_spec<n> note (the historical glm51-gb300-bench shape).
    noted = build_atlas_table(
        [
            _make_atlas_row(num_speculative_tokens=None, mtp=True, extra={}, notes="sharegpt c=1 mtp-K3: 1.65x"),
            _make_atlas_row(num_speculative_tokens=None, mtp=True, extra={"note": "natural c=1 mtp-num_spec2"}),
        ],
        campaign_id="c-20260525T081650Z",
    )
    assert noted.column("num_speculative_tokens").to_pylist() == [3, 2]
    # A non-MTP baseline with no K signal stays None (must not collide with K arms).
    baseline = build_atlas_table(
        [_make_atlas_row(num_speculative_tokens=None, mtp=False, extra={}, notes="baseline-no-spec")],
        campaign_id="c-20260525T081650Z",
    )
    assert baseline.column("num_speculative_tokens").to_pylist() == [None]


def test_build_atlas_table_true_grain_is_unique_after_numspec_lift():
    """The atlas natural key (campaign_id, cell_id, concurrency, mean_input_tokens,
    mean_output_tokens, cache_mode, num_speculative_tokens) is unique across rows
    that share (cell_id, concurrency) but differ by prefill ISL, warm/cold, or
    MTP-K -- exactly the collapse the medallion grain fix guards against."""
    rows = [
        # Same cell + c=1, different prefill ISL (prefill sweep point).
        _make_atlas_row(cell_id="cellA", concurrency=1, mean_input_tokens=512.0),
        _make_atlas_row(cell_id="cellA", concurrency=1, mean_input_tokens=4096.0),
        # Same cell + c=256, warm vs cold re-measurement.
        _make_atlas_row(cell_id="cellA", concurrency=256, cache_mode="cold"),
        _make_atlas_row(cell_id="cellA", concurrency=256, cache_mode="warm"),
        # Same cell + c=1, MTP K2 vs K3 recorded only in extra.
        _make_atlas_row(cell_id="cellB", concurrency=1, mtp=True,
                        num_speculative_tokens=None, extra={"num_speculative_tokens": 2}),
        _make_atlas_row(cell_id="cellB", concurrency=1, mtp=True,
                        num_speculative_tokens=None, extra={"spec_decode_k": 3}),
    ]
    t = build_atlas_table(rows, campaign_id="c-20260525T081650Z")
    grain_cols = (
        "campaign_id", "cell_id", "concurrency", "mean_input_tokens",
        "mean_output_tokens", "cache_mode", "num_speculative_tokens",
    )
    cols = {n: t.column(n).to_pylist() for n in grain_cols}
    keys = list(zip(*(cols[n] for n in grain_cols)))
    assert len(keys) == len(set(keys)) == 6  # true grain: all distinct
    # ...whereas the legacy (campaign_id, cell_id, concurrency) key collapses them.
    legacy = list(zip(cols["campaign_id"], cols["cell_id"], cols["concurrency"]))
    assert len(set(legacy)) == 3 < len(keys)


def _quality_ts() -> datetime:
    return datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc)


def test_build_quality_table_long_format_flat_key_fallback():
    # A cell with metric_kind + heterogeneous flat accuracy keys (the existing
    # EAGLE3-sweep convention): long-format rows for the accuracy/loss keys only;
    # hyperparameters + non-numeric keys are excluded; a serving cell yields none.
    rows = [
        _make_atlas_row(
            cell_id="acc-gb16",
            extra={
                "metric_kind": "train_accuracy_proxy",
                "acc_proxy_24k_samples": 0.311,
                "loss_24k_samples": 3.454,
                "batch_size": 4,        # hyperparameter -> excluded
                "global_batch": 128,    # hyperparameter -> excluded
                "note": "eagle3",       # non-numeric -> excluded
            },
        ),
        _make_atlas_row(cell_id="serving", extra={"max_num_seqs": 16}),  # no metric_kind
    ]
    table = build_quality_table(
        rows, campaign_id="x-20260525T081650Z",
        captured_at_utc=_quality_ts(), published_at_utc=_quality_ts(),
    )
    assert set(table.column("metric_name").to_pylist()) == {"acc_proxy_24k_samples", "loss_24k_samples"}
    assert set(table.column("metric_kind").to_pylist()) == {"train_accuracy_proxy"}
    assert set(table.column("cell_id").to_pylist()) == {"acc-gb16"}


def test_build_quality_table_canonical_quality_metrics_subdict():
    # Canonical convention: extra["quality_metrics"]={name: value} -- only those
    # numeric values are emitted (hyperparameters outside the sub-dict ignored).
    rows = [_make_atlas_row(extra={
        "metric_kind": "acceptance",
        "quality_metrics": {"acceptance_length": 3.2, "draft_hit_rate": 0.61},
        "global_batch": 128,
    })]
    table = build_quality_table(
        rows, campaign_id="x-20260525T081650Z",
        captured_at_utc=_quality_ts(), published_at_utc=_quality_ts(),
    )
    assert set(table.column("metric_name").to_pylist()) == {"acceptance_length", "draft_hit_rate"}
    assert table.column("metric_value").to_pylist() == [3.2, 0.61] or set(table.column("metric_value").to_pylist()) == {3.2, 0.61}


def test_build_quality_table_empty_for_serving_only_campaign():
    rows = [_make_atlas_row(extra={"max_num_seqs": 16})]
    table = build_quality_table(
        rows, campaign_id="x-20260525T081650Z",
        captured_at_utc=_quality_ts(), published_at_utc=_quality_ts(),
    )
    assert table.num_rows == 0


_SOL_SCHEMA_COLUMNS = [
    "campaign_id", "cell_id", "category", "sol_level",
    "gpu_time_share_pct", "pct_sol", "bound", "ceiling_key",
    "ceiling_value", "ceiling_units", "measured_value", "measured_units",
    "attributed_bytes_total", "attributed_flops_total",
    "arithmetic_intensity_flops_per_byte", "kernel_name", "hw_key",
    "source_artifact", "source_artifact_sha256", "sol_ceilings_yaml_sha256",
    "captured_at_utc", "published_at_utc", "focus", "sol_rigor",
]


def _stage_sol_cells(campaign_dir: Path) -> None:
    """Add cells/cellA/{kernels.json,dcgm_correlation.json} SoL artifacts."""
    cell = campaign_dir / "cells" / "cellA"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "kernels.json").write_text(json.dumps({
        "schema_version": 1,
        "captured_sources": ["zymtrace"],
        "top_kernels": [],
        "per_gpu": [],
        "per_category": {"NCCL": 300, "MoE": 100, "FMHA": 100},
        "top_python_during_cuda": [],
    }))
    (cell / "dcgm_correlation.json").write_text(json.dumps({
        "schema_version": 1,
        "captured_sources": ["dcgm"],
        "hw_key": "b200_sm100",
        "sweep_start_utc": "2026-05-25T06:18:00Z",
        "sweep_end_utc": "2026-05-25T06:20:00Z",
        "duration_s": 120.0,
        "n_gpus": 8,
        "dcgm_group_level": "prof",
        "scrape_interval_s": 1.0,
        "short_sweep_warning": False,
        "resources": [
            {"peak_key": "hbm3e_tbps", "metric": "DCGM_FI_PROF_DRAM_ACTIVE",
             "is_fallback": False, "n_gpus": 8, "measured_bytes_total": 5.0e13,
             "measured_bytes_per_s": 4.32e11, "measured_tflops_avg": None,
             "peak_per_gpu": 8.0, "peak_per_gpu_units": "TB/s",
             "peak_aggregate": 64.0, "sol_pct": 5.4, "notes": []},
        ],
        "queries": [],
        "dry_run": False,
        "per_category_attribution": [
            {"category": "NCCL", "time_share_pct": 60.0,
             "attributed_bytes_total": 1.1e10, "attributed_flops_total": None,
             "effective_bw_during_category_window": 1.35e8,
             "effective_tflops_during_category_window": None,
             "sol_pct_bw": 0.0075, "sol_pct_compute": None,
             "bound": "bandwidth", "ceiling_metric": "nvlink5_tbps"},
        ],
    }))


def test_build_sol_table_schema_stable_and_levels(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    _stage_sol_cells(campaign_dir)
    rows = [_make_atlas_row()]
    table = build_sol_table(
        campaign_dir, campaign_dir.name, rows,
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
        focus="latency", sol_rigor="L3",
    )
    assert [f.name for f in table.schema] == _SOL_SCHEMA_COLUMNS
    levels = set(table.column("sol_level").to_pylist())
    assert {"L1", "L2", "L3"} <= levels  # no ncu fixture -> no L4
    d = table.to_pylist()
    l1 = [r for r in d if r["sol_level"] == "L1"]
    l3 = [r for r in d if r["sol_level"] == "L3"]
    # L1 carries time-share, never a %SoL (sample-share proxy only).
    assert all(r["pct_sol"] is None for r in l1)
    assert any(r["gpu_time_share_pct"] and r["gpu_time_share_pct"] > 0 for r in l1)
    # L3 carries a measured %SoL + ceiling.
    assert any(r["pct_sol"] == 5.4 for r in l3)
    assert all(r["focus"] == "latency" and r["sol_rigor"] == "L3" for r in d)


def test_build_sol_table_empty_when_no_cells(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)  # no cells/ dir
    table = build_sol_table(
        campaign_dir, campaign_dir.name, [_make_atlas_row()],
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
        focus="mixed", sol_rigor="none",
    )
    assert table.num_rows == 0
    assert [f.name for f in table.schema] == _SOL_SCHEMA_COLUMNS


def test_build_tpm_table_peak_only_without_sla():
    rows = [
        _make_atlas_row(concurrency=1, output_tps_per_gpu=9.15),
        _make_atlas_row(concurrency=2, output_tps_per_gpu=18.81),
    ]
    table = build_tpm_table(
        rows, "x-20260525T081650Z",
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
    )
    ops = set(table.column("operating_point").to_pylist())
    assert ops == {"peak"}  # no SLA thresholds -> peak-only


def test_build_tpm_table_emits_sla_rows_with_thresholds():
    rows = [
        _make_atlas_row(concurrency=1, output_tps_per_gpu=9.15, ttft_avg_ms=182.04),
        _make_atlas_row(concurrency=2, output_tps_per_gpu=18.81, ttft_avg_ms=264.31),
    ]
    # ttft<=200 -> only the c=1 row qualifies; peak is the c=2 row.
    table = build_tpm_table(
        rows, "x-20260525T081650Z",
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
        ttft_sla_ms=200.0,
    )
    d = table.to_pylist()
    ops = {r["operating_point"] for r in d}
    assert ops == {"peak", "sla"}
    sla = [r for r in d if r["operating_point"] == "sla"]
    assert all(r["concurrency"] == 1 for r in sla)  # SLA point = the qualifying row
    assert len(sla) == 3  # 3 bases


def test_build_cost_table_usd_per_1m_tokens(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    # 10 tok/s/GPU output -> tokens/hour/GPU = 36000; $4.50/GPU-hr ->
    # $/1M out = 4.5*1e6/(10*3600) = 125.0
    rows = [_make_atlas_row(output_tps_per_gpu=10.0, total_tps_per_gpu=40.0)]
    table = build_cost_table(
        campaign_dir, rows, campaign_dir.name,
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
        usd_per_gpu_hour={"B200": 4.50},
    )
    d = table.to_pylist()
    assert len(d) == 1
    assert abs(d[0]["usd_per_1m_output_tokens"] - 125.0) < 1e-6
    # total 40 tok/s/GPU -> $/1M total = 4.5*1e6/(40*3600) = 31.25
    assert abs(d[0]["usd_per_1m_total_tokens"] - 31.25) < 1e-6
    assert d[0]["usd_per_gpu_hour"] == 4.50
    # No DCGM power staged -> tokens_per_watt null.
    assert d[0]["tokens_per_watt"] is None


def test_build_cost_table_cost_null_without_config(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    table = build_cost_table(
        campaign_dir, [_make_atlas_row(output_tps_per_gpu=10.0)], campaign_dir.name,
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
    )
    d = table.to_pylist()
    assert d[0]["usd_per_1m_output_tokens"] is None
    assert d[0]["usd_per_gpu_hour"] is None


def test_build_cost_table_tokens_per_watt_from_dcgm_power(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    # Stage a per-cell dcgm_correlation.json carrying power_watts_per_gpu.
    cell = campaign_dir / "cells" / "cellA"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "dcgm_correlation.json").write_text(json.dumps({
        "schema_version": 1, "captured_sources": ["dcgm"], "hw_key": "b200_sm100",
        "resources": [], "queries": [], "power_watts_per_gpu": 700.0,
    }))
    # The peak point's row must originate from cellA so the power join hits.
    rows = [_make_atlas_row(cell_id="cellA", output_tps_per_gpu=350.0)]
    table = build_cost_table(
        campaign_dir, rows, campaign_dir.name,
        captured_at_utc=datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc),
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
    )
    d = table.to_pylist()
    assert d[0]["power_watts_per_gpu"] == 700.0
    # tokens_per_watt = 350 / 700 = 0.5
    assert abs(d[0]["tokens_per_watt"] - 0.5) < 1e-6


def test_publish_reads_tpm_config_block(tmp_path: Path):
    """A config.yaml `tpm:` block makes publish emit sla rows into tpm_v1."""
    import pyarrow.parquet as pq

    campaign_dir = _stage_campaign(tmp_path)
    # Re-stage atlas with a decode metric (tpot) so an SLA point can exist; the
    # config block sets both thresholds the rows meet.
    rows = [
        _make_atlas_row(concurrency=1, ttft_avg_ms=182.0, tpot_median_ms=20.0),
        _make_atlas_row(concurrency=2, ttft_avg_ms=264.0, tpot_median_ms=22.0,
                        output_tps_per_gpu=18.81),
    ]
    with (campaign_dir / "atlas.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    (campaign_dir / "config.yaml").write_text(
        "name: test\ntpm:\n  ttft_sla_ms: 300\n  tpot_sla_ms: 50\n  gpus_per_node: 4\n"
    )
    fake = _FakeS3Client()
    result = publish(
        campaign_dir,
        cfg=_stub_cfg(),
        dry_run=True,
        s3_client_factory=lambda _cfg: fake,
    )
    tpm = pq.read_table(result.tpm.local_path).to_pylist()
    ops = {r["operating_point"] for r in tpm}
    assert "sla" in ops  # config block threaded through -> sla rows land
    assert all(r["gpus_per_node"] == 4 for r in tpm)  # node size from config


def test_publish_default_cost_populates_without_config_block(tmp_path: Path):
    """No cost: block -> the v1.49.0 default public-list table still populates
    $/1M for a default-rate hardware (B200), and no spurious unmatched-hardware
    warning fires for the default H100/H200 keys."""
    import pyarrow.parquet as pq

    campaign_dir = _stage_campaign(tmp_path)  # B200 rows, config has no cost: block
    fake = _FakeS3Client()
    result = publish(
        campaign_dir,
        cfg=_stub_cfg(),
        dry_run=True,
        s3_client_factory=lambda _cfg: fake,
    )
    cost = pq.read_table(result.cost.local_path).to_pylist()
    assert cost, "expected a cost_v1 row"
    # B200 default = $8.60/GPU-hr -> $/1M output non-null.
    assert cost[0]["usd_per_gpu_hour"] == 8.60
    assert cost[0]["usd_per_1m_output_tokens"] is not None


def test_build_campaign_row_reads_sourcemd_provenance(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    rows = [
        _make_atlas_row(cell_id="cellA"),
        _make_atlas_row(cell_id="cellB"),
    ]
    table = build_campaign_row(
        campaign_dir,
        rows,
        published_at_utc=datetime(2026, 5, 25, 16, 0, 0, tzinfo=timezone.utc),
        publisher_operator="ci-bot",
        publisher_host="ci-runner-1",
    )
    assert table.num_rows == 1
    row = {name: table.column(name).to_pylist()[0] for name in table.schema.names}
    assert row["campaign_id"] == campaign_dir.name
    assert row["operator"] == "security@example.com"
    assert row["cluster_context"] == "<cluster>"
    assert row["cell_count"] == 2
    assert row["backends"] == "vllm-sweep"
    assert row["atlas_row_count"] == 2
    assert row["publisher_operator"] == "ci-bot"
    assert row["publisher_host"] == "ci-runner-1"
    assert row["captured_at_utc"] == datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc)
    # config_yaml + SOURCE.md were created by _stage_campaign, so digests are non-empty.
    assert len(row["config_yaml_sha256"]) == 64
    assert len(row["source_md_sha256"]) == 64


# ---------------------------------------------------------------------------
# S3 key layout
# ---------------------------------------------------------------------------


def test_s3_key_uses_hive_layout():
    captured = datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc)
    atlas_key = s3_key_for(ATLAS_TABLE_NAME, "glm51-phase6-20260525T081650Z", captured)
    assert atlas_key == (
        f"{S3_PREFIX}/atlas_v1/dt=2026-05-25/"
        "campaign=glm51-phase6-20260525T081650Z/part-0.parquet"
    )
    campaign_key = s3_key_for(CAMPAIGN_TABLE_NAME, "glm51-phase6-20260525T081650Z", captured)
    assert campaign_key.endswith("campaign_v1/dt=2026-05-25/campaign=glm51-phase6-20260525T081650Z/part-0.parquet")


# ---------------------------------------------------------------------------
# Upload semantics (no real boto3)
# ---------------------------------------------------------------------------


class _FakeS3Client:
    """Minimal boto3 S3 client stub usable as ``s3_client_factory`` injection."""

    def __init__(self, *, existing_keys: set[str] | None = None):
        self.existing_keys = set(existing_keys or ())
        self.put_calls: list[dict[str, Any]] = []
        self.head_calls: list[str] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.head_calls.append(Key)
        if Key not in self.existing_keys:
            raise FileNotFoundError(f"NoSuchKey: {Key}")
        return {"ContentLength": 0}

    def put_object(self, *, Bucket: str, Key: str, Body: Any) -> dict[str, Any]:
        self.put_calls.append({"bucket": Bucket, "key": Key})
        self.existing_keys.add(Key)
        return {"ETag": '"deadbeef"'}


def _stub_cfg() -> S3Config:
    return S3Config(endpoint="https://example.test", bucket="perf-lake", access_key="AK", secret_key="SK")


def test_upload_skip_returns_skipped_when_key_exists(tmp_path: Path):
    local = tmp_path / "f.parquet"
    local.write_bytes(b"hello")
    fake = _FakeS3Client(existing_keys={"a/b/part-0.parquet"})
    out = upload_to_s3(
        local,
        bucket="perf-lake",
        key="a/b/part-0.parquet",
        cfg=_stub_cfg(),
        if_exists=IF_EXISTS_SKIP,
        s3_client_factory=lambda _cfg: fake,
    )
    assert out["skipped"] is True
    assert fake.put_calls == []


def test_upload_fail_raises_when_key_exists(tmp_path: Path):
    local = tmp_path / "f.parquet"
    local.write_bytes(b"hello")
    fake = _FakeS3Client(existing_keys={"a/b/part-0.parquet"})
    with pytest.raises(FileExistsError, match="already exists"):
        upload_to_s3(
            local,
            bucket="perf-lake",
            key="a/b/part-0.parquet",
            cfg=_stub_cfg(),
            if_exists=IF_EXISTS_FAIL,
            s3_client_factory=lambda _cfg: fake,
        )


def test_upload_overwrite_replaces(tmp_path: Path):
    local = tmp_path / "f.parquet"
    local.write_bytes(b"new bytes")
    fake = _FakeS3Client(existing_keys={"a/b/part-0.parquet"})
    out = upload_to_s3(
        local,
        bucket="perf-lake",
        key="a/b/part-0.parquet",
        cfg=_stub_cfg(),
        if_exists=IF_EXISTS_OVERWRITE,
        s3_client_factory=lambda _cfg: fake,
    )
    assert out["skipped"] is False
    assert fake.put_calls == [{"bucket": "perf-lake", "key": "a/b/part-0.parquet"}]


def test_upload_writes_when_key_missing(tmp_path: Path):
    local = tmp_path / "f.parquet"
    local.write_bytes(b"first write")
    fake = _FakeS3Client()
    out = upload_to_s3(
        local,
        bucket="perf-lake",
        key="a/b/part-0.parquet",
        cfg=_stub_cfg(),
        if_exists=IF_EXISTS_FAIL,
        s3_client_factory=lambda _cfg: fake,
    )
    assert out["skipped"] is False
    assert fake.put_calls == [{"bucket": "perf-lake", "key": "a/b/part-0.parquet"}]


# ---------------------------------------------------------------------------
# publish() orchestrator
# ---------------------------------------------------------------------------


def test_publish_dry_run_writes_local_only(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    fake = _FakeS3Client()
    result = publish(
        campaign_dir,
        cfg=_stub_cfg(),
        dry_run=True,
        s3_client_factory=lambda _cfg: fake,
    )
    assert result.dry_run is True
    assert result.atlas.local_path.is_file()
    assert result.campaign.local_path.is_file()
    assert result.atlas.row_count == 3
    assert result.campaign.row_count == 1
    assert fake.put_calls == []
    assert fake.head_calls == []


def test_publish_real_run_uploads_eight_objects(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    fake = _FakeS3Client()
    result = publish(
        campaign_dir,
        cfg=_stub_cfg(),
        dry_run=False,
        s3_client_factory=lambda _cfg: fake,
    )
    assert result.dry_run is False
    assert len(fake.put_calls) == 8
    keys = {call["key"] for call in fake.put_calls}
    assert any("/atlas_v1/" in k for k in keys)
    assert any("/campaign_v1/" in k for k in keys)
    assert any("/sol_v1/" in k for k in keys)
    assert any("/tpm_v1/" in k for k in keys)
    assert any("/cost_v1/" in k for k in keys)
    assert any("/quality_v1/" in k for k in keys)
    assert any("/champion_v1/" in k for k in keys)
    assert any("/roofline_v1/" in k for k in keys)
    # The staged campaign has throughput-bearing rows (2 distinct cells), so
    # tpm_v1 lands a non-empty rollup: peak point x 3 bases per group.
    assert result.tpm.row_count > 0
    # cost_v1: one row per (group, operating_point); peak-only here -> 1 row.
    assert result.cost.row_count == 1


def test_publish_skip_on_existing_keys(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    captured = datetime(2026, 5, 25, 8, 16, 50, tzinfo=timezone.utc)
    pre_existing = {
        s3_key_for(ATLAS_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(CAMPAIGN_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(SOL_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(TPM_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(COST_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(QUALITY_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(CHAMPION_TABLE_NAME, campaign_dir.name, captured),
        s3_key_for(ROOFLINE_TABLE_NAME, campaign_dir.name, captured),
    }
    fake = _FakeS3Client(existing_keys=set(pre_existing))
    result = publish(
        campaign_dir,
        cfg=_stub_cfg(),
        dry_run=False,
        if_exists=IF_EXISTS_SKIP,
        s3_client_factory=lambda _cfg: fake,
    )
    assert fake.put_calls == []
    assert result.atlas.skipped is True
    assert result.campaign.skipped is True
    assert result.sol.skipped is True
    assert result.tpm.skipped is True
    assert result.cost.skipped is True


# ---------------------------------------------------------------------------
# Completeness gate
# ---------------------------------------------------------------------------


def _write_status(campaign_dir: Path, **overrides) -> None:
    status = {
        "sol_complete": True,
        "plot_ready_points": 3,
        "non_plot_ready_full_cells": 0,
        "rendered_pages": ["scatter (page 1)"],
        "omitted_pages": [],
    }
    status.update(overrides)
    (campaign_dir / "report_status.json").write_text(json.dumps(status))


def test_publish_lands_incomplete_campaign_and_records_gap(tmp_path: Path):
    """Always-publish policy (v1.33.0): an incomplete (no-SoL) campaign LANDS by
    default (no refusal) and records the gap on the lake row; --strict refuses."""
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(
        campaign_dir,
        sol_complete=False,
        sol_rigor="none",
        focus="latency",
        omitted_pages=[{"page": "Speed-of-Light roofline (page 4)", "why": "x", "how_to_fix": "y"}],
    )
    fake = _FakeS3Client()
    result = publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, s3_client_factory=lambda _cfg: fake)
    assert result.campaign.row_count == 1
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    cols = {name: table.column(name)[0].as_py() for name in table.column_names}
    assert cols["sol_complete"] is False
    assert cols["focus"] == "latency"
    assert cols["sol_rigor"] == "none"


def test_publish_strict_refuses_incomplete_campaign(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(
        campaign_dir,
        sol_complete=False,
        omitted_pages=[{"page": "Speed-of-Light roofline (page 4)", "why": "x", "how_to_fix": "y"}],
    )
    fake = _FakeS3Client()
    with pytest.raises(CampaignIncompleteError, match="Speed-of-Light"):
        publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True, s3_client_factory=lambda _cfg: fake)
    assert fake.put_calls == []


def test_publish_refuses_unrendered_campaign(tmp_path: Path):
    """The one hard requirement that survives the always-publish policy: you
    must run report_render first (no report_status.json -> nothing to publish)."""
    campaign_dir = _stage_campaign(tmp_path)
    (campaign_dir / "report_status.json").unlink()
    with pytest.raises(CampaignIncompleteError, match="report_render was never run"):
        publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, s3_client_factory=lambda _cfg: _FakeS3Client())


def test_publish_lands_zero_plot_ready_latency_focus(tmp_path: Path):
    """focus=latency with 0 throughput-scatter points is a first-class result."""
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(campaign_dir, plot_ready_points=0, focus="latency")
    fake = _FakeS3Client()
    result = publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, s3_client_factory=lambda _cfg: fake)
    assert result.campaign.row_count == 1


def test_publish_lands_zero_plot_ready_accuracy_focus(tmp_path: Path):
    """focus=accuracy with 0 throughput-scatter points is a first-class result
    (training-accuracy / EAGLE3-acceptance sweeps have no throughput scatter)."""
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(campaign_dir, plot_ready_points=0, focus="accuracy")
    fake = _FakeS3Client()
    result = publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, s3_client_factory=lambda _cfg: fake)
    assert result.campaign.row_count == 1


def test_publish_strict_refuses_zero_plot_ready(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(campaign_dir, plot_ready_points=0)  # focus defaults "mixed"
    with pytest.raises(CampaignIncompleteError, match="throughput-scatter"):
        publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True, s3_client_factory=lambda _cfg: _FakeS3Client())


def test_campaign_row_records_focus_and_sol_rigor(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(campaign_dir, focus="throughput", sol_rigor="L4")
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    cols = {name: table.column(name)[0].as_py() for name in table.column_names}
    assert cols["focus"] == "throughput"
    assert cols["sol_rigor"] == "L4"


def test_campaign_row_has_completeness_columns(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    for col in ("sol_complete", "plot_ready_points", "omitted_pages", "partial_pages",
                "sol_per_arm_complete"):
        assert col in table.column_names
    assert table.column("sol_complete")[0].as_py() is True
    assert table.column("plot_ready_points")[0].as_py() == 3
    # Default-complete staged campaign has no partial pages.
    assert table.column("partial_pages")[0].as_py() == ""
    # Per-arm coverage column defaults True (pre-field report_status.json is not
    # retroactively marked incomplete).
    assert table.column("sol_per_arm_complete")[0].as_py() is True


def test_campaign_row_records_per_arm_incomplete(tmp_path: Path):
    """A report_status.json with sol_per_arm_complete=false -> the campaign_v1
    sol_per_arm_complete column records False (baseline+variant gap is queryable)."""
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(campaign_dir, sol_per_arm_complete=False, arms_uncovered=["armB", "armC"])
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    assert table.column("sol_per_arm_complete")[0].as_py() is False


def test_campaign_row_records_partial_pages(tmp_path: Path):
    """A report_status.json with partial_pages -> comma-joined campaign_v1 column."""
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(
        campaign_dir,
        partial_pages=[
            {
                "page": "Byte-grounded per-kernel SoL scatter (page 5)",
                "why": "AI unmeasured (--set=basic)",
                "how_to_fix": "re-capture with --roofline-min",
            }
        ],
    )
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    assert (
        table.column("partial_pages")[0].as_py()
        == "Byte-grounded per-kernel SoL scatter (page 5)"
    )


def test_campaign_row_defaults_experiment_id_to_campaign_id(tmp_path: Path):
    """A pre-v1.34.0 SOURCE.md (no experiment_id:) still yields a non-empty join
    key: experiment_id defaults to campaign_id, family/bundle default to ''."""
    campaign_dir = _stage_campaign(tmp_path)
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    cols = {name: table.column(name)[0].as_py() for name in table.column_names}
    assert cols["experiment_id"] == campaign_dir.name
    assert cols["experiment_family"] == ""
    assert cols["evidence_bundle_path"] == ""


def test_campaign_row_reads_experiment_join_keys(tmp_path: Path):
    """When campaign_init wrote experiment_id/family/evidence_bundle_path into
    SOURCE.md, they land as first-class campaign_v1 columns."""
    campaign_dir = _stage_campaign(tmp_path)
    (campaign_dir / "SOURCE.md").write_text(
        "# Campaign\n\n"
        "- captured_at: 20260525T081650Z\n"
        "- config: /tmp/test.yaml\n"
        "- experiment_id: glm51-nvfp4kv-controlled-ab-20260525T081650Z\n"
        "- family: nvfp4-kv\n"
        "- evidence_bundle_path: <external-workspace>\n"
    )
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    cols = {name: table.column(name)[0].as_py() for name in table.column_names}
    assert cols["experiment_id"] == "glm51-nvfp4kv-controlled-ab-20260525T081650Z"
    assert cols["experiment_family"] == "nvfp4-kv"
    assert cols["evidence_bundle_path"] == "<external-workspace>"


def test_campaign_row_reads_source_attribution_columns(tmp_path: Path):
    """The flat source-provenance bullets campaign_init wrote (from the bundle's
    ```provenance``` block) land as first-class campaign_v1 source columns."""
    campaign_dir = _stage_campaign(tmp_path)
    (campaign_dir / "SOURCE.md").write_text(
        "# Campaign\n\n"
        "- captured_at: 20260525T081650Z\n"
        "- vllm_repo: example/vllm\n"
        "- vllm_branch: feature/nvfp4-kv\n"
        "- vllm_commit: b5743e12e\n"
        "- delivery: overlay\n"
        "- experiment_status: verified\n"
        "- title: NVFP4 vs FP8\n"
        "- code_sha: eafb4b4\n"
    )
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    cols = {name: table.column(name)[0].as_py() for name in table.column_names}
    assert cols["vllm_commit"] == "b5743e12e"
    assert cols["vllm_branch"] == "feature/nvfp4-kv"
    assert cols["delivery"] == "overlay"
    assert cols["experiment_status"] == "verified"
    assert cols["title"] == "NVFP4 vs FP8"
    assert cols["code_sha"] == "eafb4b4"


def test_campaign_row_source_columns_default_when_absent(tmp_path: Path):
    """A pre-provenance bundle still publishes; source columns default empty,
    experiment_status defaults 'active'."""
    campaign_dir = _stage_campaign(tmp_path)
    table = build_campaign_row(campaign_dir, _read_atlas_rows(campaign_dir))
    cols = {name: table.column(name)[0].as_py() for name in table.column_names}
    assert cols["vllm_commit"] == ""
    assert cols["experiment_status"] == "active"


def test_publish_strict_blocks_verdict_without_pinned_source(tmp_path: Path):
    """A verdict-tier campaign with no provenance block is refused under strict
    (the source problem appears in the combined refusal message)."""
    campaign_dir = _stage_campaign(tmp_path)
    _write_status(campaign_dir, focus="latency")
    (campaign_dir / "verdict.json").write_text(
        json.dumps({"tier": "verdict", "trials": 3, "same_node": True,
                    "baseline_named": True})
    )
    with pytest.raises(CampaignIncompleteError, match="provenance block"):
        publish(campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True,
                s3_client_factory=lambda _cfg: _FakeS3Client())


def test_source_problems_blocks_unpinned_verdict(tmp_path: Path):
    """lake-side wrapper: verdict tier + no provenance -> a problem."""
    from tools.perf_tune_report.lake_writer import source_problems

    campaign_dir = _stage_campaign(tmp_path)
    (campaign_dir / "verdict.json").write_text(json.dumps({"tier": "verdict"}))
    probs = source_problems(campaign_dir)
    assert probs and "provenance block" in probs[0]


def test_source_problems_passes_pinned_verdict(tmp_path: Path):
    """lake-side wrapper: a clean pinned source commit clears the gate."""
    from tools.perf_tune_report.lake_writer import source_problems

    campaign_dir = _stage_campaign(tmp_path)
    (campaign_dir / "verdict.json").write_text(json.dumps({"tier": "verdict"}))
    (campaign_dir / "config.yaml").write_text(
        "name: test\ncells: []\n"
        "provenance:\n"
        "  schema: experiment_provenance_v1\n"
        "  identity: {run_id: test-20260525T081650Z, status: verified}\n"
        "  source:\n"
        "    - {repo: example/vllm, commit: b5743e12e, dirty: false}\n"
        "  verdict: {tier: verdict}\n"
    )
    assert source_problems(campaign_dir) == []


def test_source_problems_drafts_never_blocked(tmp_path: Path):
    """A draft (no verdict.json) is never blocked by the source gate."""
    from tools.perf_tune_report.lake_writer import source_problems

    campaign_dir = _stage_campaign(tmp_path)
    assert source_problems(campaign_dir) == []


def _read_atlas_rows(campaign_dir: Path):
    from tools.perf_tune_report.schema import read_jsonl

    return read_jsonl(campaign_dir / "atlas.jsonl")


# ---------------------------------------------------------------------------
# S3 config resolution
# ---------------------------------------------------------------------------


def test_resolve_s3_config_reads_env(monkeypatch):
    monkeypatch.setenv("PERFLAKE_LAKE_S3_ACCESS_KEY", "envAK")
    monkeypatch.setenv("PERFLAKE_LAKE_S3_SECRET_KEY", "envSK")
    monkeypatch.delenv("PERFLAKE_LAKE_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("PERFLAKE_LAKE_S3_BUCKET", raising=False)
    cfg = resolve_s3_config(
        endpoint=None,
        bucket=None,
        access_key_file=None,
        secret_key_file=None,
    )
    assert cfg.endpoint == "https://object-store.example.com"
    assert cfg.bucket == "perf-lake"
    assert cfg.access_key == "envAK"
    assert cfg.secret_key == "envSK"


def test_resolve_s3_config_files_win_over_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PERFLAKE_LAKE_S3_ACCESS_KEY", "envAK")
    monkeypatch.setenv("PERFLAKE_LAKE_S3_SECRET_KEY", "envSK")
    ak_file = tmp_path / "ak.txt"
    sk_file = tmp_path / "sk.txt"
    ak_file.write_text("fileAK\n")
    sk_file.write_text("fileSK\n")
    cfg = resolve_s3_config(
        endpoint="https://custom.test",
        bucket="custom-bucket",
        access_key_file=str(ak_file),
        secret_key_file=str(sk_file),
    )
    assert cfg.access_key == "fileAK"
    assert cfg.secret_key == "fileSK"
    assert cfg.endpoint == "https://custom.test"
    assert cfg.bucket == "custom-bucket"


def test_resolve_s3_config_missing_keys_exits(monkeypatch):
    monkeypatch.delenv("PERFLAKE_LAKE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("PERFLAKE_LAKE_S3_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit, match="S3 access/secret key missing"):
        resolve_s3_config(
            endpoint=None, bucket=None,
            access_key_file=None, secret_key_file=None,
        )


# ---------------------------------------------------------------------------
# CLI smoke (dry-run path, no boto3)
# ---------------------------------------------------------------------------


def test_cli_dry_run_succeeds_without_s3(monkeypatch, tmp_path: Path, capsys):
    campaign_dir = _stage_campaign(tmp_path)
    monkeypatch.setenv("PERFREPORT_CAMPAIGNS_DIR", str(tmp_path))
    monkeypatch.setenv("PERFLAKE_LAKE_S3_ACCESS_KEY", "envAK")
    monkeypatch.setenv("PERFLAKE_LAKE_S3_SECRET_KEY", "envSK")
    # --no-strict: this fixture is intentionally SoL-incomplete (no cells/*/ SoL
    # artifacts -> sol_v1 empty); publish defaults strict now, so opt out to
    # exercise the first-class intentional-gap publish path.
    rc = main([
        "publish_to_lake",
        "--campaign", campaign_dir.name,
        "--dry-run",
        "--no-strict",
        "--json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    envelope = json.loads(out)
    assert envelope["tool"] == "perf_tune_report_publish_to_lake"
    assert envelope["dry_run"] is True
    assert envelope["bucket"] == "perf-lake"
    assert "atlas_v1" in envelope["tables"]
    assert "campaign_v1" in envelope["tables"]
    assert "sol_v1" in envelope["tables"]
    assert "tpm_v1" in envelope["tables"]
    assert "cost_v1" in envelope["tables"]
    assert envelope["tables"]["atlas_v1"]["row_count"] == 3
    assert envelope["tables"]["campaign_v1"]["row_count"] == 1
    # No cells/*/ SoL artifacts staged by default -> sol_v1 is published empty.
    assert envelope["tables"]["sol_v1"]["row_count"] == 0
    assert envelope["tables"]["sol_v1"]["s3_key"].startswith(f"{S3_PREFIX}/sol_v1/dt=")
    # tpm_v1: cellA + cellB share one (model, hw, quant, TP, strategy, mtp)
    # identity -> 1 group; publish emits the peak point x 3 bases = 3 rows.
    assert envelope["tables"]["tpm_v1"]["row_count"] == 3
    assert envelope["tables"]["tpm_v1"]["s3_key"].startswith(f"{S3_PREFIX}/tpm_v1/dt=")
    # cost_v1: 1 group x peak = 1 row ($/1M now non-null via the v1.49.0 default
    # public-list rate for B200; tokens-per-watt still null without DCGM power).
    assert envelope["tables"]["cost_v1"]["row_count"] == 1
    assert envelope["tables"]["cost_v1"]["s3_key"].startswith(f"{S3_PREFIX}/cost_v1/dt=")
    assert envelope["tables"]["atlas_v1"]["s3_key"].startswith(f"{S3_PREFIX}/atlas_v1/dt=")


def test_publish_appends_lake_provenance_to_evidence_bundle(monkeypatch, tmp_path: Path, capsys):
    """A real publish writes campaign + s3 paths back into the evidence bundle's
    SOURCE.md named by the campaign SOURCE.md evidence_bundle_path (close-the-loop)."""
    # Evidence bundle with a SOURCE.md.
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SOURCE.md").write_text("# SOURCE\n\n- experiment_id: x-20260525T081650Z\n")
    # Campaign that points back at the bundle.
    campaign_dir = _stage_campaign(tmp_path)
    (campaign_dir / "SOURCE.md").write_text(
        "# Campaign\n\n"
        "- captured_at: 20260525T081650Z\n"
        "- config: /tmp/test.yaml\n"
        f"- evidence_bundle_path: {bundle}\n"
    )
    monkeypatch.setenv("PERFREPORT_CAMPAIGNS_DIR", str(tmp_path))
    monkeypatch.setenv("PERFLAKE_LAKE_S3_ACCESS_KEY", "envAK")
    monkeypatch.setenv("PERFLAKE_LAKE_S3_SECRET_KEY", "envSK")

    import tools.perf_tune_report.lake_writer as lw

    monkeypatch.setattr(lw, "_make_s3_client", lambda cfg: _FakeS3Client())
    # --no-strict: SoL-incomplete fixture (intentional-gap publish path).
    rc = main(["publish_to_lake", "--campaign", campaign_dir.name, "--no-strict", "--json"])
    assert rc == 0
    updated = (bundle / "SOURCE.md").read_text()
    assert f"campaign={campaign_dir.name}" in updated
    assert "atlas_v1: s3://perf-lake/" in updated
    assert "campaign_v1: s3://perf-lake/" in updated
    # Idempotent: a second publish does not duplicate the block.
    rc2 = main(["publish_to_lake", "--campaign", campaign_dir.name, "--if-exists", "overwrite", "--no-strict", "--json"])
    assert rc2 == 0
    assert (bundle / "SOURCE.md").read_text().count(
        "## Perf-lake publish (auto-appended") == 1


def test_cli_strict_is_default_refuses_incomplete(monkeypatch, tmp_path: Path, capsys):
    """Strict-by-default flip: a SoL-incomplete campaign must FAIL publish unless
    --no-strict is passed (workspace rigor policy, docs/METHODOLOGY.md)."""
    campaign_dir = _stage_campaign(tmp_path)  # SoL-incomplete (no cells/*/ SoL)
    monkeypatch.setenv("PERFREPORT_CAMPAIGNS_DIR", str(tmp_path))
    monkeypatch.setenv("PERFLAKE_LAKE_S3_ACCESS_KEY", "envAK")
    monkeypatch.setenv("PERFLAKE_LAKE_S3_SECRET_KEY", "envSK")
    # No --no-strict -> strict is the default now -> refuse.
    rc = main(["publish_to_lake", "--campaign", campaign_dir.name, "--dry-run", "--json"])
    assert rc == 2


def test_cli_missing_atlas_fails_with_clear_message(monkeypatch, tmp_path: Path, capsys):
    campaign_dir = tmp_path / "bare-20260525T081650Z"
    campaign_dir.mkdir()
    monkeypatch.setenv("PERFREPORT_CAMPAIGNS_DIR", str(tmp_path))
    monkeypatch.setenv("PERFLAKE_LAKE_S3_ACCESS_KEY", "envAK")
    monkeypatch.setenv("PERFLAKE_LAKE_S3_SECRET_KEY", "envSK")
    rc = main([
        "publish_to_lake",
        "--campaign", campaign_dir.name,
        "--dry-run",
        "--json",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "atlas.jsonl not found" in err
