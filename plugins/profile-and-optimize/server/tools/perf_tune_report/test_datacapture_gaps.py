"""Tests for the ledger-to-atlas data-capture gap fields (added 2026-06-07).

Covers the 5 gaps -- routing descriptor, spec-decode acceptance length, KV-cache
token capacity, DeepEP/EP mode, and per-row DCGM utilization -- verifying they
default safely, round-trip through write_jsonl/read_jsonl, and surface as first-class
atlas_v1 columns. See perf-tune-report/UPSTREAM-REQUEST-atlas-ledger-datacapture-gaps.md.
"""

from __future__ import annotations

from tools.perf_tune_report.lake_writer import build_atlas_table
from tools.perf_tune_report.schema import AtlasCell, STATUS_FULL, read_jsonl, write_jsonl


def _cell_with_gaps() -> AtlasCell:
    return AtlasCell(
        cell_id="kvr-reuse10", model="GLM-5.1", hardware="GB300", quant="NVFP4",
        tensor_parallel=4, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=8192, concurrency=64, status=STATUS_FULL,
        router_policy="prefix-affinity", prefix_reuse=10.0, per_replica_cache_hit=0.83,
        acceptance_length=1.955, kv_cache_tokens=524288, ep_mode="deepep-ll",
        dcgm_sm_active=0.41, dcgm_dram_active=0.28, dcgm_tensor_active=0.12,
    )


def test_datacapture_gap_fields_default_safely():
    c = AtlasCell(
        cell_id="x", model="m", hardware="GB300", quant="NVFP4",
        tensor_parallel=4, parallel_strategy="TP", mtp=False,
        max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
    )
    assert c.router_policy == "" and c.ep_mode == ""
    assert c.prefix_reuse is None and c.per_replica_cache_hit is None
    assert c.acceptance_length is None and c.kv_cache_tokens is None
    assert c.dcgm_sm_active is None and c.dcgm_dram_active is None and c.dcgm_tensor_active is None


def test_datacapture_gap_fields_roundtrip(tmp_path):
    out = tmp_path / "atlas.jsonl"
    write_jsonl([_cell_with_gaps()], out)
    (back,) = read_jsonl(out)
    assert back.router_policy == "prefix-affinity"
    assert back.prefix_reuse == 10.0
    assert back.per_replica_cache_hit == 0.83
    assert back.acceptance_length == 1.955
    assert back.kv_cache_tokens == 524288
    assert back.ep_mode == "deepep-ll"
    assert back.dcgm_sm_active == 0.41
    assert back.dcgm_dram_active == 0.28
    assert back.dcgm_tensor_active == 0.12


def test_datacapture_gap_fields_in_atlas_table():
    table = build_atlas_table([_cell_with_gaps()], campaign_id="kvr-20260607T000000Z")
    names = set(table.schema.names)
    for col in (
        "router_policy", "prefix_reuse", "per_replica_cache_hit", "acceptance_length",
        "kv_cache_tokens", "ep_mode", "dcgm_sm_active", "dcgm_dram_active", "dcgm_tensor_active",
    ):
        assert col in names, f"{col} missing from atlas_v1 schema"
    d = table.to_pydict()
    assert d["router_policy"][0] == "prefix-affinity"
    assert d["acceptance_length"][0] == 1.955
    assert d["kv_cache_tokens"][0] == 524288
    assert d["ep_mode"][0] == "deepep-ll"
    assert d["dcgm_tensor_active"][0] == 0.12


def test_datacapture_gap_minimal_cell_atlas_table_defaults():
    # A cell that does not set the gap fields still builds a valid table (defaults).
    minimal = AtlasCell(
        cell_id="x", model="m", hardware="GB300", quant="NVFP4",
        tensor_parallel=4, parallel_strategy="TP", mtp=False,
        max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
    )
    d = build_atlas_table([minimal], campaign_id="c-20260607T000000Z").to_pydict()
    assert d["router_policy"][0] == "" and d["ep_mode"][0] == ""
    assert d["acceptance_length"][0] is None and d["kv_cache_tokens"][0] is None
    assert d["dcgm_sm_active"][0] is None
