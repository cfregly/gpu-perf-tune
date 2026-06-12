"""Tests for exact-variant capture planning and materialization."""

from __future__ import annotations

import json
from pathlib import Path

from tools.perf_tune_report.capture_signature import (
    build_plan,
    materialize_reuse,
    signature_for_row,
)
from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.schema import AtlasCell


def _row(cell_id: str, **overrides) -> AtlasCell:
    base = dict(
        cell_id=cell_id,
        model="nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
        hardware="GB300",
        quant="NVFP4",
        tensor_parallel=4,
        parallel_strategy="EP",
        mtp=False,
        max_num_batched_tokens=12288,
        concurrency=64,
        status="full",
        ttft_avg_ms=100.0,
        request_throughput_avg=3.0,
        output_tps_per_gpu=500.0,
        tpot_median_ms=20.0,
        cache_mode="cold",
        dataset="random",
        cudagraph_mode="full",
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8_e4m3",
        image="infr/vllm:v2.12.3",
        backend="vllm-sweep",
        raw_path="raw.txt",
        captured_at="2026-06-07T00:00:00Z",
        extra={"max_num_seqs": 96},
    )
    base.update(overrides)
    return AtlasCell(**base)


def _write_campaign(path: Path, rows: list[AtlasCell]) -> Path:
    path.mkdir(parents=True)
    with (path / "atlas.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    return path


def test_signature_distinguishes_first_class_variant_fields():
    """The 2026-06-07 first-class variant fields change the signature (so e.g. MTP-K=2
    vs K=3, async on vs off are distinct variants), and the legacy extra fallback holds."""
    base = _row("base")
    assert signature_for_row(base).hash != signature_for_row(
        _row("k3", num_speculative_tokens=3)).hash
    assert signature_for_row(base).hash != signature_for_row(
        _row("async", async_scheduling=True)).hash
    assert signature_for_row(base).hash != signature_for_row(
        _row("pc", enable_prefix_caching=True)).hash
    # first-class max_num_seqs and the legacy extra["max_num_seqs"] resolve to the same value
    assert signature_for_row(_row("a", max_num_seqs=96, extra={})).hash == \
        signature_for_row(_row("b", extra={"max_num_seqs": 96})).hash
    # None max_num_batched_tokens (accuracy row) does not crash the signature
    assert signature_for_row(_row("acc", max_num_batched_tokens=None)).hash


def test_signature_changes_on_exact_variant_axes():
    base = _row("base")
    assert signature_for_row(base).hash == signature_for_row(_row("same")).hash
    assert signature_for_row(base).hash != signature_for_row(
        _row("tp1", tensor_parallel=1, parallel_strategy="TP")
    ).hash
    assert signature_for_row(base).hash != signature_for_row(
        _row("mns128", extra={"max_num_seqs": 128})
    ).hash
    assert signature_for_row(base).hash != signature_for_row(
        _row("sglang", backend="sglang-sweep")
    ).hash
    assert signature_for_row(base).hash != signature_for_row(
        _row("fimoe", extra={"max_num_seqs": 96, "env": {"VLLM_USE_FLASHINFER_MOE_FP4": "1"}})
    ).hash


def test_capture_plan_groups_missing_and_finds_exact_reuse(tmp_path: Path):
    source = _write_campaign(tmp_path / "source", [_row("src")])
    target = _write_campaign(tmp_path / "target", [
        _row("match"),
        _row("different-mns", extra={"max_num_seqs": 128}),
    ])
    src_cell = source / "cells" / "src"
    src_cell.mkdir(parents=True)
    (src_cell / "roofline_sweep.json").write_text('{"schema":"roofline_sweep_points_v1"}\n')

    plan = build_plan([target], [source])
    assert len(plan.reuse_candidates) == 1
    cand = plan.reuse_candidates[0]
    assert cand.artifact == "roofline_sweep.json"
    assert cand.source_cell == "src"
    assert cand.target_cell == "match"

    missing_roofline = [
        g for g in plan.missing_groups
        if g.artifact == "roofline_sweep.json"
    ]
    assert len(missing_roofline) == 1
    assert missing_roofline[0].cells[0]["cell_id"] == "different-mns"


def test_capture_plan_resolves_legacy_arm_artifact_dir(tmp_path: Path):
    """Legacy variant imports can have atlas cell_id != cells/<dir>."""
    camp = _write_campaign(tmp_path / "campaign", [
        _row("qwen3-next-xeng-vd-v021-Kengine", extra={"arm": "qwen3-next-xeng-vd-v021"}),
    ])
    physical = camp / "cells" / "qwen3-next-xeng-vd-v021"
    physical.mkdir(parents=True)
    (physical / "dcgm_correlation.json").write_text('{"schema_version":1}\n')

    plan = build_plan([camp], [camp])
    cells = plan.to_dict()["cells"]
    assert cells[0]["cell_id"] == "qwen3-next-xeng-vd-v021-Kengine"
    assert cells[0]["artifact_cell_id"] == "qwen3-next-xeng-vd-v021"
    assert cells[0]["artifacts"]["dcgm_correlation.json"] is True


def test_materialize_reuse_copies_artifact_and_writes_provenance(tmp_path: Path):
    source = _write_campaign(tmp_path / "source", [_row("src")])
    target = _write_campaign(tmp_path / "target", [_row("match")])
    src_cell = source / "cells" / "src"
    src_cell.mkdir(parents=True)
    (src_cell / "kernels.json").write_text('{"schema_version":1}\n')
    plan = build_plan([target], [source])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n")

    result = materialize_reuse(plan_path)
    assert len(result.copied) == 1
    dst = target / "cells" / "match" / "kernels.json"
    assert dst.is_file()
    reuse = json.loads((dst.parent / "capture_reuse.json").read_text())
    assert reuse["schema_version"] == "capture_reuse_v1"
    assert reuse["artifacts"][0]["source_cell"] == "src"


def test_capture_plan_and_materialize_cli(tmp_path: Path):
    source = _write_campaign(tmp_path / "source", [_row("src")])
    target = _write_campaign(tmp_path / "target", [_row("match")])
    src_cell = source / "cells" / "src"
    src_cell.mkdir(parents=True)
    (src_cell / "dcgm_correlation.json").write_text('{"schema_version":1}\n')
    plan_path = tmp_path / "plan.json"

    rc = main([
        "capture_plan",
        "--campaign", str(target),
        "--source-campaign", str(source),
        "--out", str(plan_path),
        "--json",
    ])
    assert rc == 0
    plan = json.loads(plan_path.read_text())
    assert plan["schema_version"] == "capture_plan_v1"
    assert len(plan["reuse_candidates"]) == 1

    rc = main(["materialize_capture_reuse", "--plan", str(plan_path), "--json"])
    assert rc == 0
    assert (target / "cells" / "match" / "dcgm_correlation.json").is_file()
