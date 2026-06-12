"""Tests for the always-on prefill/decode roofline sweep importer + page."""

import json
from datetime import datetime, timezone
from pathlib import Path

from tools.perf_tune_report.importers.roofline_sweep import import_roofline_sweep_bundle


def _cell(c, isl, osl, tpot, ttft, out_tps, sm, ten, dram):
    return {
        "tag": f"c{c}", "c": c, "isl": isl, "osl": osl, "num_prompts": max(c, 6),
        "bench": {
            "duration": 20.0, "completed": c, "total_input_tokens": isl * c,
            "total_output_tokens": osl * c, "request_throughput": out_tps / max(osl, 1),
            "output_throughput": out_tps, "total_token_throughput": out_tps * 1.5,
            "median_ttft_ms": ttft, "median_tpot_ms": tpot, "median_itl_ms": tpot,
        },
        "dcgm_steady": {
            "sm_active_mean": sm, "tensor_active_mean": ten, "dram_active_mean": dram,
            "fp16_active_mean": 0.0, "nvlink_tx_Bps_mean": 1e9, "nvlink_rx_Bps_mean": 1e9,
            "dmon_samples": 100,
        },
    }


def _write_bundle(tmp: Path) -> Path:
    b = tmp / "rsweep"
    b.mkdir()
    decode = [
        _cell(1, 256, 512, 12.5, 95, 79, 0.29, 0.019, 0.156),
        _cell(64, 256, 512, 24.3, 510, 1734, 0.62, 0.025, 0.373),
    ]
    prefill = [
        {**_cell(16, 2048, 4, 127, 800, 51, 0.80, 0.19, 0.27)},
        {**_cell(16, 8192, 4, 506, 4047, 12, 0.85, 0.19, 0.25)},
    ]
    (b / "decode_sweep.jsonl").write_text("\n".join(json.dumps(x) for x in decode) + "\n")
    (b / "prefill_sweep.jsonl").write_text("\n".join(json.dumps(x) for x in prefill) + "\n")
    return b


def test_import_roofline_sweep_writes_cells_and_dcgm_util(tmp_path):
    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    r = import_roofline_sweep_bundle(
        bundle, camp, overrides={"tensor_parallel": 4, "hardware": "GB300",
                                 "quant": "NVFP4", "model": "zai-org/GLM-5.1",
                                 "cell_id": "glm51-tp4"},
    )
    assert r.decode_points == 2 and r.prefill_points == 2
    dec = camp / "cells" / "glm51-tp4-decode" / "normalized.json"
    rows = json.loads(dec.read_text())
    assert len(rows) == 2
    # DCGM utilization rides in extra (-> atlas_v1.extra_json in the lake)
    util = rows[0]["extra"]["dcgm_util"]
    assert util["dram_active"] == 0.156 and util["tensor_active"] == 0.019
    assert rows[0]["extra"]["phase"] == "decode"
    # analytical roofline coords ride alongside (model in the registry -> resolved)
    rl = rows[0]["extra"]["roofline"]
    assert rl["arithmetic_intensity"] is not None and rl["arithmetic_intensity"] > 0
    assert rl["achieved_tflops_per_gpu"] is not None and rl["achieved_tflops_per_gpu"] > 0
    assert rl["hbm_delivered_Bps_per_gpu"] is not None and rl["hbm_delivered_Bps_per_gpu"] > 0
    # the renderer-page artifact exists with both phases + the embedded shape
    art = json.loads((camp / "cells" / "glm51-tp4-decode" / "roofline_sweep.json").read_text())
    assert art["schema"] == "roofline_sweep_points_v1"
    assert len(art["decode"]) == 2 and len(art["prefill"]) == 2
    assert art["hardware"] == "GB300" and art["tensor_parallel"] == 4
    assert art["kv_dtype"] == "fp8"
    # self-contained analytical shape so the renderer/lake need no registry hit
    assert art["analytical_shape"]["hidden_size"] == 6144
    assert art["analytical_summary"]["is_moe"] is True
    # decode is memory-bound: AI far left of the GB300 NVFP4 ridge (1875)
    assert art["decode"][0]["arithmetic_intensity"] < 1875
    assert art["decode"][1]["arithmetic_intensity"] > art["decode"][0]["arithmetic_intensity"]


def test_import_roofline_sweep_unknown_model_degrades_cleanly(tmp_path):
    """A model not in the registry (and no --model-config) -> no analytical block;
    the renderer falls back to the DCGM proxy. Import must still succeed."""
    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    r = import_roofline_sweep_bundle(
        bundle, camp, overrides={"tensor_parallel": 4, "hardware": "GB300",
                                 "quant": "NVFP4", "model": "acme/Unknown-42B",
                                 "cell_id": "unk"},
    )
    assert r.decode_points == 2
    art = json.loads((camp / "cells" / "unk-decode" / "roofline_sweep.json").read_text())
    assert "analytical_shape" not in art  # unknown family -> no embed
    assert art["decode"][0]["arithmetic_intensity"] is None


def test_import_roofline_sweep_full_context_descriptor_overrides(tmp_path):
    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    import_roofline_sweep_bundle(
        bundle, camp,
        overrides={
            "tensor_parallel": 4,
            "hardware": "GB300",
            "quant": "NVFP4",
            "model": "zai-org/GLM-5.1",
            "cell_id": "glm51-tp4",
            "dataset": "random",
            "cudagraph_mode": "full",
            "gpu_memory_utilization": 0.92,
            "kv_cache_dtype": "fp8_e4m3",
            "image": "infr/vllm:v2.12.3",
            "data_parallel": 2,
            "pipeline_parallel": 1,
            "cache_mode": "cold",
        },
    )
    rows = json.loads((camp / "cells" / "glm51-tp4-decode" / "normalized.json").read_text())
    row = rows[0]
    assert row["dataset"] == "random"
    assert row["cudagraph_mode"] == "full"
    assert row["gpu_memory_utilization"] == 0.92
    assert row["kv_cache_dtype"] == "fp8_e4m3"
    assert row["image"] == "infr/vllm:v2.12.3"
    assert row["data_parallel"] == 2
    assert row["pipeline_parallel"] == 1
    assert row["cache_mode"] == "cold"


def test_build_roofline_v1_lake_table(tmp_path, monkeypatch):
    """roofline_v1 flattens every (c, ISL) point with analytical coords + ceilings
    so Superset can render the roofline scatter + util-vs-C lines from the lake."""
    from tools.perf_tune_report import lake_writer
    from tools.perf_tune_report.schema import read_jsonl
    from tools.perf_tune_report import aggregator

    yaml_path = tmp_path / "perf-tune-report" / "configs" / "sol-ceilings.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        "gb300_nvl72:\n  hw_name: GB300\n"
        "  nvfp4_dense_pflops: {value: 15.0, units: PFLOPS}\n"
        "  hbm3e_tbps: {value: 8.0, units: TB/s}\n"
        "category_ceiling_map:\n  MoE: {metric: nvfp4_dense_pflops, bound: compute}\n"
    )
    monkeypatch.setenv("SOL_CEILINGS_YAML", str(yaml_path))

    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    import_roofline_sweep_bundle(
        bundle, camp, overrides={"tensor_parallel": 4, "hardware": "GB300",
                                 "quant": "NVFP4", "model": "zai-org/GLM-5.1",
                                 "cell_id": "glm51-tp4"},
    )
    aggregator.aggregate(camp)
    rows = read_jsonl(camp / "atlas.jsonl")
    now = datetime.now(timezone.utc)
    table = lake_writer.build_roofline_table(camp, "camp", rows,
                                             captured_at_utc=now, published_at_utc=now)
    # 2 decode + 2 prefill points
    assert table.num_rows == 4
    d = table.to_pylist()
    by_phase = {r["phase"] for r in d}
    assert by_phase == {"decode", "prefill"}
    dec = [r for r in d if r["phase"] == "decode"]
    assert all(r["arithmetic_intensity"] is not None and r["arithmetic_intensity"] > 0 for r in dec)
    assert all(r["achieved_tflops_per_gpu"] is not None for r in dec)
    # byte-grounded HBM% resolved against the ceiling, and per-GPU peaks present
    assert all(r["hbm_delivered_pct"] is not None for r in dec)
    assert dec[0]["compute_peak_pflops_per_gpu"] == 15.0
    assert dec[0]["hbm_peak_tbps_per_gpu"] == 8.0
    assert abs(dec[0]["ridge_ai"] - 1875.0) < 1.0
    # DCGM proxy columns
    assert dec[0]["dram_active_pct"] is not None


def test_source_links_from_provenance_and_registry():
    from tools.perf_tune_report import provenance
    prov = {
        "schema": "experiment_provenance_v1",
        "identity": {"run_id": "x", "title": "GLM-5.1 nvfp4-KV roofline"},
        "source": [
            {"repo": "example/vllm", "branch": "feature/nvfp4-kv",
             "commit": "6554db7dc", "delivery": "overlay",
             "image": "infr/vllm:v2.12.3"},
            {"repo": "example/perf-tune-glm51", "commit": "eafb4b4"},
        ],
    }
    registry = {"repo": "example/vllm", "branches": [
        {"branch": "feature/nvfp4-kv", "purpose": "NVFP4 KV + sparse-MLA decode"},
    ]}
    links = provenance.source_links(prov, registry)
    assert len(links) == 2
    assert links[0]["url"] == "https://github.com/example/vllm/commit/6554db7dc"
    assert links[0]["purpose"] == "NVFP4 KV + sparse-MLA decode"
    assert links[0]["delivery"] == "overlay"
    # harness entry (no branch) -> commit URL, no purpose
    assert links[1]["url"] == "https://github.com/example/perf-tune-glm51/commit/eafb4b4"
    assert provenance.source_links(None) == []


def test_source_page_renders_when_provenance_present(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from tools.perf_tune_report import aggregator
    from tools.perf_tune_report.renderer import render_report

    yaml_path = tmp_path / "sol-ceilings.yaml"
    yaml_path.write_text(
        "gb300_nvl72:\n  hw_name: GB300\n"
        "  nvfp4_dense_pflops: {value: 15.0, units: PFLOPS}\n"
        "  hbm3e_tbps: {value: 8.0, units: TB/s}\n"
        "category_ceiling_map:\n  MoE: {metric: nvfp4_dense_pflops, bound: compute}\n"
    )
    monkeypatch.setenv("SOL_CEILINGS_YAML", str(yaml_path))
    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    import_roofline_sweep_bundle(
        bundle, camp, overrides={"tensor_parallel": 4, "hardware": "GB300",
                                 "quant": "NVFP4", "model": "zai-org/GLM-5.1",
                                 "cell_id": "glm51-tp4"},
    )
    # campaign_init normally writes provenance.json; stage it directly here
    (camp / "provenance.json").write_text(json.dumps({
        "schema": "experiment_provenance_v1",
        "identity": {"run_id": "glm51-roofline", "title": "GLM-5.1 roofline"},
        "source": [{"repo": "example/vllm", "branch": "feature/nvfp4-kv",
                    "commit": "6554db7dc", "delivery": "overlay"}],
    }))
    aggregator.aggregate(camp)
    status = render_report.render_report(camp / "atlas.jsonl", camp / "report.pdf", title="t")
    assert "source under test" in status.to_dict()["rendered_pages"]


def test_roofline_mandatory_gate():
    """Page 7 is mandatory for a throughput/mixed serving campaign; latency /
    accuracy / 0-point campaigns are exempt."""
    from tools.perf_tune_report.lake_writer import RenderStatusSummary, roofline_problems

    omitted = "Prefill/Decode roofline (page 7)"
    # throughput campaign with plot-ready points but page 7 omitted -> gated
    rs = RenderStatusSummary(rendered=True, sol_complete=True, plot_ready_points=9,
                             omitted_pages=omitted, focus="throughput")
    assert roofline_problems(rs)
    # mixed focus too
    rs_mixed = RenderStatusSummary(rendered=True, sol_complete=True, plot_ready_points=5,
                                   omitted_pages=omitted, focus="mixed")
    assert roofline_problems(rs_mixed)
    # latency-only run -> exempt
    rs_lat = RenderStatusSummary(rendered=True, sol_complete=True, plot_ready_points=0,
                                 omitted_pages=omitted, focus="latency")
    assert roofline_problems(rs_lat) == []
    # throughput but 0 plot-ready points (e.g. all cells failed) -> exempt
    rs_zero = RenderStatusSummary(rendered=True, sol_complete=False, plot_ready_points=0,
                                  omitted_pages=omitted, focus="throughput")
    assert roofline_problems(rs_zero) == []
    # page 7 present (not omitted) -> no problem
    rs_ok = RenderStatusSummary(rendered=True, sol_complete=True, plot_ready_points=9,
                                omitted_pages="", focus="throughput")
    assert roofline_problems(rs_ok) == []


def test_roofline_page_renders_and_sets_sol_l3(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from tools.perf_tune_report import aggregator
    from tools.perf_tune_report.renderer import render_report

    # self-contained minimal ceilings YAML (CI-safe; no sibling-repo dependency)
    yaml_path = tmp_path / "sol-ceilings.yaml"
    yaml_path.write_text(
        "gb300_nvl72:\n"
        "  hw_name: NVIDIA GB300\n"
        "  nvfp4_dense_pflops: {value: 15.0, units: PFLOPS}\n"
        "  fp8_dense_pflops: {value: 7.5, units: PFLOPS}\n"
        "  bf16_dense_pflops: {value: 3.75, units: PFLOPS}\n"
        "  hbm3e_tbps: {value: 8.0, units: TB/s}\n"
        "category_ceiling_map:\n"
        "  MoE: {metric: nvfp4_dense_pflops, bound: compute}\n"
    )
    monkeypatch.setenv("SOL_CEILINGS_YAML", str(yaml_path))

    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    import_roofline_sweep_bundle(
        bundle, camp, overrides={"tensor_parallel": 4, "hardware": "GB300",
                                 "quant": "NVFP4", "model": "zai-org/GLM-5.1",
                                 "cell_id": "glm51-tp4"},
    )
    aggregator.aggregate(camp)
    status = render_report.render_report(camp / "atlas.jsonl", camp / "report.pdf",
                                         title="roofline test")
    d = status.to_dict()
    assert "prefill/decode roofline (page 7)" in d["rendered_pages"]
    assert d["sol_complete"] is True
    assert d["sol_rigor"] == "L3"
    assert d["dcgm_grounded"] is True
    assert (camp / "report.pdf").stat().st_size > 1000


def test_per_arm_coverage_uses_extra_arm_artifact_dir(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from tools.perf_tune_report import aggregator
    from tools.perf_tune_report.renderer import render_report

    yaml_path = tmp_path / "sol-ceilings.yaml"
    yaml_path.write_text(
        "gb300_nvl72:\n"
        "  hw_name: NVIDIA GB300\n"
        "  nvfp4_dense_pflops: {value: 15.0, units: PFLOPS}\n"
        "  hbm3e_tbps: {value: 8.0, units: TB/s}\n"
        "category_ceiling_map:\n"
        "  MoE: {metric: nvfp4_dense_pflops, bound: compute}\n"
    )
    monkeypatch.setenv("SOL_CEILINGS_YAML", str(yaml_path))

    bundle = _write_bundle(tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    import_roofline_sweep_bundle(
        bundle, camp, overrides={"tensor_parallel": 4, "hardware": "GB300",
                                 "quant": "NVFP4", "model": "zai-org/GLM-5.1",
                                 "cell_id": "physical-arm"},
    )
    logical = {
        "cell_id": "physical-arm-Kengine", "model": "zai-org/GLM-5.1", "hardware": "GB300",
        "quant": "NVFP4", "tensor_parallel": 4, "parallel_strategy": "TP",
        "mtp": False, "max_num_batched_tokens": 12288, "concurrency": 64,
        "status": "full", "ttft_avg_ms": 1.0, "request_throughput_avg": 1.0,
        "output_tps_per_gpu": 1.0, "backend": "vllm-sweep",
        "extra": {"arm": "physical-arm"},
    }
    logical_dir = camp / "cells" / "logical"
    logical_dir.mkdir()
    (logical_dir / "normalized.json").write_text(json.dumps([logical]))

    aggregator.aggregate(camp)
    status = render_report.render_report(camp / "atlas.jsonl", camp / "report.pdf", title="t")
    d = status.to_dict()
    assert d["arms_total"] == 1
    assert d["arms_with_roofline"] == 1
    assert d["arms_uncovered"] == []
    assert d["sol_per_arm_complete"] is True
