"""Tests for champion_select: the baseline-vs-top-X cross-engine production pick.

Covers the library (select + gates + 4-layer SoL summary + write_outputs), the
perf-lake champion_v1 table + campaign_v1 champion fields, and the renderer page
8 integration (rendered when champion_select.json is present, omitted loudly when
absent). The multi-variant cross-engine cells live in self-contained tmp_path
fixtures (not the shared 40-cell synthetic_atlas.jsonl, whose exact counts other
tests assert).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.perf_tune_report import champion_select as cs


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _row(cell, eng, c, tput_gpu, tpot):
    return {
        "cell_id": cell, "model": "org/Model-X", "hardware": "GB300", "quant": "NVFP4",
        "tensor_parallel": 4, "parallel_strategy": "TP", "mtp": False,
        "max_num_batched_tokens": 12288, "concurrency": c, "status": "full",
        "ttft_avg_ms": 120.0, "request_throughput_avg": 5.0,
        "output_tps_per_gpu": tput_gpu, "tpot_median_ms": tpot,
        "backend": "sglang-sweep" if eng == "sglang" else "vllm-sweep",
    }


def _stage_campaign(tmp_path: Path, *, with_sol=True) -> Path:
    camp = tmp_path / "20260607T000000Z-model-x-crossengine"
    (camp / "cells").mkdir(parents=True)
    rows = [
        _row("mx-v-base", "vllm", 32, 300.0, 40.0),    # baseline (vLLM)
        _row("mx-v-ep", "vllm", 32, 360.0, 42.0),      # +20% tput, TPOT 42 <= SLO 44 -> PASS
        _row("mx-v-cg", "vllm", 32, 330.0, 55.0),      # +10% tput but TPOT 55 > SLO -> SLO-FAIL
        _row("mx-s-base", "sglang", 32, 280.0, 35.0),  # sglang baseline
        _row("mx-s-attn", "sglang", 32, 410.0, 30.0),  # cross-engine champion
    ]
    (camp / "atlas.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    )
    if with_sol:
        # 4-layer SoL artifacts for the cross-engine champion (mx-s-attn).
        champ = camp / "cells" / "mx-s-attn"
        champ.mkdir(parents=True)
        (champ / "kernels.json").write_text(json.dumps({
            "schema_version": 1, "captured_sources": ["zymtrace"],
            "top_kernels": [], "per_gpu": [], "top_python_during_cuda": [],
            "per_category": {"MoE": 5000, "FMHA": 3000, "NCCL": 1000},
        }))
        (champ / "dcgm_correlation.json").write_text(json.dumps({
            "schema_version": 1, "captured_sources": ["dcgm"], "hw_key": "gb300_nvl72", "queries": [],
            "resources": [
                {"peak_key": "hbm3e_tbps", "metric": "DCGM_FI_PROF_DRAM_ACTIVE", "sol_pct": 24.2},
                {"peak_key": "nvfp4_dense_pflops", "metric": "PIPE_TENSOR_ACTIVE", "sol_pct": 3.1},
            ],
            "per_category_attribution": [{"category": "MoE", "time_share_pct": 55.0, "sol_pct_bw": 0.2}],
        }))
        (champ / "ncu_kernels.json").write_text(json.dumps({
            "schema_version": 1, "captured_sources": ["ncu"], "hw_key": "gb300_nvl72",
            "kernels": [{"name": "k0", "sol_pct": 80}],
        }))
        (champ / "roofline_sweep.json").write_text(json.dumps({
            "schema": "roofline_sweep_points_v1", "hardware": "GB300", "tensor_parallel": 4,
            "quant": "NVFP4", "model": "org/Model-X",
            "decode": [{"phase": "decode", "c": 32, "tensor_active": 0.031, "dram_active": 0.242, "sm_active": 0.40}],
            "prefill": [{"phase": "prefill", "isl": 4096, "tensor_active": 0.6, "dram_active": 0.3, "sm_active": 0.8}],
        }))
    return camp


# --------------------------------------------------------------------------- #
# select() + ranking
# --------------------------------------------------------------------------- #
def test_accuracy_floor_derives_gate_from_local_eval(tmp_path: Path):
    """champion_select reads the campaign's local eval_acc cells (import_model_eval) and
    derives the accuracy gate from --accuracy-floor (worst measured metric vs floor)."""
    camp = _stage_campaign(tmp_path)
    atlas = camp / "atlas.jsonl"
    eval_row = {
        "cell_id": "model-eval", "model": "org/Model-X", "hardware": "GB300", "quant": "NVFP4",
        "tensor_parallel": 4, "parallel_strategy": "TP", "mtp": False,
        "max_num_batched_tokens": 0, "concurrency": 1, "status": "full",
        "extra": {"metric_kind": "eval_acc",
                  "quality_metrics": {"gpqa_acc": 0.52, "mmlu_pro_acc": 0.90}},
    }
    atlas.write_text(atlas.read_text() + json.dumps(eval_row, sort_keys=True) + "\n")
    common = dict(focus="throughput", focus_c=32, top=3, trials=3, same_node=True,
                  workloads_present=("aa", "sonnet", "sharegpt", "random", "code"))
    # floor below the worst measured metric (0.52) -> pass
    res = cs.select(camp, accuracy_floor=0.40, **common)
    acc = next(g for g in res.gates if g.name == "accuracy")
    assert acc.status == "pass"
    assert "gpqa_acc=0.520" in acc.detail
    # the eval cell is not recommended (no perf metric)
    assert res.recommended_cell != "model-eval"
    # floor above the worst measured metric -> fail
    res2 = cs.select(camp, accuracy_floor=0.60, **common)
    assert next(g for g in res2.gates if g.name == "accuracy").status == "fail"


def test_select_picks_cross_engine_champion(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3,
                    trials=3, same_node=True,
                    workloads_present=("aa", "sonnet", "sharegpt", "random", "code"),
                    accuracy_gate="pass")
    assert res.baseline_cell == "mx-v-base"
    assert res.recommended_cell == "mx-s-attn"      # cross-engine winner
    assert res.recommended_engine == "sglang"
    champ = next(v for v in res.variants if v.cell_id == "mx-s-attn")
    assert champ.pct_win_vs_baseline == pytest.approx(36.7, abs=0.2)
    assert champ.slo_verdict == "PASS-SLO"


def test_select_excludes_slo_fail_from_top(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3)
    top_ids = [v.cell_id for v in res.variants if not v.is_baseline]
    # mx-v-cg (TPOT 55 > SLO 44) is SLO-FAIL -> ranked below the 3 SLO-passing arms.
    assert "mx-v-cg" not in top_ids
    assert "mx-s-attn" in top_ids and "mx-v-ep" in top_ids


def test_select_four_layer_sol_summary(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3)
    champ = next(v for v in res.variants if v.cell_id == "mx-s-attn")
    assert champ.sol.l1_present and champ.sol.l2_present and champ.sol.l3_present and champ.sol.l4_present
    assert champ.sol.sol_rigor == "L4"
    assert champ.sol.hbm_pct_sol == pytest.approx(24.2)
    assert champ.has_roofline is True


def test_select_resolves_legacy_arm_artifact_dir(tmp_path: Path):
    camp = tmp_path / "20260607T000000Z-legacy-arm"
    (camp / "cells").mkdir(parents=True)
    row = _row("mx-s-attn-Kengine", "sglang", 32, 410.0, 30.0)
    row["extra"] = {"arm": "mx-s-attn", "engine": "sglang"}
    (camp / "atlas.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n")
    physical = camp / "cells" / "mx-s-attn"
    physical.mkdir(parents=True)
    (physical / "roofline_sweep.json").write_text(json.dumps({
        "schema": "roofline_sweep_points_v1", "hardware": "GB300", "tensor_parallel": 4,
        "quant": "NVFP4", "model": "org/Model-X",
        "decode": [{"phase": "decode", "c": 32, "tensor_active": 0.031, "dram_active": 0.242, "sm_active": 0.40}],
        "prefill": [],
    }))

    res = cs.select(camp, focus="throughput", focus_c=32, top=1)
    variant = res.variants[0]
    assert variant.cell_id == "mx-s-attn-Kengine"
    assert variant.has_roofline is True
    assert variant.sol.sol_rigor == "L3"
    assert res.roofline_overlay["mx-s-attn-Kengine"]["schema"] == "roofline_sweep_points_v1"


# --------------------------------------------------------------------------- #
# DRAFT vs VERDICT gating
# --------------------------------------------------------------------------- #
def test_verdict_when_all_gates_pass(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3,
                    trials=3, same_node=True,
                    workloads_present=("aa", "sonnet", "sharegpt", "random", "code"),
                    accuracy_gate="pass")
    assert res.tier == "verdict"
    assert all(g.status == "pass" for g in res.gates)


def test_draft_when_workloads_missing(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3,
                    trials=3, same_node=True,
                    workloads_present=("aa", "sonnet"),  # missing sharegpt/random/code
                    accuracy_gate="pass")
    assert res.tier == "draft"
    mw = next(g for g in res.gates if g.name == "multi_workload")
    assert mw.status == "fail"


def test_draft_when_accuracy_unknown(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3,
                    trials=3, same_node=True,
                    workloads_present=("aa", "sonnet", "sharegpt", "random", "code"))
    assert res.tier == "draft"  # accuracy_gate defaults unknown


def test_draft_when_champion_not_byte_grounded(tmp_path: Path):
    camp = _stage_campaign(tmp_path, with_sol=False)  # no dcgm/roofline -> no L3
    res = cs.select(camp, focus="throughput", focus_c=32, top=3,
                    trials=3, same_node=True,
                    workloads_present=("aa", "sonnet", "sharegpt", "random", "code"),
                    accuracy_gate="pass")
    assert res.tier == "draft"
    dg = next(g for g in res.gates if g.name == "dcgm_grounded")
    assert dg.status == "fail"


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def test_write_outputs(tmp_path: Path):
    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3)
    json_path, md_path = cs.write_outputs(res, camp)
    assert json_path.name == "champion_select.json" and json_path.is_file()
    assert md_path.name == "CHAMPION.md" and md_path.is_file()
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == "champion_select_v1"
    assert payload["recommended_cell"] == "mx-s-attn"
    md = md_path.read_text()
    assert "RECOMMENDED FOR PRODUCTION" in md
    assert "mx-s-attn" in md


def test_select_missing_atlas_raises(tmp_path: Path):
    camp = tmp_path / "empty"
    camp.mkdir()
    with pytest.raises(FileNotFoundError):
        cs.select(camp)


# --------------------------------------------------------------------------- #
# perf-lake champion_v1 + campaign_v1 champion fields
# --------------------------------------------------------------------------- #
def test_lake_champion_table_and_campaign_fields(tmp_path: Path):
    pytest.importorskip("pyarrow")
    from tools.perf_tune_report.lake_writer import build_champion_table, build_campaign_row
    from tools.perf_tune_report.schema import read_jsonl

    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3,
                    trials=3, same_node=True,
                    workloads_present=("aa", "sonnet", "sharegpt", "random", "code"),
                    accuracy_gate="pass")
    cs.write_outputs(res, camp)
    now = datetime.now(timezone.utc)

    ct = build_champion_table(camp, camp.name, captured_at_utc=now, published_at_utc=now)
    recs = {r["cell_id"]: r for r in ct.to_pylist()}
    assert "mx-v-base" in recs and "mx-s-attn" in recs
    assert recs["mx-s-attn"]["is_recommended"] is True
    assert recs["mx-v-base"]["is_recommended"] is False
    assert recs["mx-s-attn"]["sol_rigor"] == "L4"
    assert recs["mx-s-attn"]["champion_tier"] == "verdict"

    cr = build_campaign_row(camp, read_jsonl(camp / "atlas.jsonl"))
    cols = {n: cr.column(n)[0].as_py() for n in cr.column_names}
    assert cols["recommended_cell"] == "mx-s-attn"
    assert cols["recommended_engine"] == "sglang"
    assert cols["champion_tier"] == "verdict"
    assert cols["champion_baseline_cell"] == "mx-v-base"


def test_lake_champion_table_empty_without_champion_json(tmp_path: Path):
    pytest.importorskip("pyarrow")
    from tools.perf_tune_report.lake_writer import build_champion_table

    camp = _stage_campaign(tmp_path)  # no champion_select.json written
    now = datetime.now(timezone.utc)
    ct = build_champion_table(camp, camp.name, captured_at_utc=now, published_at_utc=now)
    assert ct.num_rows == 0


# --------------------------------------------------------------------------- #
# renderer page 8 integration
# --------------------------------------------------------------------------- #
def test_render_report_includes_champion_page(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from tools.perf_tune_report.renderer.render_report import render_report

    camp = _stage_campaign(tmp_path)
    res = cs.select(camp, focus="throughput", focus_c=32, top=3)
    cs.write_outputs(res, camp)
    status = render_report(camp / "atlas.jsonl", camp / "report.pdf", title="champ test")
    assert any("champion" in p.lower() for p in status.rendered_pages)
    assert (camp / "report.pdf").stat().st_size > 0


def test_render_report_omits_champion_page_loudly_when_absent(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from tools.perf_tune_report.renderer.render_report import render_report

    camp = _stage_campaign(tmp_path)  # no champion_select.json
    status = render_report(camp / "atlas.jsonl", camp / "report.pdf", title="no champ")
    assert any("Champion" in o["page"] for o in status.omitted_pages)
    assert all("champion" not in p.lower() for p in status.rendered_pages)
