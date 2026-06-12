"""Unit tests for the perf_tune_report fleet_leaderboard verb (cross-model leaderboards)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.fleet_leaderboard import (
    build_aa,
    build_pareto,
    build_throughput,
    canon_model,
    dominates,
    read_all_rows,
    write_leaderboards,
)
from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.schema import AtlasCell, write_jsonl


def _row(**kw) -> AtlasCell:
    base = dict(cell_id="c1", model="GLM-5.1-NVFP4", hardware="GB300", quant="NVFP4",
                tensor_parallel=4, parallel_strategy="TP", mtp=False,
                max_num_batched_tokens=8192, concurrency=1, status="full",
                ttft_avg_ms=120.0, request_throughput_avg=0.2, output_tps_per_user=70.0,
                output_tps_per_gpu=400.0, tpot_median_ms=13.0, backend="vllm-sweep")
    base.update(kw)
    return AtlasCell(**base)


def _stage(campaigns_root: Path, name: str, rows: list[AtlasCell]) -> Path:
    d = campaigns_root / name
    d.mkdir(parents=True)
    write_jsonl(rows, d / "atlas.jsonl")
    return d


# --------------------------------------------------------------------------- #
def test_canon_model_collapses_drift():
    assert canon_model("zai-org/GLM-5.1") == "GLM-5.1"
    assert canon_model("GLM-5.1-NVFP4") == "GLM-5.1"
    assert canon_model("nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4") == "Qwen3-Next-80B-A3B"
    assert canon_model("google/gemma-4-26B-A4B-it") == "Gemma-4-26B-A4B"
    assert "GGUF" in canon_model("GLM-5.1-UD-Q2_K_XL (unsloth GGUF, llama.cpp)")


def test_build_throughput_picks_peak_and_excludes_aa():
    rows = [
        _row(model="M-NVFP4", cell_id="sweep", concurrency=32, output_tps_per_gpu=1000.0),
        _row(model="M-NVFP4", cell_id="sweep", concurrency=256, output_tps_per_gpu=2500.0),  # peak
        _row(model="M-NVFP4", cell_id="sweep", concurrency=128, output_tps_per_gpu=1800.0),
        _row(model="M-NVFP4", cell_id="aa-1k", concurrency=1, output_tps_per_gpu=99999.0),  # excluded
    ]
    cfgs = build_throughput(rows)
    assert len(cfgs) == 1
    c = cfgs[0]
    assert c["model"] == "M" and c["quant"] == "NVFP4" and c["tp"] == 4
    assert c["tps_gpu"] == 2500.0 and c["conc"] == 256  # peak, not the aa row


def test_build_throughput_skips_failed_status():
    rows = [_row(model="X", cell_id="s", status="failed", output_tps_per_gpu=9.0)]
    assert build_throughput(rows) == []


def test_resolve_gpu_hr_override_and_config(tmp_path: Path):
    from tools.perf_tune_report.fleet_leaderboard import resolve_gpu_hr, GPU_HR_DEFAULT
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "cost.yaml").write_text("usd_per_gpu_hour:\n  GB300: 9.99\n  default: 4.00\n")
    assert resolve_gpu_hr("GB300", cfg, 12.0) == 12.0   # --gpu-hr override wins
    assert resolve_gpu_hr("GB300", cfg) == 9.99          # per-hardware rate
    assert resolve_gpu_hr("UNKNOWN-HW", cfg) == 4.00     # falls back to default
    assert resolve_gpu_hr("GB300", tmp_path / "nope") == GPU_HR_DEFAULT  # no config -> built-in default


def test_build_quality_extracts_eval_acc_and_renders():
    from tools.perf_tune_report.fleet_leaderboard import build_quality, build_pareto, render_pareto_md
    eval_row = _row(
        model="zai-org/GLM-5.1", cell_id="model-eval", concurrency=1,
        ttft_avg_ms=None, request_throughput_avg=None, output_tps_per_user=None,
        output_tps_per_gpu=None, tpot_median_ms=None, max_num_batched_tokens=0,
        backend="aiperf", captured_at="2026-06-07T00:00:00Z",
        extra={"metric_kind": "eval_acc",
               "quality_metrics": {"gpqa_acc": 0.52, "mmlu_pro_acc": 0.903}},
    )
    q = build_quality([eval_row])
    assert q["GLM-5.1"]["gpqa_acc"] == 0.52
    assert q["GLM-5.1"]["mmlu_pro_acc"] == 0.903
    # render shows the quality section (kept separate from the perf Pareto)
    md = render_pareto_md(build_pareto([eval_row]), "GB300", 8.60, q)
    assert "Model quality (lm-eval serving evals" in md
    assert "gpqa_acc" in md and "GLM-5.1" in md
    # empty quality -> a "none measured yet" note, not a crash
    assert "None measured yet" in render_pareto_md({}, "GB300", 8.60, {})


def test_build_aa_latest_captured_wins():
    rows = [
        _row(model="A", cell_id="aa-1k", concurrency=1, output_tps_per_user=100.0,
             captured_at="2026-06-01T00:00:00Z"),
        _row(model="A", cell_id="aa-1k", concurrency=1, output_tps_per_user=222.0,
             captured_at="2026-06-05T00:00:00Z"),  # newer wins
    ]
    aa = build_aa(rows)
    assert aa["A"][("aa-1k", 1)]["opu"] == 222.0


def test_build_pareto_frontier_and_dominance():
    # A dominates B on all 3 axes; C wins latency but loses throughput -> A,C frontier; B dominated.
    def model(name, opu, ttft, tps):
        return [
            _row(model=name, cell_id="aa-1k", concurrency=1, output_tps_per_user=opu, ttft_avg_ms=ttft),
            _row(model=name, cell_id="sweep", concurrency=256, tensor_parallel=4, output_tps_per_gpu=tps),
        ]
    rows = model("AAA", 200, 100, 2000) + model("BBB", 100, 300, 1000) + model("CCC", 250, 50, 500)
    M = build_pareto(rows)
    assert set(M) == {"AAA", "BBB", "CCC"}
    assert dominates(M["AAA"], M["BBB"]) is True
    assert dominates(M["CCC"], M["BBB"]) is False  # C loses on throughput
    frontier = [n for n in M if not any(dominates(M[o], M[n]) for o in M if o != n)]
    assert set(frontier) == {"AAA", "CCC"}


def test_build_pareto_requires_all_three_axes():
    # Only an AA row, no throughput row -> excluded from the Pareto.
    rows = [_row(model="OnlyAA", cell_id="aa-1k", concurrency=1, output_tps_per_user=150.0)]
    assert build_pareto(rows) == {}


def test_read_all_rows_filters_hardware(tmp_path: Path):
    _stage(tmp_path, "camp-gb", [_row(model="GBmodel", hardware="GB300")])
    _stage(tmp_path, "camp-b200", [_row(model="B200model", hardware="B200")])
    rows = read_all_rows(tmp_path, hardware_filter="GB300")
    assert all("GB300" in (r.hardware or "") for r in rows)
    assert any(r.model == "GBmodel" for r in rows)
    assert not any(r.model == "B200model" for r in rows)


def test_write_leaderboards_emits_three_files(tmp_path: Path):
    rows = (
        [_row(model="Fast", cell_id="aa-1k", concurrency=1, output_tps_per_user=220.0, ttft_avg_ms=100.0),
         _row(model="Fast", cell_id="sweep", concurrency=256, output_tps_per_gpu=2900.0)]
    )
    out = tmp_path / "out"
    res = write_leaderboards(rows, out, hw="GB300", gpu_hr=8.60)
    for key in ("aa", "throughput", "pareto"):
        assert Path(res[key]).is_file()
    assert "Fast" in Path(res["throughput"]).read_text()
    assert res["aa_models"] == 1 and res["pareto_models"] == 1


def test_cli_fleet_leaderboard_smoke(tmp_path: Path, capsys):
    campaigns = tmp_path / "campaigns"
    _stage(campaigns, "camp-20260607T000000Z", [
        _row(model="Champ", cell_id="aa-1k", concurrency=1, output_tps_per_user=221.0, ttft_avg_ms=170.0),
        _row(model="Champ", cell_id="sweep", concurrency=160, output_tps_per_gpu=2916.0),
    ])
    rc = main(["fleet_leaderboard", "--campaigns-dir", str(campaigns),
               "--out", str(tmp_path / "out"), "--json"])
    assert rc == 0
    assert (tmp_path / "out" / "FLEET-MODEL-SELECTION-GB300.md").is_file()
    assert "Champ" in (tmp_path / "out" / "THROUGHPUT-FLEET-LEADERBOARD-GB300.md").read_text()
