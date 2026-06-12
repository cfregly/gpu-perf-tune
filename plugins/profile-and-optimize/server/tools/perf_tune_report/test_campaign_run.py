"""Unit tests for the v1.20.0 campaign_run orchestrator (Phase 2b)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from tools.perf_tune_report.orchestrator.campaign_run import (
    CampaignRunResult,
    CellPlan,
    CellStepResult,
    StepFns,
    _verdict_rollup,
    run_campaign,
    run_one_cell,
)


# -----------------------------------------------------------------------------
# CellPlan schema
# -----------------------------------------------------------------------------


def test_cell_plan_basic():
    cell = CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1, 4, 16))
    assert cell.id == "cell-1"
    assert cell.backend == "vllm-sweep"
    assert cell.concurrencies == (1, 4, 16)


def test_cell_plan_rejects_invalid_id():
    with pytest.raises(ValueError, match="alphanumeric"):
        CellPlan(id="bad id with spaces!", backend="vllm-sweep", concurrencies=(1,))


def test_cell_plan_rejects_invalid_backend():
    with pytest.raises(ValueError, match="must be one of"):
        CellPlan(id="cell-1", backend="bogus-backend", concurrencies=(1,))


def test_cell_plan_accepts_trtllm_backend():
    cell = CellPlan(id="cell-trt", backend="trtllm", concurrencies=(1,))
    assert cell.backend == "trtllm"


def test_cell_plan_accepts_aa_backend():
    cell = CellPlan(
        id="aa-10k",
        backend="aa",
        concurrencies=(1, 10),
        backend_config={"model": "m", "url": "http://x:8000", "shape": "aa-10k"},
    )
    assert cell.backend == "aa"
    assert cell.backend_config["shape"] == "aa-10k"


def test_cell_plan_rejects_empty_concurrencies():
    with pytest.raises(ValueError, match="non-empty"):
        CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=())


# -----------------------------------------------------------------------------
# Verdict rollup
# -----------------------------------------------------------------------------


def _mk_cell_result(cell_id: str, verdict: str, bench_ok: bool = True) -> CellStepResult:
    return CellStepResult(
        cell_id=cell_id, started_at="t", ended_at="t", elapsed_s=0.0,
        drain_ok=True, helm_ok=True, warmup_ok=True, bench_ok=bench_ok,
        zymtrace_ok=True, import_ok=True, aggregate_ok=True,
        render_ok=True, baseline_record_ok=True,
        baseline_diff_verdict=verdict, resume_ok=True,
    )


def test_verdict_rollup_all_green():
    assert _verdict_rollup([_mk_cell_result("a", "GREEN"), _mk_cell_result("b", "GREEN")]) == "GREEN"


def test_verdict_rollup_any_red_is_red():
    assert _verdict_rollup([_mk_cell_result("a", "GREEN"), _mk_cell_result("b", "RED")]) == "RED"


def test_verdict_rollup_yellow_no_red():
    assert _verdict_rollup([_mk_cell_result("a", "GREEN"), _mk_cell_result("b", "YELLOW")]) == "YELLOW"


def test_verdict_rollup_empty_is_na():
    assert _verdict_rollup([]) == "NA"


# -----------------------------------------------------------------------------
# run_one_cell — happy path + always-resume contract
# -----------------------------------------------------------------------------


def _make_stub_fns(
    *,
    drain_ok: bool = True,
    helm_ok: bool = True,
    warmup_ok: bool = True,
    bench_ok: bool = True,
    zymtrace_ok: bool = True,
    import_ok: bool = True,
    aggregate_ok: bool = True,
    render_ok: bool = True,
    baseline_record_ok: bool = True,
    diff_verdict: str = "GREEN",
    resume_ok: bool = True,
    resume_raises: bool = False,
) -> StepFns:
    """Build a StepFns where every step returns the configured outcome."""
    def _resume_fn(nodes, cell_id):
        if resume_raises:
            raise RuntimeError("simulated resume failure")
        return resume_ok

    return StepFns(
        drain=lambda nodes, cell_id: drain_ok,
        helm_upgrade=lambda ns, rel, chart, overrides: helm_ok,
        warmup=lambda ns, rel, cell_id: warmup_ok,
        bench=lambda cdir, cell: bench_ok,
        zymtrace=lambda cdir, cell_id: zymtrace_ok,
        import_bundle=lambda src, dst, cell_id: import_ok,
        aggregate=lambda cdir: aggregate_ok,
        render=lambda cdir, pdf: render_ok,
        baseline_record=lambda cdir, cell_id: baseline_record_ok,
        baseline_diff=lambda cdir, cell_id, comp: diff_verdict,
        resume=_resume_fn,
    )


def test_run_one_cell_happy_path(tmp_path):
    cell = CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1, 4))
    fns = _make_stub_fns()
    result = run_one_cell(
        cell,
        campaign_dir=tmp_path,
        target_namespace="ns",
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        drain_nodes=("nodeA", "nodeB"),
        comparator_baseline="",
        step_fns=fns,
    )
    assert result.cell_id == "cell-1"
    assert result.drain_ok
    assert result.helm_ok
    assert result.bench_ok
    assert result.baseline_diff_verdict == "GREEN"
    assert result.resume_ok  # always-resume worked


def test_run_one_cell_always_resumes_on_bench_failure(tmp_path):
    """If bench fails mid-cell, the Slurm-on-K8s resume MUST still run."""
    cell = CellPlan(id="cell-fail", backend="vllm-sweep", concurrencies=(1,))
    fns = _make_stub_fns(bench_ok=False)
    result = run_one_cell(
        cell,
        campaign_dir=tmp_path,
        target_namespace="ns",
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        drain_nodes=("nodeA",),
        comparator_baseline="",
        step_fns=fns,
    )
    assert result.drain_ok       # drain succeeded
    assert not result.bench_ok    # bench failed
    assert result.resume_ok       # CRITICAL: resume still ran in finally


def test_run_one_cell_handles_resume_exception(tmp_path):
    """If resume itself raises, the cell result records that without dying."""
    cell = CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,))
    fns = _make_stub_fns(resume_raises=True)
    result = run_one_cell(
        cell,
        campaign_dir=tmp_path,
        target_namespace="ns",
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        drain_nodes=("nodeA",),
        comparator_baseline="",
        step_fns=fns,
    )
    assert not result.resume_ok  # explicit False
    assert any("resume raised" in n for n in result.notes)


def test_run_one_cell_no_drain_nodes_no_drain(tmp_path):
    """When drain_nodes is empty, drain/resume are no-ops (pass)."""
    cell = CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,))
    fns = _make_stub_fns()
    result = run_one_cell(
        cell,
        campaign_dir=tmp_path,
        target_namespace="ns",
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        drain_nodes=(),
        comparator_baseline="",
        step_fns=fns,
    )
    assert result.drain_ok
    assert result.resume_ok


def test_run_one_cell_helm_fail_aborts_cell(tmp_path):
    """helm upgrade returning False aborts subsequent steps for this cell."""
    cell = CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,))
    fns = _make_stub_fns(helm_ok=False)
    result = run_one_cell(
        cell,
        campaign_dir=tmp_path,
        target_namespace="ns",
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        drain_nodes=("nodeA",),
        comparator_baseline="",
        step_fns=fns,
    )
    assert result.drain_ok
    assert not result.helm_ok
    assert not result.bench_ok  # skipped because helm failed
    assert result.resume_ok  # always-resume still ran


# -----------------------------------------------------------------------------
# Endpoint-only backends (aa / aiperf): skip helm + warmup
# -----------------------------------------------------------------------------


def test_run_one_cell_aa_skips_helm_and_warmup(tmp_path):
    """aa cells target an existing endpoint URL: helm + warmup must NOT run."""
    helm_calls: list[int] = []
    warmup_calls: list[int] = []
    fns = StepFns(
        drain=lambda nodes, cell_id: True,
        helm_upgrade=lambda ns, rel, chart, ov: (helm_calls.append(1), True)[1],
        warmup=lambda ns, rel, cell_id: (warmup_calls.append(1), True)[1],
        bench=lambda cdir, cell: True,
        zymtrace=lambda cdir, cell_id: True,
        import_bundle=lambda src, dst, cell_id: True,
        aggregate=lambda cdir: True,
        render=lambda cdir, pdf: True,
        baseline_record=lambda cdir, cell_id: True,
        baseline_diff=lambda cdir, cell_id, comp: "NA",
        resume=lambda nodes, cell_id: True,
    )
    cell = CellPlan(
        id="aa-10k",
        backend="aa",
        concurrencies=(1, 10),
        backend_config={"model": "m", "url": "http://x:8000", "shape": "aa-10k"},
    )
    result = run_one_cell(
        cell,
        campaign_dir=tmp_path,
        target_namespace="ns",
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        drain_nodes=(),
        comparator_baseline="",
        step_fns=fns,
    )
    assert helm_calls == []   # helm_upgrade never invoked
    assert warmup_calls == []  # warmup never invoked
    assert result.helm_ok      # recorded as no-op pass
    assert result.warmup_ok
    assert result.bench_ok


# -----------------------------------------------------------------------------
# step_bench backend dispatch (aa / aiperf) — production_steps
# -----------------------------------------------------------------------------


def test_step_bench_aa_returns_false_when_config_incomplete(tmp_path):
    from tools.perf_tune_report.orchestrator.production_steps import step_bench
    # missing url + shape
    cell = CellPlan(
        id="aa-1k", backend="aa", concurrencies=(1,),
        backend_config={"model": "m"},
    )
    assert step_bench(tmp_path, cell) is False


def test_step_bench_aiperf_returns_false_when_config_incomplete(tmp_path):
    from tools.perf_tune_report.orchestrator.production_steps import step_bench
    cell = CellPlan(
        id="replay", backend="aiperf", concurrencies=(1,), backend_config={},
    )
    assert step_bench(tmp_path, cell) is False


def test_step_bench_aa_happy_path(tmp_path, monkeypatch):
    """A complete aa cell delegates to aa_bench.run_cell and reports True."""
    from tools.perf_tune_report.orchestrator.production_steps import step_bench
    from tools.perf_tune_report.runners import aa_bench

    captured: dict = {}

    class _FakeRes:
        row_count = 2

    def _fake_run_cell(cell_cfg, campaign_dir, **kwargs):
        captured.update(kwargs)
        captured["cell_id"] = cell_cfg.cell_id
        captured["concurrencies"] = cell_cfg.concurrencies
        return _FakeRes()

    monkeypatch.setattr(aa_bench, "run_cell", _fake_run_cell)
    cell = CellPlan(
        id="aa-10k",
        backend="aa",
        concurrencies=(1, 10),
        helm_overrides={"model": "GLM-5.1-NVFP4", "tensor_parallel": 8},
        backend_config={
            "model": "served-model",
            "url": "http://x:8000",
            "shape": "aa-10k",
            "mode": "synthetic",
            "request_count": 10,
        },
    )
    assert step_bench(tmp_path, cell) is True
    assert captured["shape_name"] == "aa-10k"
    assert captured["model"] == "served-model"
    assert captured["mode"] == "synthetic"
    assert captured["concurrencies"] == (1, 10)


def test_step_bench_aiperf_happy_path(tmp_path, monkeypatch):
    from tools.perf_tune_report.orchestrator.production_steps import step_bench
    from tools.perf_tune_report.runners import aiperf_bench

    captured: dict = {}

    class _FakeRes:
        row_count = 1

    def _fake_run_cell(cell_cfg, campaign_dir, **kwargs):
        captured.update(kwargs)
        return _FakeRes()

    monkeypatch.setattr(aiperf_bench, "run_cell", _fake_run_cell)
    cell = CellPlan(
        id="replay",
        backend="aiperf",
        concurrencies=(1, 8),
        backend_config={
            "namespace": "<slurm-namespace>",
            "bench_pod": "aiperf-bench",
            "kube_context": "ctx",
            "endpoint_url": "http://svc:8000",
            "served_model": "GLM-5.1-NVFP4",
        },
    )
    assert step_bench(tmp_path, cell) is True
    assert captured["endpoint_url"] == "http://svc:8000"
    assert captured["dataset_split"] == "2025_07"  # default applied


# -----------------------------------------------------------------------------
# run_campaign — multi-cell + fail-fast
# -----------------------------------------------------------------------------


def test_run_campaign_two_cells_green(tmp_path):
    (tmp_path / "cells").mkdir()
    cells = [
        CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,)),
        CellPlan(id="cell-2", backend="vllm-sweep", concurrencies=(1,)),
    ]
    fns = _make_stub_fns()
    result = run_campaign(
        cells,
        campaign_dir=tmp_path,
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        step_fns=fns,
    )
    assert result.cells_attempted == 2
    assert result.cells_completed == 2
    assert result.overall_verdict == "GREEN"
    assert result.aborted_at_cell is None


def test_run_campaign_fail_fast_on_red(tmp_path):
    (tmp_path / "cells").mkdir()
    cells = [
        CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,)),
        CellPlan(id="cell-2-bad", backend="vllm-sweep", concurrencies=(1,)),
        CellPlan(id="cell-3", backend="vllm-sweep", concurrencies=(1,)),
    ]
    # Build a custom stub that returns RED only for cell-2-bad.
    diffs = {"cell-1": "GREEN", "cell-2-bad": "RED", "cell-3": "GREEN"}
    def _baseline_diff(cdir, cell_id, comp):
        return diffs.get(cell_id, "NA")
    fns = StepFns(
        drain=lambda nodes, cell_id: True,
        helm_upgrade=lambda ns, rel, chart, overrides: True,
        warmup=lambda ns, rel, cell_id: True,
        bench=lambda cdir, cell: True,
        zymtrace=lambda cdir, cell_id: True,
        import_bundle=lambda src, dst, cell_id: True,
        aggregate=lambda cdir: True,
        render=lambda cdir, pdf: True,
        baseline_record=lambda cdir, cell_id: True,
        baseline_diff=_baseline_diff,
        resume=lambda nodes, cell_id: True,
    )
    result = run_campaign(
        cells,
        campaign_dir=tmp_path,
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        step_fns=fns,
    )
    # Aborted after the RED cell; cell-3 never ran
    assert result.cells_attempted == 2
    assert result.cells_skipped == 1
    assert result.aborted_at_cell == "cell-2-bad"
    assert result.overall_verdict == "RED"


def test_run_campaign_continue_on_red(tmp_path):
    (tmp_path / "cells").mkdir()
    cells = [
        CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,)),
        CellPlan(id="cell-2-bad", backend="vllm-sweep", concurrencies=(1,)),
        CellPlan(id="cell-3", backend="vllm-sweep", concurrencies=(1,)),
    ]
    diffs = {"cell-1": "GREEN", "cell-2-bad": "RED", "cell-3": "GREEN"}
    fns = StepFns(
        drain=lambda nodes, cell_id: True,
        helm_upgrade=lambda ns, rel, chart, overrides: True,
        warmup=lambda ns, rel, cell_id: True,
        bench=lambda cdir, cell: True,
        zymtrace=lambda cdir, cell_id: True,
        import_bundle=lambda src, dst, cell_id: True,
        aggregate=lambda cdir: True,
        render=lambda cdir, pdf: True,
        baseline_record=lambda cdir, cell_id: True,
        baseline_diff=lambda cdir, cell_id, comp: diffs.get(cell_id, "NA"),
        resume=lambda nodes, cell_id: True,
    )
    result = run_campaign(
        cells,
        campaign_dir=tmp_path,
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        step_fns=fns,
        continue_on_red=True,
    )
    # All 3 attempted
    assert result.cells_attempted == 3
    assert result.aborted_at_cell is None
    assert result.overall_verdict == "RED"  # because a RED is in the set


def test_run_campaign_writes_receipts(tmp_path):
    cells = [CellPlan(id="cell-1", backend="vllm-sweep", concurrencies=(1,))]
    fns = _make_stub_fns()
    result = run_campaign(
        cells,
        campaign_dir=tmp_path,
        target_release="rel",
        chart_dir=tmp_path,
        base_values=tmp_path / "values.yaml",
        step_fns=fns,
    )
    receipt = tmp_path / "commands" / "cell-cell-1-receipt.json"
    assert receipt.is_file()
    data = json.loads(receipt.read_text())
    assert data["cell_id"] == "cell-1"
    assert data["baseline_diff_verdict"] == "GREEN"


# -----------------------------------------------------------------------------
# CONTRACT integration — verb is in CLI
# -----------------------------------------------------------------------------


def test_contract_verb_set():
    """The perf_tune_report CONTRACT verb set (champion_select + import_variant_ab
    joined in v1.66.0)."""
    from tools.perf_tune_report.perf_tune_report_cli import CONTRACT
    assert set(CONTRACT) == {
        "campaign_init", "cell_run", "atlas_aggregate",
        "report_render", "report_smoke", "publish_to_lake",
        "import_perf_bench", "campaign_run",
        "kernel_profile", "graph_diff",
        "raw_bench_compare", "import_ncu", "import_nsys", "dcgm_correlate",
        "experiments_index", "tpm_summary", "value_view",
        "import_roofline_sweep", "fleet_leaderboard",
        "champion_select", "import_variant_ab",
        "capture_plan", "materialize_capture_reuse",
        "experiment_inventory", "portability_view", "import_model_eval", "trend_view",
        "kernel_reproducer_scaffold", "import_workloads",
    }


def test_contract_campaign_run_is_ack_gated():
    from tools.perf_tune_report.perf_tune_report_cli import CONTRACT
    assert CONTRACT["campaign_run"]["safety"] == "submits_jobs"
    assert CONTRACT["campaign_run"]["ack"] == "--i-understand-this-submits-jobs"
    assert "--config" in CONTRACT["campaign_run"]["required"]
    assert "--campaign" in CONTRACT["campaign_run"]["required"]
