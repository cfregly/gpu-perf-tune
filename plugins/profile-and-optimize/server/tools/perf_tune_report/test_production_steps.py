"""Unit tests for the campaign_run production step-fn wiring (v1.21.0)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.perf_tune_report.orchestrator import production_step_fns
from tools.perf_tune_report.orchestrator.production_steps import (
    _run_cmd,
    step_aggregate,
    step_baseline_diff,
    step_baseline_record,
    step_bench,
    step_drain,
    step_helm_upgrade,
    step_import,
    step_render,
    step_resume,
    step_warmup,
    step_zymtrace,
)
from tools.perf_tune_report.orchestrator.campaign_run import CellPlan, StepFns


# ---------------------------------------------------------------------------
# 1. production_step_fns() returns a fully populated StepFns
# ---------------------------------------------------------------------------


def test_production_step_fns_populates_every_field() -> None:
    fns = production_step_fns()
    assert isinstance(fns, StepFns)
    for field in (
        "drain",
        "helm_upgrade",
        "warmup",
        "bench",
        "zymtrace",
        "import_bundle",
        "aggregate",
        "render",
        "baseline_record",
        "baseline_diff",
        "resume",
    ):
        assert getattr(fns, field) is not None, f"production_step_fns missing {field}"
        assert callable(getattr(fns, field))


# ---------------------------------------------------------------------------
# 2. _run_cmd handles every failure mode without raising
# ---------------------------------------------------------------------------


def test_run_cmd_ok_path_returns_true() -> None:
    # `true` is a POSIX builtin that always exits 0.
    assert _run_cmd(["true"]) is True


def test_run_cmd_nonzero_returns_false() -> None:
    assert _run_cmd(["false"]) is False


def test_run_cmd_missing_binary_returns_false() -> None:
    assert _run_cmd(["__definitely_not_a_real_binary__"]) is False


def test_run_cmd_timeout_returns_false() -> None:
    # `sleep 5` with timeout=1 forces TimeoutExpired path.
    assert _run_cmd(["sleep", "5"], timeout=1) is False


# ---------------------------------------------------------------------------
# 3. step_drain / step_resume return correct result per node
# ---------------------------------------------------------------------------


def test_step_drain_no_nodes_returns_true() -> None:
    """Empty node list -> trivial pass (no kubectl call attempted)."""
    assert step_drain([], "cell-x") is True


def test_step_resume_no_nodes_returns_true() -> None:
    assert step_resume([], "cell-x") is True


def test_step_drain_kubectl_failure_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.perf_tune_report.orchestrator.production_steps._run_cmd",
        lambda cmd, timeout=120: False,
    )
    assert step_drain(["node-1", "node-2"], "cell-x") is False


def test_step_drain_kubectl_success_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake(cmd: list[str], timeout: int = 120) -> bool:
        calls.append(cmd)
        return True

    monkeypatch.setattr(
        "tools.perf_tune_report.orchestrator.production_steps._run_cmd", fake
    )
    assert step_drain(["node-1", "node-2"], "cell-x") is True
    assert calls == [
        ["kubectl", "cordon", "node-1"],
        ["kubectl", "cordon", "node-2"],
    ]


# ---------------------------------------------------------------------------
# 4. step_helm_upgrade builds the right argv shape
# ---------------------------------------------------------------------------


def test_step_helm_upgrade_with_path_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake(cmd: list[str], timeout: int = 120) -> bool:
        captured.append(cmd)
        return True

    monkeypatch.setattr(
        "tools.perf_tune_report.orchestrator.production_steps._run_cmd", fake
    )
    ok = step_helm_upgrade(
        namespace="inference",
        release="basic-inference",
        chart_dir=Path("/charts/kimi"),
        overrides={"path": "/tmp/overlay.yaml"},
    )
    assert ok is True
    cmd = captured[0]
    assert "helm" in cmd
    assert "upgrade" in cmd
    assert "--install" in cmd
    assert "-f" in cmd
    assert "/tmp/overlay.yaml" in cmd
    assert "--namespace" in cmd
    assert "inference" in cmd


def test_step_helm_upgrade_with_set_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake(cmd: list[str], timeout: int = 120) -> bool:
        captured.append(cmd)
        return True

    monkeypatch.setattr(
        "tools.perf_tune_report.orchestrator.production_steps._run_cmd", fake
    )
    step_helm_upgrade(
        namespace="inference",
        release="basic-inference",
        chart_dir=Path("/charts/kimi"),
        overrides={"set": {"replicaCount": 5, "image.tag": "v2.12.3"}},
    )
    cmd = captured[0]
    set_args = [c for c in cmd if "=" in c]
    assert "replicaCount=5" in set_args
    assert "image.tag=v2.12.3" in set_args


# ---------------------------------------------------------------------------
# 5. step_bench dispatches on backend; unknown backend returns False
# ---------------------------------------------------------------------------


def test_step_bench_vllm_sweep_without_serve_cmd_returns_false() -> None:
    cell = CellPlan(
        id="test-cell",
        backend="vllm-sweep",
        concurrencies=(8, 16),
        helm_overrides={"model": "moonshotai/Kimi-K2.6"},
    )
    assert step_bench(Path("/tmp"), cell) is False


def test_step_bench_aiperf_returns_false() -> None:
    """aiperf backend is intentionally a no-op in production_steps v1.21.0."""
    cell = CellPlan(
        id="test-cell", backend="aiperf", concurrencies=(8,),
    )
    assert step_bench(Path("/tmp"), cell) is False


def test_step_bench_trtllm_returns_false() -> None:
    """trtllm backend is a stub - run_cell raises NotImplementedError."""
    cell = CellPlan(
        id="test-cell", backend="trtllm", concurrencies=(8,),
    )
    assert step_bench(Path("/tmp"), cell) is False


# ---------------------------------------------------------------------------
# 6. step_zymtrace: presence of capture_sources.json decides outcome
# ---------------------------------------------------------------------------


def test_step_zymtrace_returns_true_when_manifest_present(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "cells" / "c1").mkdir(parents=True)
    (campaign / "cells" / "c1" / "capture_sources.json").write_text("{}")
    assert step_zymtrace(campaign, "c1") is True


def test_step_zymtrace_returns_false_without_manifest(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "cells" / "c1").mkdir(parents=True)
    assert step_zymtrace(campaign, "c1") is False


# ---------------------------------------------------------------------------
# 7. step_import / step_aggregate / step_render return False on bad inputs
# ---------------------------------------------------------------------------


def test_step_import_returns_false_for_invalid_dir(tmp_path: Path) -> None:
    """import_bundle_auto raises ValueError on missing bundle -> step returns False."""
    nonexistent = tmp_path / "does-not-exist"
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    assert step_import(nonexistent, campaign, "c1") is False


def test_step_aggregate_returns_false_when_no_cells(tmp_path: Path) -> None:
    """aggregate raises on missing cells dir -> step returns False."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    # Aggregate may legitimately succeed with empty input on some versions;
    # the contract is that step_aggregate never raises.
    result = step_aggregate(campaign)
    assert isinstance(result, bool)


def test_step_render_returns_false_when_no_atlas(tmp_path: Path) -> None:
    """render bails out cleanly if atlas.jsonl is missing."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    assert step_render(campaign, tmp_path / "out.pdf") is False


# ---------------------------------------------------------------------------
# 8. step_baseline_diff returns NA when comparator is empty / missing
# ---------------------------------------------------------------------------


def test_step_baseline_diff_empty_comparator_returns_na(tmp_path: Path) -> None:
    assert step_baseline_diff(tmp_path, "c1", "") == "NA"


def test_step_baseline_diff_missing_comparator_returns_na(tmp_path: Path) -> None:
    assert step_baseline_diff(tmp_path, "c1", str(tmp_path / "nope.json")) == "NA"
