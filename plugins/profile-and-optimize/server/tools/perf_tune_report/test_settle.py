"""Unit tests for the aa-backend settle discipline (shape prewarm + burn-in).

Productized from a 2026-06-11 GB300 settle audit: deploy-first
measurements run 6-37% low without shape-matched prewarm and a discarded
burn-in pass; recorded-trial sigma collapses once the discipline is applied.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.perf_tune_report.runners.aa_bench import run_cell as run_cell_aa
from tools.perf_tune_report.runners.common import CellConfig
from tools.perf_tune_report.runners.settle import build_prewarm_command, prewarm
from tools.perf_tune_report.schema import STATUS_FULL


def _cell(concurrencies=(1, 10)) -> CellConfig:
    return CellConfig(
        cell_id="aa-1k",
        model="m",
        hardware="GB300",
        quant="NVFP4",
        tensor_parallel=4,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=4096,
        concurrencies=tuple(concurrencies),
    )


_KUBE = {"namespace": "ns", "bench_pod": "pod-bench", "kube_context": "ctx"}


# --- prewarm command builder -------------------------------------------------


def test_build_prewarm_command_shape_dims():
    cmd = build_prewarm_command("http://svc:8000/", "moonshotai/Kimi-K2.6", ["aa-1k", "aa-10k"])
    assert cmd[:2] == ["python", "-c"]
    code = cmd[2]
    assert "http://svc:8000/v1/completions" in code
    # generic warmup + the two shape dims (input_tokens, output_tokens)
    assert "(50, 256)" in code
    assert "(1000, 1000)" in code and "(10000, 1500)" in code
    assert "ignore_eos" in code and "SHAPE_PREWARM_OK" in code


def test_build_prewarm_command_unknown_shape():
    with pytest.raises(ValueError):
        build_prewarm_command("http://svc:8000", "m", ["aa-nope"])


def test_prewarm_returns_false_on_failure():
    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
    assert prewarm("http://x:8000", "m", ["aa-1k"], kube_wrap=lambda c: c, subprocess_runner=runner) is False


# --- run_cell integration ----------------------------------------------------


class _FakeRunner:
    """Tracks command order; serves prewarm, bench, and kubectl-cp calls."""

    def __init__(self):
        self.kinds: list[str] = []

    def __call__(self, cmd, capture_output=True, text=True, check=False):
        if cmd[:2] == ["kubectl", "cp"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "profile_export_aiperf.json").write_text(
                json.dumps({"median_ttft_ms": 100.0, "request_throughput": 1.0, "output_throughput": 200.0})
            )
            self.kinds.append("cp")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "python" in cmd and "-c" in cmd:
            code = cmd[cmd.index("-c") + 1]
            if "SHAPE_PREWARM_OK" in code:
                self.kinds.append("prewarm")
                return subprocess.CompletedProcess(cmd, 0, stdout="SHAPE_PREWARM_OK\n", stderr="")
            self.kinds.append("scrape")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        self.kinds.append("bench")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


def test_run_cell_settle_discipline_order_and_discard(tmp_path: Path):
    runner = _FakeRunner()
    sleeps: list[int] = []
    result = run_cell_aa(
        _cell(),
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://svc:8000",
        kube=_KUBE,
        spec_scrape=False,
        prewarm_shapes=["aa-1k"],
        burn_in=True,
        settle_s=30,
        subprocess_runner=runner,
        sleeper=sleeps.append,
    )
    assert result.status == STATUS_FULL
    # Order: prewarm first, then burn-in bench, then 2 recorded benches (+cps).
    assert runner.kinds[0] == "prewarm"
    assert runner.kinds[1] == "bench"  # burn-in
    assert runner.kinds.count("bench") == 3  # burn-in + c=1 + c=10
    # Burn-in is discarded: rows come only from the 2 recorded points.
    assert result.row_count == 2
    # Settles: post-prewarm + post-burn-in + 1 inter-point = 3 x 30s.
    assert sleeps == [30, 30, 30]
    # The settle log records both steps.
    log = (result.cell_dir / "commands" / "settle.log").read_text()
    assert "prewarm" in log and "burn-in" in log and "DISCARDED" in log


def test_run_cell_settle_off_by_default(tmp_path: Path):
    runner = _FakeRunner()
    sleeps: list[int] = []
    result = run_cell_aa(
        _cell(concurrencies=(1,)),
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://svc:8000",
        kube=_KUBE,
        spec_scrape=False,
        subprocess_runner=runner,
        sleeper=sleeps.append,
    )
    assert result.row_count == 1
    assert "prewarm" not in runner.kinds
    assert runner.kinds.count("bench") == 1  # no burn-in
    assert sleeps == []  # no settles when discipline is off
    assert not (result.cell_dir / "commands" / "settle.log").exists()


def test_run_cell_local_burn_in_dir_skipped_by_normalizer(tmp_path: Path):
    # Local mode: burn-in writes to raw/burnin which the normalizer ignores.
    calls: list[list[str]] = []

    def runner(cmd, capture_output=True, text=True, check=False):
        calls.append(list(cmd))
        out = Path(cmd[cmd.index("--output-artifact-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_export_aiperf.json").write_text(
            json.dumps({"median_ttft_ms": 1.0, "request_throughput": 1.0, "output_throughput": 8.0})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    result = run_cell_aa(
        _cell(concurrencies=(1,)),
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://x:8000",
        burn_in=True,
        settle_s=0,
        subprocess_runner=runner,
        sleeper=lambda s: None,
    )
    assert result.row_count == 1  # burnin dir not normalized into a row
    burn_dirs = [c[c.index("--output-artifact-dir") + 1] for c in calls]
    assert any(d.endswith("burnin") for d in burn_dirs)
