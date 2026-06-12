"""Unit tests for first-class spec-decode acceptance-length capture.

Covers the /metrics scrape parser + window math (``spec_metrics``), row
attachment, and the ``aa_bench`` kube-mode integration that brackets each
per-concurrency AIPerf run with pre/post scrapes (formalizing the ad-hoc
shell scrape this feature replaces).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.perf_tune_report.runners.aa_bench import run_cell as run_cell_aa
from tools.perf_tune_report.runners.common import CellConfig
from tools.perf_tune_report.runners.spec_metrics import (
    attach_windows_to_rows,
    build_scrape_command,
    compute_spec_window,
    parse_spec_totals,
)
from tools.perf_tune_report.schema import AtlasCell, STATUS_FULL


def _scrape_text(drafts: float, draft_tokens: float, accepted: float, *, pos0: float = 0.0) -> str:
    """Realistic /metrics excerpt (vLLM spec-decode counter shape): spec
    counters + a ``_created`` timestamp series (must be ignored) + a non-spec
    line the parser must skip."""
    return "\n".join(
        [
            f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="m"}} {drafts}',
            'vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 1.7811135773931787e+09',
            f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {draft_tokens}',
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {accepted}',
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="m",position="0"}} {pos0}',
            'vllm:generation_tokens_total{engine="0",model_name="m"} 30240.0',
        ]
    )


# --- scrape command ---------------------------------------------------------


def test_build_scrape_command_targets_metrics_endpoint():
    cmd = build_scrape_command("http://svc:8000/")
    assert cmd[:2] == ["python", "-c"]
    assert "http://svc:8000/metrics" in cmd[2]
    assert "vllm:spec_decode" in cmd[2]


# --- parser + window math ---------------------------------------------------


def test_parse_spec_totals_sums_engines_and_skips_created():
    text = (
        _scrape_text(100.0, 400.0, 135.0, pos0=80.0)
        + "\n"
        + 'vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 50.0'
    )
    totals = parse_spec_totals(text)
    assert totals["drafts"] == 150.0  # engine 0 + engine 1
    assert totals["draft_tokens"] == 400.0
    assert totals["accepted_tokens"] == 135.0
    assert totals["accepted_per_pos"] == {0: 80.0}


def test_compute_spec_window_al_and_accept_rate():
    # AL = 1 + accepted/drafts, accept_rate = accepted/draft_tokens.
    pre = _scrape_text(100.0, 400.0, 135.0, pos0=60.0)
    post = _scrape_text(200.0, 800.0, 270.0, pos0=140.0)
    win = compute_spec_window(pre, post)
    assert win is not None
    assert win["num_drafts"] == 100.0
    assert win["num_draft_tokens"] == 400.0
    assert win["num_accepted_tokens"] == 135.0
    assert win["al"] == 1.0 + 135.0 / 100.0
    assert win["accept_rate"] == 135.0 / 400.0
    assert win["per_pos_accept_rate"] == {"0": 80.0 / 100.0}


def test_compute_spec_window_no_drafts_is_none():
    # Spec-OFF deploys export no spec counters (zero delta) -> no window, not 0.0.
    pre = _scrape_text(100.0, 400.0, 135.0)
    assert compute_spec_window(pre, pre) is None
    assert compute_spec_window("", "") is None


# --- row attachment ---------------------------------------------------------


def _row(concurrency: int) -> AtlasCell:
    return AtlasCell(
        cell_id="aa-1k",
        model="m",
        hardware="GB300",
        quant="NVFP4",
        tensor_parallel=4,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=4096,
        concurrency=concurrency,
        status=STATUS_FULL,
    )


def test_attach_windows_to_rows():
    windows = {
        1: {
            "num_drafts": 100.0,
            "num_draft_tokens": 400.0,
            "num_accepted_tokens": 135.0,
            "al": 2.35,
            "accept_rate": 0.3375,
            "per_pos_accept_rate": {},
        }
    }
    rows = attach_windows_to_rows([_row(1), _row(10)], windows)
    assert rows[0].acceptance_length == 2.35
    assert rows[0].spec_accept_rate == 0.3375
    assert rows[0].extra["spec_num_drafts"] == 100.0
    assert rows[0].extra["spec_num_accepted_tokens"] == 135.0
    # No window for c=10 (e.g. scrape lost): row passes through untouched.
    assert rows[1].acceptance_length is None
    assert rows[1].spec_accept_rate is None


# --- aa_bench kube-mode integration ------------------------------------------


_KUBE = {"namespace": "ns", "bench_pod": "pod-bench", "kube_context": "ctx"}


def _cell(concurrencies=(1,)) -> CellConfig:
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


class _FakeRunner:
    """Dispatches on command shape: scrapes return successive /metrics texts,
    ``kubectl cp`` materializes the per-concurrency AIPerf report locally."""

    def __init__(self, scrape_texts: list[str], scrape_rc: int = 0):
        self.scrape_texts = list(scrape_texts)
        self.scrape_rc = scrape_rc
        self.commands: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, check=False):
        self.commands.append(list(cmd))
        if cmd[:2] == ["kubectl", "cp"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "profile_export_aiperf.json").write_text(
                json.dumps(
                    {
                        "median_ttft_ms": 100.0,
                        "request_throughput": 1.0,
                        "output_throughput": 200.0,
                    }
                )
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "python" in cmd and "-c" in cmd:  # the /metrics scrape
            out = self.scrape_texts.pop(0) if self.scrape_texts else ""
            return subprocess.CompletedProcess(
                cmd, self.scrape_rc, stdout=out, stderr="scrape boom"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="bench ok", stderr="")


def test_run_cell_aa_kube_captures_spec_window(tmp_path: Path):
    runner = _FakeRunner(
        [
            _scrape_text(100.0, 400.0, 135.0),  # pre-c1
            _scrape_text(200.0, 800.0, 270.0),  # post-c1
        ]
    )
    result = run_cell_aa(
        _cell(),
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://svc:8000",
        kube=_KUBE,
        subprocess_runner=runner,
    )
    assert result.status == STATUS_FULL
    assert result.spec_windows is not None
    assert result.spec_windows[1]["al"] == 2.35
    assert result.spec_windows[1]["accept_rate"] == 0.3375
    spec_dir = result.cell_dir / "spec_metrics"
    # Raw scrapes persisted per concurrency + computed windows + audit log.
    assert (spec_dir / "metrics-pre-c1.prom").is_file()
    assert (spec_dir / "metrics-post-c1.prom").is_file()
    windows = json.loads((spec_dir / "spec_window.json").read_text())
    assert windows["1"]["al"] == 2.35
    assert "ok" in (spec_dir / "scrape.log").read_text()
    # The scrape runs in the bench pod (kubectl exec), like the bench itself.
    scrape_cmds = [c for c in runner.commands if "python" in c and "-c" in c]
    assert all(c[:2] == ["kubectl", "exec"] for c in scrape_cmds)
    # AL / accept-rate land on the normalized rows at the (cell, c) grain.
    rows = json.loads((result.cell_dir / "normalized.json").read_text())
    assert rows[0]["acceptance_length"] == 2.35
    assert rows[0]["spec_accept_rate"] == 0.3375
    assert rows[0]["extra"]["spec_num_drafts"] == 100.0


def test_run_cell_aa_scrape_failure_never_fails_the_cell(tmp_path: Path):
    runner = _FakeRunner([], scrape_rc=1)
    result = run_cell_aa(
        _cell(),
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://svc:8000",
        kube=_KUBE,
        subprocess_runner=runner,
    )
    # Bench rows are intact; the lost scrape is an explicit artifact (the
    # motivating bug: a silently-lost scrape with no trace).
    assert result.status == STATUS_FULL
    assert result.spec_windows == {}
    log = (result.cell_dir / "spec_metrics" / "scrape.log").read_text()
    assert "pre-c1 FAILED" in log
    rows = json.loads((result.cell_dir / "normalized.json").read_text())
    assert rows[0]["acceptance_length"] is None


def test_run_cell_aa_spec_scrape_opt_out(tmp_path: Path):
    runner = _FakeRunner([_scrape_text(1, 1, 1)])
    result = run_cell_aa(
        _cell(),
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://svc:8000",
        kube=_KUBE,
        spec_scrape=False,
        subprocess_runner=runner,
    )
    assert result.spec_windows is None
    assert not (result.cell_dir / "spec_metrics").exists()
    assert not any("python" in c for c in runner.commands)
