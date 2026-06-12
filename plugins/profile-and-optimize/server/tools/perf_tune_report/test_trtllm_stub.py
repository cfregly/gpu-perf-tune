"""Unit tests for the v1.20.0 TRT-LLM stub backend (Phase 2d)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.perf_tune_report.runners.common import CellConfig
from tools.perf_tune_report.runners.trtllm_bench import TrtllmResult, run_cell
from tools.perf_tune_report.schema import (
    BACKEND_AIPERF,
    BACKEND_TRTLLM,
    BACKEND_VLLM_SWEEP,
    BACKENDS,
    AtlasCell,
)


def test_backend_trtllm_constant():
    assert BACKEND_TRTLLM == "trtllm"


def test_backend_trtllm_in_backends_set():
    """Atlas schema accepts backend='trtllm' even though the runner is a stub."""
    assert BACKEND_TRTLLM in BACKENDS
    assert {BACKEND_VLLM_SWEEP, BACKEND_AIPERF, BACKEND_TRTLLM} <= BACKENDS


def test_atlas_cell_accepts_trtllm_backend():
    """AtlasCell schema validation must accept 'trtllm' as a backend value."""
    cell = AtlasCell(
        cell_id="x", model="m", hardware="H100", quant="FP8",
        tensor_parallel=8, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=1024, concurrency=1, status="full",
        backend=BACKEND_TRTLLM,
    )
    assert cell.backend == "trtllm"


def test_run_cell_raises_not_implemented(tmp_path):
    cfg = CellConfig(
        cell_id="x", model="m", hardware="H100", quant="FP8",
        tensor_parallel=8, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=1024, concurrencies=(1,),
    )
    with pytest.raises(NotImplementedError, match="TRT-LLM backend is a stub"):
        run_cell(
            cfg, tmp_path,
            namespace="ns", bench_pod="pod", kube_context="ctx",
            endpoint_url="http://x", served_model="m",
        )


def test_run_cell_error_message_cites_plan():
    cfg = CellConfig(
        cell_id="x", model="m", hardware="H100", quant="FP8",
        tensor_parallel=8, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=1024, concurrencies=(1,),
    )
    try:
        run_cell(
            cfg, Path("/tmp"),
            namespace="ns", bench_pod="pod", kube_context="ctx",
            endpoint_url="http://x", served_model="m",
        )
    except NotImplementedError as exc:
        msg = str(exc)
        assert "inference-experiment-plan" in msg
        assert "stub" in msg


def test_trtllm_result_dataclass_shape():
    """Mirror the other runners' return shape for downstream callers."""
    result = TrtllmResult(
        cell_dir=Path("/tmp"),
        status="full",
        row_count=5,
        dry_run=True,
        commands=("foo", "bar"),
    )
    assert result.status == "full"
    assert result.row_count == 5
    assert result.dry_run is True
    assert result.commands == ("foo", "bar")
