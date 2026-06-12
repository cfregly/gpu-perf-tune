"""trtllm backend stub.

Phase 2d of the workspace ``inference-experiment-platform-plan`` documents
the multi-backend extension path. The user's stated backend scope for that
workspace is "vLLM-only for now"; this stub exists so that:

1. The atlas schema's ``backend`` field accepts ``"trtllm"`` (added v1.20.0)
   so existing imports + the downstream perf-lake publish + warehouse joins do not
   need re-versioning when TRT-LLM lands.
2. A future contributor knows exactly which file to fill in + which
   contract to honor. The wiring contract mirrors ``aiperf_bench.run_cell``
   and ``vllm_sweep.run_cell``: build the per-concurrency AtlasCell rows
   under ``<campaign>/cells/<cell-id>/normalized.json``.

Implementation prerequisites tracked outside this stub:

- A reference ``trtllm-build`` + ``trtllm-bench`` runbook (the TRT-LLM
  upstream has both). Vendor it under
  ``server/inference-tools/trtllm-bench/`` mirroring the existing aiperf
  vendoring pattern.
- Decision on the canonical TRT-LLM image (NVIDIA NGC vs operator-built).
- KV-cache + quantization parity probes vs vLLM (NVFP4, FP8) so atlas rows
  are comparable.

When implementing, the entry point signature MUST match the AIPerf runner:

    def run_cell(
        cell_cfg: CellConfig,
        campaign_dir: Path,
        *,
        namespace: str,
        bench_pod: str,
        kube_context: str,
        endpoint_url: str,
        served_model: str,
        dataset_split: str = "2025_07",
        conversation_count: int | None = None,
        dry_run: bool = False,
    ) -> "TrtllmResult": ...

The CLI in ``perf_tune_report_cli.cmd_cell_run`` already accepts ``--backend
trtllm`` if/when this file no longer raises NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.perf_tune_report.runners.common import CellConfig


@dataclass(frozen=True)
class TrtllmResult:
    """Mirrors AiperfResult / VllmSweepResult shape for downstream callers."""

    cell_dir: Path
    status: str
    row_count: int
    dry_run: bool
    commands: tuple[str, ...] = ()


def run_cell(
    cell_cfg: CellConfig,
    campaign_dir: Path,
    *,
    namespace: str,
    bench_pod: str,
    kube_context: str,
    endpoint_url: str,
    served_model: str,
    dataset_split: str = "2025_07",
    conversation_count: int | None = None,
    dry_run: bool = False,
    **_unused: Any,
) -> TrtllmResult:
    """Run one TRT-LLM bench cell (stub).

    Raises ``NotImplementedError`` until a future contributor implements
    the backend. The CLI ``perf_tune_report_cell_run --backend trtllm`` path
    will surface this as a clean FATAL with the tracking-plan reference.
    """
    raise NotImplementedError(
        "TRT-LLM backend is a stub (Phase 2d of inference-experiment-plan). "
        "Atlas schema accepts backend='trtllm'; CLI dispatch is wired; only "
        "the run_cell implementation remains. See "
        "tools/perf_tune_report/runners/trtllm_bench.py docstring for the contract."
    )
