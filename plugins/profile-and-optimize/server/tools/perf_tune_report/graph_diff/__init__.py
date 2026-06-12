"""perf_tune_report_graph_diff verb (v1.21.0).

Diff two ``torch.compile`` dynamo+inductor log dumps captured from
side-A vs side-B vLLM configs and emit a structured ``graph_diff.json``
+ per-graph unified diffs. Backs the ``inference-graph-diff`` skill.

Read-only on the cluster: the caller pre-collects the side-A.log and
side-B.log via the documented ``TORCH_LOGS=+dynamo,+inductor``
incantation; this verb only reads them and writes diff artifacts.
"""

from tools.perf_tune_report.graph_diff.graph_diff import (
    GraphDiffResult,
    diff_graph_logs,
    extract_fx_graphs,
    extract_inductor_passes,
)

__all__ = [
    "GraphDiffResult",
    "diff_graph_logs",
    "extract_fx_graphs",
    "extract_inductor_passes",
]
