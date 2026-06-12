"""Unit tests for the graph_diff verb (v1.21.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_tune_report.graph_diff import (
    GraphDiffResult,
    diff_graph_logs,
    extract_fx_graphs,
    extract_inductor_passes,
)


# ---------------------------------------------------------------------------
# Synthetic log fixtures
# ---------------------------------------------------------------------------

_LOG_SIDE_A = """[2026-05-26 03:01:02] starting vllm engine
[dynamo] tracing forward(...)
[inductor] using pass fuse_attention
=== FX GRAPH ===
def forward(x):
    a = matmul(x, weight)
    b = layernorm(a)
    c = silu(b)
    return c
=== END FX GRAPH ===
[dynamo] graph break: data-dependent control flow
[inductor] using pass canonicalize_views
=== FX GRAPH ===
def forward(x):
    a = matmul(x, weight)
    return a
=== END FX GRAPH ===
[2026-05-26 03:01:08] engine ready
"""

_LOG_SIDE_B = """[2026-05-26 04:01:02] starting vllm engine
[dynamo] tracing forward(...)
[inductor] using pass fuse_attention
[inductor] using pass fuse_allreduce_rms
=== FX GRAPH ===
def forward(x):
    a = fused_matmul_layernorm(x, weight)
    c = silu(a)
    return c
=== END FX GRAPH ===
[inductor] using pass canonicalize_views
=== FX GRAPH ===
def forward(x):
    a = matmul(x, weight)
    return a
=== END FX GRAPH ===
[2026-05-26 04:01:08] engine ready
"""

_LOG_NO_GRAPHS = "[2026-05-26 05:01:02] starting vllm engine\n[engine ready]\n"


def _write_logs(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "side-A.log"
    b = tmp_path / "side-B.log"
    a.write_text(_LOG_SIDE_A)
    b.write_text(_LOG_SIDE_B)
    return a, b


# ---------------------------------------------------------------------------
# 1. extract_fx_graphs — parse one or more FX blocks
# ---------------------------------------------------------------------------


def test_extract_fx_graphs_finds_both_blocks() -> None:
    graphs = extract_fx_graphs(_LOG_SIDE_A)
    assert len(graphs) == 2
    assert "matmul(x, weight)" in graphs[0]


def test_extract_fx_graphs_empty_log_returns_empty_list() -> None:
    assert extract_fx_graphs(_LOG_NO_GRAPHS) == []


# ---------------------------------------------------------------------------
# 2. extract_inductor_passes — dedup + order preserve
# ---------------------------------------------------------------------------


def test_extract_passes_dedups_in_order() -> None:
    passes = extract_inductor_passes(_LOG_SIDE_A)
    assert passes == ["fuse_attention", "canonicalize_views"]


def test_extract_passes_handles_added_pass() -> None:
    passes = extract_inductor_passes(_LOG_SIDE_B)
    assert passes == [
        "fuse_attention",
        "fuse_allreduce_rms",
        "canonicalize_views",
    ]


# ---------------------------------------------------------------------------
# 3. diff_graph_logs — full end-to-end
# ---------------------------------------------------------------------------


def test_diff_writes_all_artifacts(tmp_path: Path) -> None:
    a, b = _write_logs(tmp_path)
    out = tmp_path / "diff-out"
    result = diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=out)
    assert isinstance(result, GraphDiffResult)
    assert result.schema == "inference_graph_diff_v1"
    # Two graphs in each side -> 2 pairs diffed.
    assert len(result.per_graph_diffs) == 2
    # Pass-level diff: side B added fuse_allreduce_rms.
    assert "fuse_allreduce_rms" in result.added_passes
    assert result.removed_passes == []
    # Per-graph artifacts exist on disk.
    assert (out / "side-A-graph0.fx").is_file()
    assert (out / "side-B-graph0.fx").is_file()
    assert (out / "graph0.diff").is_file()
    assert (out / "graph1.diff").is_file()
    assert (out / "graph_diff.json").is_file()
    # graph_diff.json round-trips with the in-memory result.
    payload = json.loads((out / "graph_diff.json").read_text())
    assert payload["schema"] == "inference_graph_diff_v1"
    assert payload["added_passes"] == ["fuse_allreduce_rms"]


def test_diff_summary_mentions_pass_changes(tmp_path: Path) -> None:
    a, b = _write_logs(tmp_path)
    out = tmp_path / "diff-out"
    result = diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=out)
    assert "added: fuse_allreduce_rms" in result.summary


def test_diff_identifies_identical_graph(tmp_path: Path) -> None:
    """graph1 is the same in both sides -> per_graph_diffs[1].identical is True."""
    a, b = _write_logs(tmp_path)
    out = tmp_path / "diff-out"
    result = diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=out)
    # graph1 (second block) is byte-identical
    assert result.per_graph_diffs[1].identical is True
    # graph0 (first block) differs
    assert result.per_graph_diffs[0].identical is False


def test_diff_node_delta_computed(tmp_path: Path) -> None:
    a, b = _write_logs(tmp_path)
    out = tmp_path / "diff-out"
    result = diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=out)
    # side A graph0 has 3 op lines (matmul, layernorm, silu); side B has 2
    # (fused_matmul_layernorm, silu). Plus graph1 unchanged. So:
    # nodes_a_total = 3 + 1 = 4, nodes_b_total = 2 + 1 = 3.
    delta = result.graph_size_delta
    assert delta["nodes_a"] == 4
    assert delta["nodes_b"] == 3
    assert delta["delta_pct"] == pytest.approx(-25.0, abs=0.1)


# ---------------------------------------------------------------------------
# 4. Dry-run + error handling
# ---------------------------------------------------------------------------


def test_dry_run_writes_no_files(tmp_path: Path) -> None:
    a, b = _write_logs(tmp_path)
    out = tmp_path / "diff-out"
    result = diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=out, dry_run=True)
    # Result is still returned with summary metrics.
    assert result.added_passes == ["fuse_allreduce_rms"]
    # But no files were written.
    assert not out.exists() or list(out.iterdir()) == []


def test_diff_missing_side_a_raises(tmp_path: Path) -> None:
    b = tmp_path / "side-B.log"
    b.write_text(_LOG_SIDE_B)
    with pytest.raises(ValueError, match="side-A log does not exist"):
        diff_graph_logs(
            side_a_log=tmp_path / "missing.log",
            side_b_log=b,
            output_dir=tmp_path / "out",
        )


def test_diff_both_logs_empty_raises(tmp_path: Path) -> None:
    a = tmp_path / "empty-A.log"
    b = tmp_path / "empty-B.log"
    a.write_text(_LOG_NO_GRAPHS)
    b.write_text(_LOG_NO_GRAPHS)
    with pytest.raises(ValueError, match="no FX graphs found in either log"):
        diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=tmp_path / "out")


def test_diff_asymmetric_graph_counts(tmp_path: Path) -> None:
    """side B has one fewer graph than side A -> pair-by-index with empty B."""
    a = tmp_path / "side-A.log"
    b = tmp_path / "side-B.log"
    a.write_text(_LOG_SIDE_A)  # 2 graphs
    b.write_text(  # 1 graph
        """[engine start]
=== FX GRAPH ===
def forward(x): return x
=== END FX GRAPH ===
"""
    )
    out = tmp_path / "diff-out"
    result = diff_graph_logs(side_a_log=a, side_b_log=b, output_dir=out)
    assert result.side_a.graph_count == 2
    assert result.side_b.graph_count == 1
    # We diff 2 pairs (the max).
    assert len(result.per_graph_diffs) == 2


# ---------------------------------------------------------------------------
# 5. CLI plumbing smoke test
# ---------------------------------------------------------------------------


def test_cli_graph_diff_smoke(tmp_path: Path) -> None:
    """End-to-end via the CLI command handler."""
    import argparse
    from tools.perf_tune_report.perf_tune_report_cli import cmd_graph_diff

    a, b = _write_logs(tmp_path)
    out = tmp_path / "diff-out"
    ns = argparse.Namespace(
        side_a_log=str(a),
        side_b_log=str(b),
        output_dir=str(out),
        side_a_label=None,
        side_b_label=None,
        notes=None,
        dry_run=False,
        json=True,
    )
    rc = cmd_graph_diff(ns)
    assert rc == 0
    assert (out / "graph_diff.json").is_file()
