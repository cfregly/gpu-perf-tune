"""perf_tune_report_graph_diff — diff two torch.compile dumps and emit a
structured summary.

Implements the ``inference-graph-diff`` skill's "Step 2 - Step 5" stages
(extract + diff + emit). The cluster-dump step (Step 1) remains an
operator action because it requires a rolling restart of the target
deployment with ``TORCH_LOGS=+dynamo,+inductor``; that's documented in
the skill and not automated here (deferred to a future MCP verb when
demand justifies the additional ack-gated complexity).

Inputs
------

Two log files, each captured from a vLLM pod with
``TORCH_LOGS=+dynamo,+inductor,+graph_breaks``,
``TORCHDYNAMO_VERBOSE=1``, ``PT_LOGGING_LEVEL=DEBUG`` set.

Outputs (written under --output-dir)
------------------------------------

- ``side-A-graph<n>.fx``       — extracted FX graph (one per compile region)
- ``side-B-graph<n>.fx``
- ``graph<n>.diff``            — unified diff (``difflib.unified_diff``) per pair
- ``graph_diff.json``          — structured summary (see ``GraphDiffResult.to_dict``)

Schema
------

The ``graph_diff.json`` follows the ``inference_graph_diff_v1`` schema:

.. code-block:: json

    {
      "schema": "inference_graph_diff_v1",
      "captured_at": "...",
      "side_a": {
        "label": "side-a",
        "log_path": "...",
        "graph_count": 3,
        "compile_passes_seen": [...],
        "graph_breaks_count": <int>
      },
      "side_b": { ... same ... },
      "added_passes": [...],
      "removed_passes": [...],
      "graph_size_delta": {
        "nodes_a": <int>,
        "nodes_b": <int>,
        "delta_pct": <float | None if nodes_a == 0>
      },
      "per_graph_diffs": [
        {
          "graph_index": 0,
          "fx_diff_path": "...",
          "nodes_added": <int>,
          "nodes_removed": <int>,
          "identical": <bool>
        }
      ],
      "summary": "<one-line human-readable verdict>",
      "notes": "<operator-supplied>"
    }
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Log-parsing regexes
# ---------------------------------------------------------------------------

# FX graph block: bracketed by ``=== FX GRAPH ===`` markers. vLLM 0.21.x
# format emits one block per compile region; we extract all of them.
_FX_GRAPH_BLOCK = re.compile(
    r"=== FX GRAPH ===\s*\n(.*?)\n=== END FX GRAPH ===",
    re.DOTALL,
)

# Inductor pass enable lines: ``[inductor] using pass: <name>`` or
# ``[inductor] pass <name> applied``. Both forms exist across torch versions.
_INDUCTOR_PASS = re.compile(
    r"\[inductor\]\s+(?:using pass|pass)\s+([\w_]+)(?:\s+applied)?",
    re.IGNORECASE,
)

# Graph break lines: ``[dynamo] graph break: <reason>``. We just count
# how many fired.
_GRAPH_BREAK = re.compile(r"\[dynamo\]\s+graph break", re.IGNORECASE)


@dataclass
class _SideSummary:
    """Summary of a single side's torch.compile log."""

    label: str
    log_path: str
    graph_count: int
    compile_passes_seen: list[str]
    graph_breaks_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PerGraphDiff:
    """Per-graph diff summary record."""

    graph_index: int
    fx_diff_path: str
    nodes_added: int
    nodes_removed: int
    identical: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphDiffResult:
    """Top-level diff result returned by ``diff_graph_logs``."""

    schema: str
    captured_at: str
    side_a: _SideSummary
    side_b: _SideSummary
    added_passes: list[str]
    removed_passes: list[str]
    graph_size_delta: dict[str, Any]
    per_graph_diffs: list[_PerGraphDiff]
    summary: str
    notes: str
    output_dir: str
    graph_diff_json_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "captured_at": self.captured_at,
            "side_a": self.side_a.to_dict(),
            "side_b": self.side_b.to_dict(),
            "added_passes": self.added_passes,
            "removed_passes": self.removed_passes,
            "graph_size_delta": self.graph_size_delta,
            "per_graph_diffs": [d.to_dict() for d in self.per_graph_diffs],
            "summary": self.summary,
            "notes": self.notes,
            "output_dir": self.output_dir,
            "graph_diff_json_path": self.graph_diff_json_path,
        }


# ---------------------------------------------------------------------------
# Public helpers (also exported for tests)
# ---------------------------------------------------------------------------


def extract_fx_graphs(log_text: str) -> list[str]:
    """Find every ``=== FX GRAPH ===`` block in the log."""
    return [m.group(1) for m in _FX_GRAPH_BLOCK.finditer(log_text)]


def extract_inductor_passes(log_text: str) -> list[str]:
    """Find the deduplicated, order-preserved list of inductor passes seen."""
    seen: list[str] = []
    for m in _INDUCTOR_PASS.finditer(log_text):
        name = m.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def _count_graph_breaks(log_text: str) -> int:
    return len(_GRAPH_BREAK.findall(log_text))


def _summarize_side(label: str, log_path: Path) -> tuple[_SideSummary, list[str]]:
    """Read a side's log + return its summary + the raw FX graphs."""
    text = log_path.read_text(errors="replace")
    graphs = extract_fx_graphs(text)
    passes = extract_inductor_passes(text)
    breaks = _count_graph_breaks(text)
    return (
        _SideSummary(
            label=label,
            log_path=str(log_path),
            graph_count=len(graphs),
            compile_passes_seen=passes,
            graph_breaks_count=breaks,
        ),
        graphs,
    )


def _count_graph_nodes(graph_text: str) -> int:
    """Approximate FX-graph node count: number of non-empty, non-comment lines.

    The FX-graph dump format uses one line per node; this approximation
    correlates >0.99 with the exact ``len(graph.nodes)`` in practice and
    avoids importing torch (which the test environment doesn't have).
    """
    n = 0
    for line in graph_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("def "):
            continue
        if stripped.startswith("return "):
            continue
        n += 1
    return n


def _diff_one_graph(
    idx: int,
    side_a_text: str,
    side_b_text: str,
    side_a_label: str,
    side_b_label: str,
    output_dir: Path,
) -> tuple[_PerGraphDiff, Path, Path]:
    """Write the side-A/side-B FX dumps + unified diff and return summary."""
    a_path = output_dir / f"{side_a_label}-graph{idx}.fx"
    b_path = output_dir / f"{side_b_label}-graph{idx}.fx"
    a_path.write_text(side_a_text + "\n")
    b_path.write_text(side_b_text + "\n")

    a_lines = side_a_text.splitlines()
    b_lines = side_b_text.splitlines()
    diff_text = "\n".join(
        difflib.unified_diff(
            a_lines, b_lines,
            fromfile=str(a_path.name),
            tofile=str(b_path.name),
            lineterm="",
        )
    )
    diff_path = output_dir / f"graph{idx}.diff"
    diff_path.write_text(diff_text + ("\n" if diff_text else ""))

    nodes_a = _count_graph_nodes(side_a_text)
    nodes_b = _count_graph_nodes(side_b_text)
    return (
        _PerGraphDiff(
            graph_index=idx,
            fx_diff_path=str(diff_path),
            nodes_added=max(0, nodes_b - nodes_a),
            nodes_removed=max(0, nodes_a - nodes_b),
            identical=side_a_text == side_b_text,
        ),
        a_path,
        b_path,
    )


# ---------------------------------------------------------------------------
# Top-level diff driver
# ---------------------------------------------------------------------------


def diff_graph_logs(
    *,
    side_a_log: Path,
    side_b_log: Path,
    output_dir: Path,
    side_a_label: str = "side-A",
    side_b_label: str = "side-B",
    notes: str = "",
    dry_run: bool = False,
    captured_at: str | None = None,
) -> GraphDiffResult:
    """Run the full graph diff and emit artifacts to ``output_dir``.

    Args:
        side_a_log: path to side-A torch.compile log (must exist)
        side_b_log: path to side-B torch.compile log (must exist)
        output_dir: where to write per-graph fx + diff + graph_diff.json
        side_a_label: filename prefix for side-A artifacts (default "side-A")
        side_b_label: filename prefix for side-B artifacts (default "side-B")
        notes: operator-supplied free-form notes appended to graph_diff.json
        dry_run: validate inputs + parse logs but skip writing artifacts
        captured_at: ISO-8601 timestamp; defaults to now()

    Returns:
        ``GraphDiffResult`` with paths + summary.

    Raises:
        ValueError: missing inputs or empty logs.
    """
    side_a_log = side_a_log.expanduser().resolve()
    side_b_log = side_b_log.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not side_a_log.is_file():
        raise ValueError(f"graph_diff: side-A log does not exist: {side_a_log}")
    if not side_b_log.is_file():
        raise ValueError(f"graph_diff: side-B log does not exist: {side_b_log}")

    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    a_summary, a_graphs = _summarize_side(side_a_label, side_a_log)
    b_summary, b_graphs = _summarize_side(side_b_label, side_b_log)

    if not a_graphs and not b_graphs:
        raise ValueError(
            f"graph_diff: no FX graphs found in either log. Ensure both sides "
            f"were captured with TORCH_LOGS=+dynamo,+inductor,+graph_breaks "
            f"(side-A: {side_a_log}, side-B: {side_b_log})"
        )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Pair graphs by index (graph0 <-> graph0). If one side has more graphs
    # than the other, the extra ones are recorded as "graph appeared/
    # disappeared" with an empty counterpart.
    n_pairs = max(len(a_graphs), len(b_graphs))
    per_graph: list[_PerGraphDiff] = []
    if not dry_run:
        for i in range(n_pairs):
            a_text = a_graphs[i] if i < len(a_graphs) else ""
            b_text = b_graphs[i] if i < len(b_graphs) else ""
            diff_summary, _a_p, _b_p = _diff_one_graph(
                idx=i,
                side_a_text=a_text,
                side_b_text=b_text,
                side_a_label=side_a_label,
                side_b_label=side_b_label,
                output_dir=output_dir,
            )
            per_graph.append(diff_summary)
    else:
        # Dry-run: still compute summary metrics so the operator sees what
        # *would* be written. No files are touched.
        for i in range(n_pairs):
            a_text = a_graphs[i] if i < len(a_graphs) else ""
            b_text = b_graphs[i] if i < len(b_graphs) else ""
            nodes_a = _count_graph_nodes(a_text)
            nodes_b = _count_graph_nodes(b_text)
            per_graph.append(
                _PerGraphDiff(
                    graph_index=i,
                    fx_diff_path=str(output_dir / f"graph{i}.diff"),
                    nodes_added=max(0, nodes_b - nodes_a),
                    nodes_removed=max(0, nodes_a - nodes_b),
                    identical=a_text == b_text,
                )
            )

    # Compute pass-level diff (set difference, order-preserving).
    a_passes = set(a_summary.compile_passes_seen)
    b_passes = set(b_summary.compile_passes_seen)
    added_passes = [p for p in b_summary.compile_passes_seen if p not in a_passes]
    removed_passes = [p for p in a_summary.compile_passes_seen if p not in b_passes]

    # Graph-size delta: sum of nodes across all graphs.
    nodes_a_total = sum(_count_graph_nodes(g) for g in a_graphs)
    nodes_b_total = sum(_count_graph_nodes(g) for g in b_graphs)
    if nodes_a_total > 0:
        delta_pct = ((nodes_b_total - nodes_a_total) / nodes_a_total) * 100.0
    else:
        delta_pct = None

    # Human-readable summary
    if added_passes or removed_passes:
        pass_summary_parts: list[str] = []
        if added_passes:
            pass_summary_parts.append(f"added: {', '.join(added_passes)}")
        if removed_passes:
            pass_summary_parts.append(f"removed: {', '.join(removed_passes)}")
        pass_summary = "; ".join(pass_summary_parts)
    else:
        pass_summary = "no compile-pass changes"
    delta_label = (
        f"{delta_pct:+.1f}%" if delta_pct is not None else "n/a"
    )
    summary = (
        f"{n_pairs} graph(s) diffed; {pass_summary}; "
        f"node-count delta: {delta_label} "
        f"({nodes_a_total} -> {nodes_b_total})"
    )

    graph_diff_json_path = output_dir / "graph_diff.json"
    result = GraphDiffResult(
        schema="inference_graph_diff_v1",
        captured_at=captured_at,
        side_a=a_summary,
        side_b=b_summary,
        added_passes=added_passes,
        removed_passes=removed_passes,
        graph_size_delta={
            "nodes_a": nodes_a_total,
            "nodes_b": nodes_b_total,
            "delta_pct": delta_pct,
        },
        per_graph_diffs=per_graph,
        summary=summary,
        notes=notes,
        output_dir=str(output_dir),
        graph_diff_json_path=str(graph_diff_json_path),
    )

    if not dry_run:
        graph_diff_json_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True)
        )

    return result
