"""Atlas-coverage summary block.

Produces the same "Coverage: 40 atlas cells | 38 full sweeps | 1 partial
sweeps | 1 failed cells | 232 plot-ready concurrency points" line that the
GLM-5.1 PDF page-1 header carries, plus the "Note: N evicted before
terminal state" footnote when any evicted cells are present.

Distinction:

- "atlas cells" = cells with status in {full, partial, failed}. These are
  cells the operator explicitly requested in the campaign config.
- "evicted" cells are tracked separately as failure context but not counted
  in the atlas total. They surface in the Note footnote so reviewers
  understand the broader sweep was attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tools.perf_tune_report.schema import (
    STATUS_EVICTED,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)


@dataclass(frozen=True)
class CoverageSummary:
    atlas_cells: int
    full_sweeps: int
    partial_sweeps: int
    failed_cells: int
    plot_ready_points: int
    evicted_cells: int
    non_plot_ready_full_cells: int = 0

    def header_line(self) -> str:
        base = (
            f"Coverage: {self.atlas_cells} atlas cells | "
            f"{self.full_sweeps} full sweeps | "
            f"{self.partial_sweeps} partial sweeps | "
            f"{self.failed_cells} failed cells | "
            f"{self.plot_ready_points} plot-ready concurrency points"
        )
        # Surface "full but unplottable" so a blank scatter is never a
        # mystery: these are STATUS_FULL cells whose rows lack the
        # ttft_avg_ms + request_throughput_avg needed to plot a point.
        if self.non_plot_ready_full_cells:
            base += f" | {self.non_plot_ready_full_cells} full-but-unplottable cells"
        return base

    def note_line(self, evicted_label: str = "cells") -> str | None:
        if self.evicted_cells == 0:
            return None
        return (
            f"Note: {self.evicted_cells} {evicted_label} were evicted before "
            f"terminal state and are represented as failure context, not plotted points."
        )


def summarize(rows: Iterable[AtlasCell]) -> CoverageSummary:
    """Compute the coverage summary from an iterable of atlas rows.

    Counts cells (unique ``cell_id``), not rows. Status is taken from the
    first row encountered for each cell -- the runners + aggregator ensure
    a cell's status is consistent across its rows.
    """
    cell_status: dict[str, str] = {}
    cell_has_plottable: dict[str, bool] = {}
    plot_ready_points = 0
    for row in rows:
        cell_status.setdefault(row.cell_id, row.status)
        cell_has_plottable.setdefault(row.cell_id, False)
        if row.has_metrics:
            plot_ready_points += 1
            cell_has_plottable[row.cell_id] = True

    full = sum(1 for s in cell_status.values() if s == STATUS_FULL)
    partial = sum(1 for s in cell_status.values() if s == STATUS_PARTIAL)
    failed = sum(1 for s in cell_status.values() if s == STATUS_FAILED)
    evicted = sum(1 for s in cell_status.values() if s == STATUS_EVICTED)
    non_plot_ready_full = sum(
        1
        for cid, s in cell_status.items()
        if s == STATUS_FULL and not cell_has_plottable[cid]
    )

    return CoverageSummary(
        atlas_cells=full + partial + failed,
        full_sweeps=full,
        partial_sweeps=partial,
        failed_cells=failed,
        plot_ready_points=plot_ready_points,
        evicted_cells=evicted,
        non_plot_ready_full_cells=non_plot_ready_full,
    )
