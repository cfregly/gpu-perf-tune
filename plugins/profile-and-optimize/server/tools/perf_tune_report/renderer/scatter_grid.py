"""Page-1 scatter grid: 5 rows (one per max_num_batched_tokens) x 2 cols.

Left column: TTFT avg (ms) vs Request throughput avg (req/s).
Right column: Output throughput/user avg (tok/s/user) vs Output throughput/GPU
avg (tok/s/GPU).

Each cell's (mbt, legend_group) gets one curve. Selected concurrency points
(default {1, 8, 32}) are labeled inline at their plotted positions.

Header carries the title, variants line, data-source line, the Coverage
block from ``coverage.summarize``, and the legend-explanation paragraph that
matches the source GLM-5.1 PDF.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from tools.perf_tune_report.coverage import CoverageSummary
from tools.perf_tune_report.renderer.style import (
    LABEL_CONCURRENCIES,
    legend_groups_in_order,
    style_for,
)
from tools.perf_tune_report.schema import AtlasCell

# Layout constants. Page-1 figure size matches A4 portrait reasonably; the
# PdfPages assembler in render_report.py uses the same value so layouts
# stay consistent across pages.
PAGE_WIDTH_IN = 11.0
PAGE_HEIGHT_IN = 14.0
HEADER_HEIGHT_IN = 2.0


def _build_grid_data(
    rows: Sequence[AtlasCell],
) -> dict[tuple[int, tuple], list[AtlasCell]]:
    """Group plot-ready rows by (max_num_batched_tokens, legend_key)."""
    grid: dict[tuple[int, tuple], list[AtlasCell]] = defaultdict(list)
    for row in rows:
        if not row.has_metrics:
            continue
        grid[(row.max_num_batched_tokens, row.legend_key)].append(row)
    for key in grid:
        grid[key].sort(key=lambda r: r.concurrency)
    return grid


def _annotate_concurrencies(ax, rows: Sequence[AtlasCell], x_key: str, y_key: str) -> None:
    """Inline-label rows whose concurrency is in ``LABEL_CONCURRENCIES``."""
    for row in rows:
        if row.concurrency not in LABEL_CONCURRENCIES:
            continue
        x = getattr(row, x_key)
        y = getattr(row, y_key)
        if x is None or y is None:
            continue
        ax.annotate(
            str(row.concurrency),
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color="#222222",
            alpha=0.85,
        )


def _plot_pane(ax, grid_subset, x_key: str, y_key: str, x_label: str, y_label: str) -> None:
    """Plot one scatter pane (one cell of the 5x2 grid)."""
    # Lazy matplotlib import so module load is cheap when only schema is used.
    for (_, _legend_key), pts in sorted(grid_subset.items(), key=lambda kv: kv[0][1]):
        if not pts:
            continue
        style = style_for(pts[0])
        xs = [getattr(r, x_key) for r in pts]
        ys = [getattr(r, y_key) for r in pts]
        ax.plot(
            xs,
            ys,
            color=style.color,
            marker=style.marker,
            markerfacecolor=style.markerfacecolor,
            markeredgecolor=style.color,
            linestyle=style.linestyle,
            linewidth=0.9,
            markersize=5,
            alpha=0.9,
        )
        _annotate_concurrencies(ax, pts, x_key, y_key)
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def render_page(
    fig,
    rows: Sequence[AtlasCell],
    coverage: CoverageSummary,
    *,
    title: str = "glm5p1 benchmark report",
    variants_line: str | None = None,
    data_source_line: str | None = None,
) -> None:
    """Render page 1 into a pre-allocated matplotlib figure."""
    import matplotlib.pyplot as plt  # noqa: F401  (kept for figure plumbing)
    from matplotlib.lines import Line2D

    grid = _build_grid_data(rows)
    mbt_values = sorted({k[0] for k in grid})
    if not mbt_values:
        fig.text(0.5, 0.5, "No plot-ready rows in atlas.", ha="center", va="center")
        return

    nrows = len(mbt_values)
    ncols = 2

    # Reserve top ~16% of the figure for the header (title + variants + data
    # source + coverage + legend explanation). Below it lives the scatter grid.
    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=[1.6] + [1.0] * nrows,
        hspace=0.55,
        wspace=0.28,
        left=0.07,
        right=0.97,
        top=0.96,
        bottom=0.05,
    )

    # Header span across both columns.
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_lines = [title]
    if variants_line:
        header_lines.append(variants_line)
    if data_source_line:
        header_lines.append(data_source_line)
    header_lines.append(coverage.header_line())
    note = coverage.note_line()
    if note:
        header_lines.append(note)
    header_lines.append(
        "TTFT vs Request Throughput   |   Interactivity vs Throughput per GPU"
    )
    header_lines.append(
        "Point labels show selected profiling concurrencies "
        f"({', '.join(str(c) for c in LABEL_CONCURRENCIES)}). "
        "Colors encode hardware+TP deployment. Circle markers are EP; "
        "X markers are TP/no-EP. Dotted lines also indicate TP/no-EP. "
        "Hollow markers indicate MTP."
    )
    header_ax.text(
        0.0,
        1.0,
        "\n".join(header_lines),
        fontsize=8,
        va="top",
        ha="left",
        family="DejaVu Sans",
        linespacing=1.5,
    )

    # Build legend handles from the legend-groups-in-display-order.
    legend_groups = legend_groups_in_order(rows)
    handles = []
    for cell in legend_groups:
        style = style_for(cell)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.color,
                marker=style.marker,
                markerfacecolor=style.markerfacecolor,
                markeredgecolor=style.color,
                linestyle=style.linestyle,
                linewidth=1.1,
                markersize=6,
                label=style.label,
            )
        )
    if handles:
        header_ax.legend(
            handles=handles,
            loc="lower right",
            ncol=2,
            fontsize=6.5,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.4,
        )

    # Per-row scatter panes.
    for r, mbt in enumerate(mbt_values, start=1):
        subset = {k: v for k, v in grid.items() if k[0] == mbt}
        ax_left = fig.add_subplot(gs[r, 0])
        _plot_pane(
            ax_left,
            subset,
            x_key="request_throughput_avg",
            y_key="ttft_avg_ms",
            x_label="Request throughput avg (req/s)",
            y_label="TTFT avg (ms)",
        )
        ax_left.set_title(
            f"max_num_batched_tokens {mbt}", fontsize=7.5, loc="left", pad=2
        )

        ax_right = fig.add_subplot(gs[r, 1])
        _plot_pane(
            ax_right,
            subset,
            x_key="output_tps_per_gpu",
            y_key="output_tps_per_user",
            x_label="Output throughput/GPU avg (tok/s/GPU)",
            y_label="Output throughput/user avg (tok/s/user)",
        )
        ax_right.set_title(
            f"max_num_batched_tokens {mbt}", fontsize=7.5, loc="left", pad=2
        )
