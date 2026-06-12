"""Page-2 heatmap-style tables.

3 rows (one per profiling concurrency C in {8, 16, 32}) x 2 columns
(metric in {output tok/s/GPU, TTFT avg (ms)}). Rows of each table = the
8 legend-group variants in display order; columns = max_num_batched_tokens
values {1024, 2048, 4096, 8192, 16384}.

Cell rendering follows the PDF convention:

- numeric value -> formatted to 1 decimal place
- ``failed`` / ``partial`` string -> rendered as gray-cell text
- missing measurement -> gray cell with blank text
"""

from __future__ import annotations

from typing import Sequence

from tools.perf_tune_report.renderer.style import legend_groups_in_order, style_for
from tools.perf_tune_report.schema import (
    STATUS_FAILED,
    STATUS_PARTIAL,
    AtlasCell,
)

# Operator-configurable; the PDF uses these three concurrencies.
TABLE_CONCURRENCIES: tuple[int, ...] = (8, 16, 32)

METRICS = (
    ("output_tps_per_gpu", "output tok/s/GPU", "{:.1f}"),
    ("ttft_avg_ms", "TTFT avg (ms)", "{:.1f}"),
)

# Reasoning-model metric columns, appended ONLY when a campaign carries them (None for
# non-reasoning models / older JSONL -> column omitted, so existing reports are unchanged).
# TTFT (first token of any type, incl. reasoning) under-reports answer latency on a reasoning
# model; TTFO (first non-reasoning token) is the answer latency. See the inference-aa-workload
# skill "Reasoning models" section.
REASONING_METRICS = (
    ("ttfo_avg_ms", "TTFO avg (ms)", "{:.1f}"),
    ("ttfo_coverage", "TTFO cov", "{:.0%}"),
    ("reasoning_token_count", "reasoning toks", "{:.0f}"),
)


def _metrics_for(rows: Sequence[AtlasCell]) -> tuple[tuple[str, str, str], ...]:
    """Base metrics + the reasoning columns iff any row carries them."""
    metrics = list(METRICS)
    for field, label, fmt in REASONING_METRICS:
        if any(getattr(r, field, None) is not None for r in rows):
            metrics.append((field, label, fmt))
    return tuple(metrics)

GRAY_CELL_BG = "#dddddd"
HEADER_BG = "#f3f3f3"


def _index_atlas(rows: Sequence[AtlasCell]):
    """Index atlas rows for fast (legend_key, mbt, concurrency) lookup, and
    track per-cell status so the table can render ``failed`` / ``partial``
    sentinels at the right spots."""
    by_lkc: dict[tuple, AtlasCell] = {}
    cell_status: dict[tuple, str] = {}  # keyed by (legend_key, mbt)
    for row in rows:
        cell_status.setdefault((row.legend_key, row.max_num_batched_tokens), row.status)
        if row.has_metrics:
            by_lkc[(row.legend_key, row.max_num_batched_tokens, row.concurrency)] = row
    return by_lkc, cell_status


def _cell_text_and_color(
    by_lkc: dict, cell_status: dict, legend_key: tuple, mbt: int, concurrency: int,
    metric: str, fmt: str,
) -> tuple[str, str]:
    """Return (display_text, background_color) for one heatmap cell."""
    row = by_lkc.get((legend_key, mbt, concurrency))
    status = cell_status.get((legend_key, mbt), "unknown")
    if row is not None and getattr(row, metric) is not None:
        return fmt.format(getattr(row, metric)), "white"
    # No measurement; map to the per-cell status.
    if status == STATUS_FAILED:
        return "failed", GRAY_CELL_BG
    if status == STATUS_PARTIAL:
        return "partial", GRAY_CELL_BG
    # Unknown / evicted at this concurrency: blank gray.
    return "", GRAY_CELL_BG


def render_page(fig, rows: Sequence[AtlasCell]) -> None:
    """Render page 2 into the figure: 3x2 grid of tables."""
    import matplotlib.pyplot as plt  # noqa: F401  (kept for figure plumbing)

    legend_groups = legend_groups_in_order(rows)
    if not legend_groups:
        fig.text(0.5, 0.5, "No atlas rows to render.", ha="center", va="center")
        return

    mbt_values = sorted({r.max_num_batched_tokens for r in rows})
    by_lkc, cell_status = _index_atlas(rows)

    metrics = _metrics_for(rows)
    nrows = len(TABLE_CONCURRENCIES)
    ncols = len(metrics)

    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=[0.35] + [1.0] * nrows,
        hspace=0.45,
        wspace=0.18,
        left=0.05,
        right=0.97,
        top=0.96,
        bottom=0.03,
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(
        0.0,
        1.0,
        "Per-concurrency atlas heatmaps\n"
        "Blank gray cells indicate no metric for that concurrency. "
        "Labels such as 'failed' or 'partial' come from atlas state plus "
        "local profile coverage.",
        fontsize=8,
        va="top",
        ha="left",
        linespacing=1.5,
    )

    for ri, c in enumerate(TABLE_CONCURRENCIES, start=1):
        for ci, (metric, metric_label, fmt) in enumerate(metrics):
            ax = fig.add_subplot(gs[ri, ci])
            ax.set_title(f"C={c}: {metric_label}", fontsize=8.5, loc="left", pad=4)
            ax.axis("off")

            # Build the table. Columns = ["variant"] + mbt_values.
            col_labels = [""] + [str(m) for m in mbt_values]
            cell_text: list[list[str]] = []
            cell_colors: list[list[str]] = []
            row_colors: list[str] = []

            for variant_cell in legend_groups:
                style = style_for(variant_cell)
                row_text = [variant_cell.legend_label]
                row_col = ["white"]
                for mbt in mbt_values:
                    text, bg = _cell_text_and_color(
                        by_lkc, cell_status, variant_cell.legend_key, mbt, c,
                        metric, fmt,
                    )
                    row_text.append(text)
                    row_col.append(bg)
                cell_text.append(row_text)
                cell_colors.append(row_col)
                row_colors.append(style.color)

            tbl = ax.table(
                cellText=cell_text,
                cellColours=cell_colors,
                colLabels=col_labels,
                cellLoc="center",
                loc="upper left",
                colWidths=[0.32] + [0.135] * len(mbt_values),
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(6.5)
            tbl.scale(1.0, 1.3)

            # Color the leftmost (label) cell of each row with the variant's
            # legend color band for visual cross-reference to page 1.
            for r_idx, row_color in enumerate(row_colors, start=1):
                tbl[(r_idx, 0)].set_text_props(color=row_color, weight="bold")

            # Tint the header row.
            for c_idx in range(len(col_labels)):
                tbl[(0, c_idx)].set_facecolor(HEADER_BG)
                tbl[(0, c_idx)].set_text_props(weight="bold")
