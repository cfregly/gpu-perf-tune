"""TPM-supported-across-hardware page (added v1.35.0).

A stakeholder/pricing-facing table page: one matplotlib table per hardware
type, rows = variant x operating-point (peak / sla), columns = the three
capacity bases (per-GPU, per-replica, per-node) for output-only and total
(input+output) TPM. Built from the same ``compute_tpm_summary`` rollup the
``tpm_summary`` CLI verb emits, so the PDF page and the .md/.csv/.json
artifacts never drift.

Total-TPM cells render ``n/a`` when the backend emitted no total-token line.
When SLA thresholds were supplied but no sweep point met them, an explicit
"no SLA point" row is drawn rather than silently omitting the variant.
"""

from __future__ import annotations

from typing import Sequence

from tools.perf_tune_report.renderer.style import color_for
from tools.perf_tune_report.schema import AtlasCell
from tools.perf_tune_report.tpm_summary import TpmGroup, TpmSummary, compute_tpm_summary

HEADER_BG = "#f3f3f3"
SLA_MISS_BG = "#fde8e8"

_COL_LABELS = [
    "Variant", "Point", "Conc",
    "Out TPM/GPU", "Out TPM/repl", "Out TPM/node",
    "Tot TPM/GPU", "Tot TPM/repl", "Tot TPM/node",
    "$/1M out", "$/1M tot",
]
_N_COLS = len(_COL_LABELS)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _point_cells(label: str, point) -> list[str]:
    return [
        label,
        point.operating_point,
        str(point.concurrency),
        _fmt(point.output_tpm_per_gpu),
        _fmt(point.output_tpm_per_replica),
        _fmt(point.output_tpm_per_node),
        _fmt(point.total_tpm_per_gpu),
        _fmt(point.total_tpm_per_replica),
        _fmt(point.total_tpm_per_node),
        _fmt_usd(point.usd_per_1m_output_tokens),
        _fmt_usd(point.usd_per_1m_total_tokens),
    ]


def _group_rows(g: TpmGroup, sla_computed: bool) -> list[tuple[list[str], bool]]:
    """Return (cell_text, is_sla_miss) tuples for one variant group."""
    rows: list[tuple[list[str], bool]] = []
    if g.peak is not None:
        rows.append((_point_cells(g.legend_label, g.peak), False))
    if g.sla is not None:
        rows.append((_point_cells(g.legend_label, g.sla), False))
    elif sla_computed:
        rows.append(
            (
                [g.legend_label, "sla", "-"] + ["no SLA point"] + ["-"] * (_N_COLS - 4),
                True,
            )
        )
    return rows


def shape_label_problems(groups: Sequence[TpmGroup]) -> list[str]:
    """Per-number exact shape (no smoothing): flag when one shared ISL/OSL caption over
    ``groups`` would smooth heterogeneous per-cell shapes.

    Returns a one-line problem naming the distinct ``(mean_isl, mean_osl)`` shapes when the
    groups carry more than one shape; empty when they share a single shape (safe to caption
    once). This is a render-layer detector -- ``_shape_caption`` consults it to label per-cell
    ("ISL/OSL: per-row (varies)") instead of silently picking the first group's shape. It is
    NOT a publish ``--strict`` gate: heterogeneous shapes across cells are allowed, only
    smoothing them to one caption is the defect (docs/METHODOLOGY.md "Per-number exact
    shape (no smoothing)").
    """
    shapes = sorted(
        {
            (g.mean_isl, g.mean_osl)
            for g in groups
            if g.mean_isl is not None or g.mean_osl is not None
        },
        key=lambda s: (s[0] or 0.0, s[1] or 0.0),
    )
    if len(shapes) <= 1:
        return []
    return [
        f"shape-label smoothing: {len(shapes)} distinct ISL/OSL shapes {shapes} "
        "under one caption -- label per-cell, do not smooth"
    ]


def _shape_caption(groups: Sequence[TpmGroup]) -> list[str]:
    """ISL/OSL + cache caption for one hardware's variant groups.

    Per-number exact shape (no smoothing): emit a single ISL/OSL caption ONLY when the
    groups share one shape (``shape_label_problems`` empty). If the per-request shapes
    diverge across groups, render "ISL/OSL: per-row (varies)" instead of silently picking
    the first group's shape -- collapsing a set of differently-shaped numbers to one label
    is the smoothing defect (docs/METHODOLOGY.md "Per-number exact shape (no smoothing)").
    """
    bits: list[str] = []
    if shape_label_problems(groups):
        bits.append("ISL/OSL: per-row (varies)")
    else:
        shapes = {
            (g.mean_isl, g.mean_osl)
            for g in groups
            if g.mean_isl is not None or g.mean_osl is not None
        }
        if shapes:
            isl, osl = next(iter(shapes))
            if isl is not None:
                bits.append(f"ISL~{isl:.0f}")
            if osl is not None:
                bits.append(f"OSL~{osl:.0f}")
    cmode = groups[0].cache_mode if groups else "unknown"
    bits.append(f"cache: {cmode}")
    return bits


def render_page(
    fig,
    rows: Sequence[AtlasCell],
    summary: TpmSummary | None = None,
    power_by_cell: dict[str, float] | None = None,
) -> bool:
    """Render the TPM-by-hardware page into ``fig``.

    ``summary`` may be precomputed by the caller; when ``None`` it is computed
    from ``rows`` with no SLA thresholds (peak-only). ``power_by_cell`` maps a
    cell_id to its mean per-GPU watts (from dcgm_correlation.json); when present
    the per-hardware caption shows tokens-per-watt at the peak point. Returns
    ``True`` when at least one variant group was drawn, ``False`` when there is
    nothing to show (the caller records the omission)."""
    if summary is None:
        summary = compute_tpm_summary(rows)
    power_by_cell = power_by_cell or {}

    if not summary.groups:
        fig.text(0.5, 0.5, "No throughput-bearing atlas rows for TPM rollup.",
                 ha="center", va="center")
        return False

    hardwares: list[str] = []
    for g in summary.groups:
        if g.hardware not in hardwares:
            hardwares.append(g.hardware)

    n = len(hardwares)
    gs = fig.add_gridspec(
        n + 1, 1,
        height_ratios=[0.5] + [1.0] * n,
        hspace=0.55, left=0.04, right=0.98, top=0.95, bottom=0.04,
    )

    header_ax = fig.add_subplot(gs[0, 0])
    header_ax.axis("off")
    sla_line = (
        f"SLA: TTFT <= {summary.ttft_sla_ms} ms, TPOT/ITL <= {summary.tpot_sla_ms} ms"
        if summary.sla_computed
        else "SLA: not set (peak-capacity only; pass --ttft-sla-ms/--tpot-sla-ms)"
    )
    ctx = f"\nData source / shape: {summary.context_line}" if summary.context_line else ""
    header_ax.text(
        0.0, 1.0,
        "TPM supported across hardware types\n"
        "TPM = tok/s * 60. 'peak' = warm sweep best-case (NOT cold steady-state); "
        "'sla' = highest tok/s/GPU meeting the latency SLA. Total-TPM is n/a when "
        "the backend emits no total-token line.\n"
        f"{sla_line}  |  per-node basis = {summary.gpus_per_node} GPUs{ctx}",
        fontsize=7.5, va="top", ha="left", linespacing=1.5,
    )

    for hi, hw in enumerate(hardwares, start=1):
        ax = fig.add_subplot(gs[hi, 0])
        # Per-hardware shape + cache caption. Per-number exact shape (no smoothing):
        # _shape_caption emits ONE ISL/OSL only when the groups share it, else
        # "per-row (varies)" -- it never smooths heterogeneous shapes to the first group.
        hw_groups = [grp for grp in summary.groups if grp.hardware == hw]
        shape_bits = _shape_caption(hw_groups)
        # tokens-per-watt at the peak point, when DCGM power is present for its cell.
        peak_pt = next((g.peak for g in hw_groups if g.peak is not None), None)
        if peak_pt is not None:
            watts = power_by_cell.get(peak_pt.cell_id)
            if watts and watts > 0:
                shape_bits.append(f"tok/W~{peak_pt.output_tps_per_gpu / watts:.2f}")
        ax.set_title(f"{hw}   ({' | '.join(shape_bits)})", fontsize=9.0, loc="left", pad=4)
        ax.axis("off")

        cell_text: list[list[str]] = []
        label_colors: list[str] = []
        miss_flags: list[bool] = []
        for g in hw_groups:
            for text, is_miss in _group_rows(g, summary.sla_computed):
                cell_text.append(text)
                label_colors.append(color_for(g.hardware, g.tensor_parallel))
                miss_flags.append(is_miss)

        if not cell_text:
            continue

        tbl = ax.table(
            cellText=cell_text,
            colLabels=_COL_LABELS,
            cellLoc="center",
            loc="upper left",
            colWidths=[0.20, 0.05, 0.05] + [0.085] * 6 + [0.085, 0.085],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.0)
        tbl.scale(1.0, 1.3)

        for r_idx, (color, is_miss) in enumerate(zip(label_colors, miss_flags), start=1):
            tbl[(r_idx, 0)].set_text_props(color=color, weight="bold")
            if is_miss:
                for c_idx in range(len(_COL_LABELS)):
                    tbl[(r_idx, c_idx)].set_facecolor(SLA_MISS_BG)

        for c_idx in range(len(_COL_LABELS)):
            tbl[(0, c_idx)].set_facecolor(HEADER_BG)
            tbl[(0, c_idx)].set_text_props(weight="bold")

    return True
