"""Page-5 Speed-of-Light Roofline Scatter (ncu byte+FLOP measured).

The byte-grounded counterpart to page 4 (``sol_roofline.py``). Where
page 4 plots zymtrace sample-share time-share as a proxy, this page
plots the proper Williams/Patterson/Asanovic Roofline: arithmetic
intensity (FLOPS/byte) on x-axis, achieved TFLOPS on y-axis, with the
hardware's bandwidth-bound diagonal and compute-bound horizontal ceiling
drawn.

Inputs come from ``ncu_kernels.json`` per-cell payloads emitted by
``importers/ncu_kernels.py`` (one row per ncu-captured kernel: name,
arithmetic_intensity, achieved_tflops, achieved_dram_pct_peak,
achieved_sm_pct_peak, category).

Layout (top to bottom):

1. Header block: campaign title + hardware identifier + ridge-point info.
2. The main log-log scatter:
   - Diagonal bandwidth-bound line: y = AI * hbm3e_tbps (in TFLOPS units)
   - Horizontal compute-bound line: y = nvfp4_dense_pflops (in TFLOPS)
   - One point per ncu-measured kernel, marker by category
   - %SoL annotation per point (max of dram_pct_peak and sm_pct_peak)
3. Caveat footer: ncu measurement caveats (launch count, kernel-name
   substring matching, time vs achieved TFLOPS conversion).

The page is conditional. Caller decides skip-when-empty in
``render_report.discover_ncu_payloads`` (same pattern as page 3 +
page 4). The page raises ``ValueError`` when called with an empty
``cell_ncu`` dict, mirroring kernel_breakdown.render_page's contract.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Page5Status:
    """Outcome of one ``render_page`` call, so the caller can record whether
    the page is a full measurement or a partial (AI-unmeasured) placeholder.

    ``partial`` is True when the page rendered but carries NO real per-kernel
    arithmetic-intensity point -- i.e. only %SoL-only markers (ncu --set=basic)
    or the empty-state message. ``reason`` is a ``render_status.PARTIAL_REASONS``
    key ("" when not partial). A DCGM workload-level fallback is NOT partial
    (it carries genuine byte/FLOP-grounded points).
    """

    partial: bool
    reason: str = ""


def _dcgm_fallback_available(
    cell_dcgm: "OrderedDict[str, dict[str, Any]] | None",
) -> bool:
    """True iff ``_plot_dcgm_fallback`` would plot at least one point.

    Pure predicate (no drawing) so the dynamic title can be decided before the
    scatter loop runs. Mirrors the per-entry conditions in
    ``_plot_dcgm_fallback``.
    """
    if not cell_dcgm:
        return False
    for payload in cell_dcgm.values():
        if float(payload.get("duration_s") or 0.0) <= 0:
            continue
        for entry in payload.get("per_category_attribution") or []:
            bytes_total = entry.get("attributed_bytes_total")
            flops_total = entry.get("attributed_flops_total")
            time_share_pct = entry.get("time_share_pct")
            if (
                bytes_total is not None
                and flops_total is not None
                and time_share_pct is not None
                and bytes_total > 0
                and time_share_pct > 0
            ):
                return True
    return False


# Category -> matplotlib marker mapping. Mirrors the per-category bound
# split: bandwidth-bound categories use circles, compute-bound use
# triangles, mixed/other use squares.
_CATEGORY_MARKERS: dict[str, str] = {
    "NCCL": "o",
    "FMHA": "o",
    "Elementwise": "o",
    "MoE": "^",
    "BMM-NVFP4": "^",
    "Triton-fused": "s",
    "cuBLAS": "s",
    "Other": "D",
}


def _peak_compute_tflops(hw_data: dict[str, Any]) -> tuple[float, str]:
    """Pick the highest-precision dense compute peak available for the scatter ceiling.

    NVFP4 dense is the default ceiling for the renderer since that's
    what the NVFP4 deploys actually use. Falls back to FP8 then BF16.
    Returns ``(value_in_tflops, source_key)``.
    """
    for key in ("nvfp4_dense_pflops", "fp8_dense_pflops", "bf16_dense_pflops"):
        entry = hw_data.get(key)
        if isinstance(entry, dict) and entry.get("value") is not None:
            return float(entry["value"]) * 1000.0, key  # PFLOPS -> TFLOPS
    return 1000.0, "fallback"  # 1 PFLOPS arbitrary fallback


def _category_compute_ceiling_tflops(
    category: str,
    ceilings: dict[str, Any],
    hardware_key: str,
) -> float | None:
    """Per-category compute ceiling (TFLOPS) from ``category_ceiling_map``.

    Used to place ``%SoL-only`` kernels (AI unmeasured, e.g. ncu --set=basic)
    at an honest y = achieved-SM% x the kernel family's natural compute
    ceiling. Returns None for bandwidth-bound categories (no compute ceiling
    applies) or when the mapping / hardware peak is absent. No magic numbers:
    the ceiling is read from sol-ceilings.yaml by key path.
    """
    cmap = ceilings.get("category_ceiling_map") or {}
    entry = cmap.get(category)
    if not isinstance(entry, dict) or entry.get("bound") != "compute":
        return None
    metric_key = entry.get("metric")
    hw_data = ceilings.get(hardware_key) or {}
    peak = hw_data.get(metric_key)
    if isinstance(peak, dict) and peak.get("value") is not None:
        return float(peak["value"]) * 1000.0  # PFLOPS -> TFLOPS
    return None


def _peak_bandwidth_tbps(hw_data: dict[str, Any]) -> tuple[float, str]:
    """Pick the HBM bandwidth peak. Returns (value_in_tbps, source_key)."""
    for key in ("hbm3e_tbps", "hbm3_tbps"):
        entry = hw_data.get(key)
        if isinstance(entry, dict) and entry.get("value") is not None:
            return float(entry["value"]), key
    return 8.0, "fallback"


def _plot_dcgm_fallback(
    ax,
    cell_dcgm: "OrderedDict[str, dict[str, Any]]",
    *,
    ridge_ai: float,
    peak_tflops: float,
    peak_tbps: float,
) -> bool:
    """Plot one point per dcgm per_category_attribution entry.

    v1.23.2 fallback for when ncu_kernels.json has all-null AI/tflops
    (e.g. capture used ``--set=basic``). Each per-category attribution
    row carries ``attributed_bytes_total`` + ``attributed_flops_total``
    so we can compute arithmetic intensity directly:

        AI = attributed_flops_total / attributed_bytes_total

    For achieved TFLOPS we need a per-second rate. The DCGM
    correlation already records ``duration_s`` at the result level and
    the per_category time_share_pct telling us what fraction of the
    sweep this category occupied. So:

        achieved_tflops = (attributed_flops_total / 1e12)
                          / (duration_s * time_share_pct/100)

    Returns True iff at least one point was plotted.
    """
    plotted = 0
    for ci, (cell_id, payload) in enumerate(cell_dcgm.items()):
        per_cat = payload.get("per_category_attribution") or []
        duration_s = float(payload.get("duration_s") or 0.0)
        if duration_s <= 0:
            continue
        for entry in per_cat:
            cat = entry.get("category", "Other")
            bytes_total = entry.get("attributed_bytes_total")
            flops_total = entry.get("attributed_flops_total")
            time_share_pct = entry.get("time_share_pct")
            if (
                bytes_total is None
                or flops_total is None
                or time_share_pct is None
                or bytes_total <= 0
                or time_share_pct <= 0
            ):
                continue
            cat_window_s = duration_s * (time_share_pct / 100.0)
            if cat_window_s <= 0:
                continue
            ai = float(flops_total) / float(bytes_total)
            achieved_tflops = (float(flops_total) / 1e12) / cat_window_s
            if ai <= 0 or achieved_tflops <= 0:
                continue
            marker = _CATEGORY_MARKERS.get(cat, "D")
            ax.scatter(
                ai,
                achieved_tflops,
                s=80,
                marker=marker,
                edgecolors="#3a55a8",
                facecolors="none",
                linewidths=1.2,
                linestyle="dotted",
                alpha=0.85,
                label=f"{cat} (DCGM)" if ci == 0 else None,
            )
            sol_bw = entry.get("sol_pct_bw")
            sol_compute = entry.get("sol_pct_compute")
            sol_pct = sol_bw if sol_bw is not None else sol_compute
            label_parts = [cat]
            if sol_pct is not None:
                label_parts.append(f"{sol_pct:.1f}%SoL")
            ax.annotate(
                "  ".join(label_parts),
                xy=(ai, achieved_tflops),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#3a55a8",
                style="italic",
            )
            plotted += 1
    if plotted > 0:
        # Add a small caption noting the fallback.
        ax.text(
            0.02,
            0.97,
            "fallback: DCGM workload-level points (ncu --set=basic missing FLOPS+bytes)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#3a55a8",
            style="italic",
        )
    return plotted > 0


def render_page(
    fig,
    cell_ncu: "OrderedDict[str, dict[str, Any]]",
    ceilings: dict[str, Any],
    hardware_key: str,
    cell_dcgm: "OrderedDict[str, dict[str, Any]] | None" = None,
) -> Page5Status:
    """Draw the SoL roofline scatter page onto a matplotlib Figure.

    Args:
        fig: matplotlib Figure (created by the caller).
        cell_ncu: ordered ``{cell_id: ncu_kernels.json payload}``. MUST
            be non-empty (caller handles skip-when-empty).
        ceilings: parsed sol-ceilings.yaml dict.
        hardware_key: which ceilings hardware row to plot rooflines for
            (e.g. ``"b200_sm100"``). MUST be present in ``ceilings``.
        cell_dcgm: optional ``{cell_id: dcgm_correlation.json payload}``.
            When provided AND no ncu kernels are plottable (all
            ``arithmetic_intensity_flops_per_byte`` / ``achieved_tflops``
            are null), the renderer falls back to plotting one point per
            ``per_category_attribution`` entry derived from DCGM
            workload-level totals. This makes page 5 useful even when
            ncu was captured with ``--set=basic`` (no FLOPS / DRAM-bytes
            counters). Added in v1.23.2.
    """
    if not cell_ncu:
        raise ValueError("sol_roofline_scatter.render_page: cell_ncu is empty")
    if hardware_key not in ceilings:
        raise ValueError(
            f"sol_roofline_scatter.render_page: hardware_key {hardware_key!r} not in ceilings"
        )

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    hw_data = ceilings[hardware_key]
    hw_name = hw_data.get("hw_name", hardware_key)
    peak_tflops, compute_src = _peak_compute_tflops(hw_data)
    peak_tbps, bw_src = _peak_bandwidth_tbps(hw_data)

    # Ridge-point arithmetic intensity (where bandwidth and compute ceilings cross).
    # peak_tflops [TFLOPS]; peak_tbps [TB/s] = 1e12 bytes/s; ridge AI = peak_tflops*1e12 / (peak_tbps*1e12) = peak_tflops/peak_tbps
    # No, let me redo: y = ai * bw -> ai_ridge = peak_compute / peak_bw.
    # Units: peak_tflops [TFLOPS = 1e12 FLOPS/s], peak_tbps*1e12 [bytes/s].
    # AI [FLOPS/byte] = (peak_tflops * 1e12) / (peak_tbps * 1e12) = peak_tflops / peak_tbps
    ridge_ai = peak_tflops / peak_tbps

    # Decide the page mode up-front (drives the honest dynamic title + the
    # partial-status return). "ncu" = real per-kernel AI points;
    # "solonly" = only %SoL markers (AI unmeasured, --set=basic);
    # "dcgm" = workload-level DCGM byte/FLOP fallback (measured, not partial);
    # "empty" = nothing plottable.
    _has_real_ai = any(
        (k.get("arithmetic_intensity_flops_per_byte") or 0) > 0
        and (k.get("achieved_tflops") or 0) > 0
        for payload in cell_ncu.values()
        for k in payload.get("kernels", [])
    )
    _has_solonly = (not _has_real_ai) and any(
        (k.get("achieved_sm_pct_peak") or 0) > 0
        for payload in cell_ncu.values()
        for k in payload.get("kernels", [])
    )
    if _has_real_ai:
        page_mode = "ncu"
    elif _has_solonly:
        page_mode = "solonly"
    elif _dcgm_fallback_available(cell_dcgm):
        page_mode = "dcgm"
    else:
        page_mode = "empty"

    _TITLES = {
        "ncu": ("Speed-of-Light Roofline Scatter (ncu byte+FLOP measured)", "black"),
        "solonly": (
            "Speed-of-Light Roofline Scatter -- PARTIAL: arithmetic intensity UNMEASURED",
            "#aa3333",
        ),
        "dcgm": (
            "Speed-of-Light Roofline Scatter (DCGM workload-level fallback)",
            "#3a55a8",
        ),
        "empty": (
            "Speed-of-Light Roofline Scatter -- NO roofline-ready data",
            "#aa3333",
        ),
    }
    title_text, title_color = _TITLES[page_mode]

    gs = gridspec.GridSpec(
        nrows=3,
        ncols=1,
        height_ratios=[0.10, 0.75, 0.15],
        hspace=0.30,
        figure=fig,
    )

    # ----------------------------------------------------------------- header
    ax_hdr = fig.add_subplot(gs[0, 0])
    ax_hdr.axis("off")
    ax_hdr.text(
        0.5,
        0.78,
        title_text,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=title_color,
    )
    ax_hdr.text(
        0.5,
        0.36,
        f"hardware: {hw_name}  |  peak: {peak_tflops:.0f} TFLOPS "
        f"({compute_src}) / {peak_tbps:.1f} TB/s ({bw_src})  |  "
        f"ridge AI: {ridge_ai:.1f} FLOPS/byte",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )
    ax_hdr.text(
        0.5,
        0.06,
        (
            "per-kernel ncu measurements: arithmetic intensity vs achieved TFLOPS, "
            "annotated with max(DRAM%peak, SM%peak)"
            if page_mode == "ncu"
            else "no per-kernel arithmetic intensity measured -- see warning below"
        ),
        ha="center",
        va="center",
        fontsize=8,
        color="#777777",
        style="italic",
    )

    # ----------------------------------------------- scatter panel (log-log)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Decide axis range from data + the ceilings.
    all_ai: list[float] = []
    all_tflops: list[float] = []
    for vid, payload in cell_ncu.items():
        for k in payload.get("kernels", []):
            ai = k.get("arithmetic_intensity_flops_per_byte")
            tf = k.get("achieved_tflops")
            if ai is not None and ai > 0:
                all_ai.append(float(ai))
            if tf is not None and tf > 0:
                all_tflops.append(float(tf))

    # Reasonable defaults if some kernels lack data.
    ai_min = min(all_ai) / 5.0 if all_ai else 0.05
    ai_max = max(all_ai + [ridge_ai]) * 5.0 if (all_ai or ridge_ai) else 1000.0
    tflops_min = min(all_tflops) / 5.0 if all_tflops else 0.1
    tflops_max = peak_tflops * 1.5

    # Generate the roofline curve.
    import numpy as np

    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 200)
    # bandwidth-bound TFLOPS = AI * peak_bw_in_TFLOPS_per_FLOPS_per_byte
    # peak_bw_in_TFLOPS/(FLOPS/byte) = peak_tbps[TB/s] * 1e12[bytes/TB] / 1e12[FLOPS/TFLOPS]
    #                                 = peak_tbps TFLOPS per (FLOPS/byte)
    bw_curve = np.minimum(ai_grid * peak_tbps, peak_tflops)

    ax.plot(ai_grid, bw_curve, "k-", linewidth=2.0, label=f"roofline ({hardware_key})")
    # Annotate ceiling segments
    ax.axhline(peak_tflops, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(ridge_ai, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)

    # Plot each kernel point. Multi-cell support: same marker shape per
    # category, different edge color per cell.
    cell_ids = list(cell_ncu.keys())
    plotted_count = 0
    sol_only_count = 0
    for ci, vid in enumerate(cell_ids):
        payload = cell_ncu[vid]
        for k in payload.get("kernels", []):
            ai = k.get("arithmetic_intensity_flops_per_byte")
            tf = k.get("achieved_tflops")
            cat = k.get("category", "Other")
            marker = _CATEGORY_MARKERS.get(cat, "D")
            if ai is None or tf is None or ai <= 0 or tf <= 0:
                # %SoL-only fallback: arithmetic intensity is unmeasured
                # (ncu --set=basic captures no FLOPS/byte counters) but the
                # SoL section gives achieved SM throughput %. Plot the point
                # at y = SM% x the kernel family's compute ceiling, parked at
                # the ridge AI, and flag AI as unmeasured (never fabricated).
                sm_pct = k.get("achieved_sm_pct_peak")
                if sm_pct is None or sm_pct <= 0:
                    continue
                ceil_tflops = (
                    _category_compute_ceiling_tflops(cat, ceilings, hardware_key)
                    or peak_tflops
                )
                y_est = (sm_pct / 100.0) * ceil_tflops
                if y_est <= 0:
                    continue
                ax.scatter(
                    ridge_ai,
                    y_est,
                    s=80,
                    marker=marker,
                    facecolors="none",
                    edgecolors="#a36b00",
                    linewidths=1.3,
                    linestyle="dotted",
                    alpha=0.9,
                    label=f"{cat} (%SoL only)" if sol_only_count == 0 else None,
                )
                ax.annotate(
                    f"{k.get('name', '?')}  {sm_pct:.0f}%SoL (SM)  AI unmeasured",
                    xy=(ridge_ai, y_est),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="#a36b00",
                    style="italic",
                )
                sol_only_count += 1
                continue
            # Use the bigger of DRAM% and SM% as the displayed %SoL.
            dram_pct = k.get("achieved_dram_pct_peak")
            sm_pct = k.get("achieved_sm_pct_peak")
            pcts = [p for p in (dram_pct, sm_pct) if p is not None]
            sol_pct = max(pcts) if pcts else None

            ax.scatter(
                ai,
                tf,
                s=80,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.85,
                label=cat if ci == 0 else None,
            )
            label_parts = [k.get("name", "?")]
            if sol_pct is not None:
                label_parts.append(f"{sol_pct:.0f}%SoL")
            ax.annotate(
                "  ".join(label_parts),
                xy=(ai, tf),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#333333",
            )
            plotted_count += 1

    # Prominent warning banner when %SoL-only points were plotted: arithmetic
    # intensity is UNMEASURED, so the x-position is a placeholder and the chart
    # must not be read as a real roofline. States what is missing + HOW TO FIX.
    if sol_only_count > 0:
        ax.text(
            0.5,
            0.985,
            f"WARNING -- PARTIAL ROOFLINE: arithmetic intensity UNMEASURED for "
            f"{sol_only_count} kernel(s).\n"
            "These were captured with ncu --set=basic (no FLOPS / DRAM-byte "
            "counters), so only %SoL (SM throughput) is measured; the hollow "
            "dotted markers are parked at the ridge AI as a PLACEHOLDER x-position "
            "(not a measurement).\n"
            "HOW TO FIX: re-capture with --roofline-min (or --set full) per "
            "perf-tune-glm51/ncu-sister/REPLAY-MODE-APPLICATION-RUNBOOK.md, then "
            "re-import + re-render for a true arithmetic-intensity roofline.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#aa3333",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#fff5f5",
                "edgecolor": "#aa3333",
                "linewidth": 1.5,
            },
        )

    # Added in v1.23.2: when no ncu kernels are plottable (all AI / TFLOPS
    # are null because ncu was captured with --set=basic) AND no %SoL-only
    # point could be drawn, fall back to workload-level DCGM-derived points
    # from per_category_attribution.
    used_dcgm_fallback = False
    if plotted_count == 0 and sol_only_count == 0 and cell_dcgm:
        used_dcgm_fallback = _plot_dcgm_fallback(
            ax, cell_dcgm, ridge_ai=ridge_ai, peak_tflops=peak_tflops, peak_tbps=peak_tbps,
        )

    # Added in v1.23.2: when STILL no points (no ncu kernels + no %SoL-only +
    # no DCGM per_category_attribution), draw a centered annotation explaining
    # why so the operator doesn't read the empty scatter as "broken".
    if plotted_count == 0 and sol_only_count == 0 and not used_dcgm_fallback:
        ax.text(
            0.5,
            0.5,
            "No roofline-ready measurements found.\n\n"
            "ncu_kernels.json has all-null arithmetic_intensity_flops_per_byte / achieved_tflops\n"
            "(captured with --set=basic; FLOPS + DRAM-bytes counters require --set=full)\n"
            "AND no dcgm_correlation.json with per_category_attribution found.\n\n"
            "See OPERATOR-TODO.md TODO-NCU-FULL-SET-RECAPTURE.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#aa3333",
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": "#fff5f5",
                "edgecolor": "#aa3333",
                "linewidth": 1.0,
            },
        )

    ax.set_xlabel("Arithmetic Intensity (FLOPS / byte)", fontsize=9)
    ax.set_ylabel("Achieved TFLOPS", fontsize=9)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(tflops_min, tflops_max)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)

    # Legend (deduped by category, since we only labelled the first cell)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=7, frameon=True)

    # ----------------------------------------------------------- caveat
    ax_cv = fig.add_subplot(gs[2, 0])
    ax_cv.axis("off")
    ax_cv.text(
        0.5,
        0.72,
        "Roofline reading: kernels on the diagonal line are at the HBM bandwidth ceiling; "
        "kernels at the horizontal line are at the compute ceiling.",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )
    ax_cv.text(
        0.5,
        0.42,
        "Caveat: ncu replay slowdown affects timing; achieved TFLOPS comes from "
        "summed FLOPS / summed wall-clock across launch-count replays.",
        ha="center",
        va="center",
        fontsize=7,
        style="italic",
        color="#666666",
    )
    ax_cv.text(
        0.5,
        0.12,
        "See AGENTS.md 'Speed-of-light framing' for the three-level rigor "
        "hierarchy (sample-share -> ncu per-kernel -> DCGM workload-level).",
        ha="center",
        va="center",
        fontsize=7,
        color="#777777",
    )

    # Report partial status so the caller records it (report_status.json +
    # completeness page). A %SoL-only or empty page is partial; a real ncu-AI
    # page or a measured DCGM fallback is not.
    if page_mode == "solonly":
        return Page5Status(partial=True, reason="ncu_scatter_solonly")
    if page_mode == "empty":
        return Page5Status(partial=True, reason="ncu_scatter_empty")
    return Page5Status(partial=False)
