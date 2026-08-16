"""Page-6b DCGM x zymtrace per-category attribution renderer.

Level-2 in the four-level Speed-of-Light rigor hierarchy:

  L1 (page 4):  zymtrace sample-share proxy
  L2 (page 6b): zymtrace x DCGM cross-attribution  (this module)
  L3 (page 6):  DCGM workload-level byte traffic
  L4 (page 5):  ncu per-kernel arithmetic intensity scatter

Reads the ``per_category_attribution`` block from
``dcgm_correlation.json`` (populated by
``tools/perf_tune_report/dcgm_correlate.py`` when ``correlate()`` was
invoked with ``kernels_json_path=``). Each row = one zymtrace category
with measured byte/FLOP attribution from the DCGM workload totals.

Layout (top to bottom):

1. Header block: campaign title + hardware + sweep window + caveat banner.
2. Per-category grouped bar chart: x-axis is category; for each category
   show two grouped bars (time-share % + %SoL of category ceiling). Labels
   beneath each bar show the absolute attributed bytes / FLOPS.
3. Caveat footer: cross-attribution is "measured per-category" but
   relies on zymtrace time-share as the attribution weight; the
   assumption is that DCGM workload totals distribute uniformly within
   each category's time window.

The page is conditional. Caller (render_report) decides skip-when-empty.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


_CATEGORY_DISPLAY_ORDER: tuple[str, ...] = (
    "NCCL",
    "MoE",
    "FMHA",
    "BMM-NVFP4",
    "Triton-fused",
    "cuBLAS",
    "Elementwise",
    "Other",
)


def _fmt_bytes(b: float | None) -> str:
    """Compact byte formatting for inline bar labels."""
    if b is None:
        return "?"
    if b >= 1e15:
        return f"{b / 1e15:.2f} PB"
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.0f} MB"
    return f"{b:.0f} B"


def _fmt_flops(f: float | None) -> str:
    if f is None:
        return "?"
    if f >= 1e18:
        return f"{f / 1e18:.2f} EFLOPS"
    if f >= 1e15:
        return f"{f / 1e15:.2f} PFLOPS"
    if f >= 1e12:
        return f"{f / 1e12:.1f} TFLOPS"
    if f >= 1e9:
        return f"{f / 1e9:.0f} GFLOPS"
    return f"{f:.0f} FLOPS"


def render_page(
    fig,
    cell_dcgm: "OrderedDict[str, dict[str, Any]]",
) -> None:
    """Draw the per-category attribution page onto a matplotlib Figure.

    Args:
        fig: matplotlib Figure (created by the caller).
        cell_dcgm: ordered ``{cell_id: dcgm_correlation.json payload}``.
            MUST be non-empty AND the first cell's payload MUST have a
            non-empty ``per_category_attribution`` list. Caller handles
            the skip-when-empty decision.
    """
    if not cell_dcgm:
        raise ValueError("dcgm_category_attribution.render_page: cell_dcgm is empty")

    first_vid = next(iter(cell_dcgm.keys()))
    payload = cell_dcgm[first_vid]
    attribution = payload.get("per_category_attribution") or []
    if not attribution:
        raise ValueError(
            "dcgm_category_attribution.render_page: per_category_attribution is empty"
        )

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec

    hw_key = payload.get("hw_key", "?")
    sweep_start = payload.get("sweep_start_utc", "?")
    sweep_end = payload.get("sweep_end_utc", "?")
    duration_s = payload.get("duration_s", 0)
    n_gpus = payload.get("n_gpus", 0)

    gs = gridspec.GridSpec(
        nrows=3,
        ncols=1,
        height_ratios=[0.16, 0.65, 0.19],
        hspace=0.30,
        figure=fig,
    )

    # ----------------------------------------------------------------- header
    ax_hdr = fig.add_subplot(gs[0, 0])
    ax_hdr.axis("off")
    ax_hdr.text(
        0.5, 0.78,
        "DCGM x zymtrace Per-Category Attribution",
        ha="center", va="center", fontsize=14, fontweight="bold",
    )
    ax_hdr.text(
        0.5, 0.46,
        f"variant: {first_vid}  |  hardware: {hw_key}  |  "
        f"GPUs: {n_gpus}  |  sweep duration: {duration_s:.0f}s",
        ha="center", va="center", fontsize=8, color="#555555",
    )
    ax_hdr.text(
        0.5, 0.18,
        "Measured DCGM workload bytes / FLOPS attributed to zymtrace categories via time-share weighting",
        ha="center", va="center", fontsize=8, color="#777777", style="italic",
    )

    # --------------------------------- grouped bar chart (time-share + %SoL)
    ax = fig.add_subplot(gs[1, 0])

    # Order by the canonical display order; categories absent from the
    # payload are skipped.
    by_cat = {a["category"]: a for a in attribution}
    cats = [c for c in _CATEGORY_DISPLAY_ORDER if c in by_cat]
    # Any unmapped categories appended at the end:
    for a in attribution:
        if a["category"] not in cats:
            cats.append(a["category"])

    if not cats:
        ax.axis("off")
        ax.text(0.5, 0.5, "(no per-category rows)", ha="center", va="center", fontsize=10)
    else:
        time_shares = [by_cat[c].get("time_share_pct") or 0.0 for c in cats]
        # %SoL: use whichever is non-None (bandwidth or compute) per category.
        sol_pcts = []
        sol_kinds = []  # "BW" or "Compute" or "?"
        for c in cats:
            row = by_cat[c]
            if row.get("sol_pct_bw") is not None:
                sol_pcts.append(row["sol_pct_bw"])
                sol_kinds.append("BW")
            elif row.get("sol_pct_compute") is not None:
                sol_pcts.append(row["sol_pct_compute"])
                sol_kinds.append("Compute")
            else:
                sol_pcts.append(0.0)
                sol_kinds.append("?")

        x = np.arange(len(cats))
        width = 0.38
        ax.bar(x - width / 2, time_shares, width, label="time-share %", edgecolor="white", linewidth=0.4)
        ax.bar(x + width / 2, sol_pcts, width, label="%SoL of category ceiling", edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8, rotation=20)
        ax.set_ylabel("percentage (%)", fontsize=8)
        ax.set_title(
            f"Time-share + Cross-attributed %SoL per Category (variant: {first_vid})",
            fontsize=10, loc="left",
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.axhline(100, color="#cc3333", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        ax.set_ylim(0, max(100.0, max(time_shares + sol_pcts, default=0) * 1.15))

        # Per-bar annotation showing the underlying byte / FLOP attribution.
        for i, c in enumerate(cats):
            row = by_cat[c]
            kind = sol_kinds[i]
            if kind == "BW":
                detail = _fmt_bytes(row.get("attributed_bytes_total"))
            elif kind == "Compute":
                detail = _fmt_flops(row.get("attributed_flops_total"))
            else:
                detail = "no ceiling"
            ax.text(
                x[i],
                -max(time_shares + sol_pcts, default=0) * 0.04,
                f"{detail} [{kind}]",
                ha="center", va="top", fontsize=6.5, color="#444444",
            )

    # ------------------------------------------------------------ caveat
    ax_cv = fig.add_subplot(gs[2, 0])
    ax_cv.axis("off")
    ax_cv.text(
        0.5, 0.75,
        "How to read: time-share is the fraction of GPU time the category occupied "
        "(zymtrace samples). %SoL is the cross-attributed measured throughput "
        "during that window divided by the category's natural ceiling.",
        ha="center", va="center", fontsize=7, color="#555555", wrap=True,
    )
    ax_cv.text(
        0.5, 0.42,
        "Caveat: cross-attribution assumes DCGM workload bytes/FLOPS distribute "
        "uniformly within each category's time window. For per-individual-kernel "
        "rigor see the L4 page (ncu scatter); for total workload truth see the L3 "
        "page (DCGM bars).",
        ha="center", va="center", fontsize=7, style="italic", color="#666666",
    )
    ax_cv.text(
        0.5, 0.10,
        "See AGENTS.md 'Speed-of-light framing' for the four-level rigor hierarchy.",
        ha="center", va="center", fontsize=7, color="#777777",
    )
