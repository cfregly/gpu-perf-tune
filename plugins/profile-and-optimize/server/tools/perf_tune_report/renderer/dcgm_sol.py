"""Page-6 DCGM byte-grounded workload SoL renderer.

The third tier of the workspace SoL rigor hierarchy:

  page 4: sample-share proxy (zymtrace)
  page 5: ncu per-kernel arithmetic intensity scatter
  page 6: DCGM workload-level byte traffic (this module)

Reads ``dcgm_correlation.json`` per-cell payloads emitted by
``tools/perf_tune_report/dcgm_correlate.py`` (one row per peak that mapped
to a DCGM metric, with ``measured_bytes_total``,
``measured_bytes_per_s``, ``measured_tflops_avg``, ``sol_pct``, and
``notes``).

Layout (top to bottom):

1. Header block: campaign title, hw_key, sweep window + DCGM group level.
2. Per-resource horizontal bar chart: one bar per resource, length =
   %SoL, annotated with absolute measured number and the peak it was
   compared against.
3. Caveat footer: DCGM scrape granularity caveat + short-sweep warning
   when applicable.

The page is conditional. Caller decides skip-when-empty via
``render_report.discover_dcgm_payloads``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def _fmt_throughput(res: dict[str, Any]) -> str:
    """Format the right-hand annotation for a resource bar."""
    if res.get("measured_bytes_per_s") is not None:
        bps = float(res["measured_bytes_per_s"])
        if bps >= 1e12:
            return f"{bps/1e12:.2f} TB/s/GPU"
        elif bps >= 1e9:
            return f"{bps/1e9:.1f} GB/s/GPU"
        else:
            return f"{bps/1e6:.0f} MB/s/GPU"
    if res.get("measured_tflops_avg") is not None:
        tf = float(res["measured_tflops_avg"])
        if tf >= 1000:
            return f"{tf/1000:.2f} PFLOPS/GPU"
        else:
            return f"{tf:.1f} TFLOPS/GPU"
    return "(no measurement)"


def _fmt_peak(res: dict[str, Any]) -> str:
    """Format the ceiling description for a resource bar."""
    val = res.get("peak_per_gpu")
    units = res.get("peak_per_gpu_units", "")
    if val is None:
        return ""
    return f"peak={val:.1f} {units}/GPU x {res.get('n_gpus', 0)} GPUs"


def render_page(
    fig,
    cell_dcgm: "OrderedDict[str, dict[str, Any]]",
) -> None:
    """Draw the DCGM workload-level SoL page onto a matplotlib Figure.

    Args:
        fig: matplotlib Figure (created by the caller).
        cell_dcgm: ordered ``{cell_id: dcgm_correlation.json payload}``.
            MUST be non-empty (caller handles skip-when-empty).
    """
    if not cell_dcgm:
        raise ValueError("dcgm_sol.render_page: cell_dcgm is empty")

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # For simplicity, the first cell drives the layout; multi-cell
    # campaigns get one page per cell from the caller iterating.
    first_vid = next(iter(cell_dcgm.keys()))
    payload = cell_dcgm[first_vid]
    resources = payload.get("resources", [])
    hw_key = payload.get("hw_key", "?")
    sweep_start = payload.get("sweep_start_utc", "?")
    sweep_end = payload.get("sweep_end_utc", "?")
    duration_s = payload.get("duration_s", 0)
    group_level = payload.get("dcgm_group_level", "?")
    short_sweep = payload.get("short_sweep_warning", False)
    n_gpus = payload.get("n_gpus", 0)

    gs = gridspec.GridSpec(
        nrows=3,
        ncols=1,
        height_ratios=[0.16, 0.69, 0.15],
        hspace=0.30,
        figure=fig,
    )

    # ----------------------------------------------------------------- header
    ax_hdr = fig.add_subplot(gs[0, 0])
    ax_hdr.axis("off")
    ax_hdr.text(
        0.5,
        0.78,
        "DCGM Byte-Grounded Workload SoL",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax_hdr.text(
        0.5,
        0.47,
        f"variant: {first_vid}  |  hardware: {hw_key}  |  "
        f"GPUs: {n_gpus}  |  DCGM group: {group_level}",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )
    ax_hdr.text(
        0.5,
        0.20,
        f"sweep window: {sweep_start} -> {sweep_end}  ({duration_s:.0f}s)",
        ha="center",
        va="center",
        fontsize=8,
        color="#777777",
        style="italic",
    )

    # ----------------------------------------------- per-resource bar chart
    ax = fig.add_subplot(gs[1, 0])
    if not resources:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "(no resources in dcgm_correlation.json)",
            ha="center",
            va="center",
            fontsize=10,
        )
    else:
        # Order: bandwidth-style first (TB/s peaks), then compute-style.
        ordered = sorted(
            resources,
            key=lambda r: (r.get("peak_per_gpu_units", "") != "TB/s", r.get("peak_key", "")),
        )
        labels = [r.get("peak_key", "?") for r in ordered]
        sol_values = [
            (r.get("sol_pct") if r.get("sol_pct") is not None else 0.0)
            for r in ordered
        ]

        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, sol_values, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of Speed-of-Light (measured byte/FLOP traffic vs peak x duration x n_gpus)", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_title(
            f"Per-Resource Workload SoL (variant: {first_vid})",
            fontsize=10,
            loc="left",
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.axvline(100, color="#cc3333", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(99, -0.6, "ceiling", color="#cc3333", fontsize=7, ha="right")

        for bar, res in zip(bars, ordered):
            sol = res.get("sol_pct")
            sol_str = f"{sol:.1f}%" if sol is not None else "n/a"
            label = f"{sol_str}  |  {_fmt_throughput(res)}  |  {_fmt_peak(res)}"
            ax.text(
                max(2.0, (sol or 0.0) + 2.0),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=6.5,
                color="#333333",
            )

    # ----------------------------------------------------------- caveat
    ax_cv = fig.add_subplot(gs[2, 0])
    ax_cv.axis("off")
    cav_y = 0.65
    if short_sweep:
        ax_cv.text(
            0.5,
            cav_y,
            "Warning: short sweep < dcgm_config.min_sweep_seconds; "
            "DCGM scrape granularity makes integration coarse.",
            ha="center",
            va="center",
            fontsize=7,
            color="#cc6600",
        )
        cav_y -= 0.30
    if group_level == "counter":
        ax_cv.text(
            0.5,
            cav_y,
            "DCGM_FI_PROF group not exported on this cluster; "
            "falling back to DCGM_FI_DEV_* counter-tier (coarser).",
            ha="center",
            va="center",
            fontsize=7,
            color="#cc6600",
        )
        cav_y -= 0.25
    elif group_level == "absent":
        ax_cv.text(
            0.5,
            cav_y,
            "DCGM exporter not detected on this cluster; "
            "no byte-grounded measurements available.",
            ha="center",
            va="center",
            fontsize=7,
            color="#cc3333",
        )
        cav_y -= 0.25
    ax_cv.text(
        0.5,
        cav_y if cav_y > 0 else 0.15,
        "See AGENTS.md 'Speed-of-light framing' for the three-level rigor "
        "hierarchy (sample-share -> ncu per-kernel -> DCGM workload-level).",
        ha="center",
        va="center",
        fontsize=7,
        color="#777777",
    )
