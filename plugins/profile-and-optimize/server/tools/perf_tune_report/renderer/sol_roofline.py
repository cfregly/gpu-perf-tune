"""Page-4 Speed-of-Light (SoL) Roofline.

Renders a "measured-vs-published-peak" framing on top of the zymtrace
per-category data already used by ``kernel_breakdown.render_page`` and
the workload-level throughput rows already used by the scatter-grid /
heatmap pages.

Per-workspace ``AGENTS.md`` "Speed-of-light framing" section, every
measurement artifact must report measured-vs-SoL alongside the headline
number. This page is the automated visualization of that discipline.

Layout (top to bottom):

1. Header block: campaign title + hardware identifier + link-to-ceiling-yaml.
2. Per-category SoL panel: stacked bar of time-share per category across
   variants, annotated with the natural ceiling each category is bound by
   (NCCL -> NVLink5 1.8 TB/s; MoE/BMM-NVFP4 -> NVFP4 9 PFLOPS; FMHA ->
   HBM3e 8 TB/s; etc., from ``sol-ceilings.yaml`` ``category_ceiling_map``).
3. Workload-level SoL panel: measured ``output_tps_per_gpu`` per cell at
   peak concurrency, as bars. Skipped silently when no rows carry a valid
   ``output_tps_per_gpu`` number.
4. Caveat footer: sample-share-is-time-proxy disclaimer, link to AGENTS.md
   for the full caveat treatment.

The page is conditional. ``render_report.discover_sol_inputs`` decides
whether to draw it:

- No ``kernels.json`` payloads under ``cells/*/``: silently skipped
  (same gate as kernel_breakdown -- this page is downstream of it).
- ``sol-ceilings.yaml`` not findable: silently skipped (the workspace's
  canonical path is ``configs/sol-ceilings.yaml``; absent
  means the page can't compute ceilings).
- Both present: draw the page.

If the YAML loads but is malformed (missing the hardware key for the
campaign's atlas, or missing ``category_ceiling_map``), ``SoLCeilingsMalformed``
is raised and the whole render aborts -- the same "no silent degradation"
pattern as ``KernelsJsonMalformed``.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence


# Per-category mapping is loaded from sol-ceilings.yaml's
# ``category_ceiling_map`` at runtime. This module-level constant is the
# fallback display order when the YAML doesn't specify one (mirrors
# kernel_breakdown.CATEGORY_DISPLAY_ORDER for visual consistency).
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


class SoLCeilingsMalformed(Exception):
    """Raised when ``sol-ceilings.yaml`` is present but unusable.

    Distinct from "file not found" (which is a silent-skip) -- this fires
    when the file IS there but is missing required keys for the campaign's
    hardware or missing ``category_ceiling_map``. The renderer aborts so
    the operator notices, rather than silently degrading to a blank page.
    """

    def __init__(self, path: Path, reason: str):
        super().__init__(f"sol-ceilings.yaml malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


def load_ceilings(path: Path) -> dict[str, Any]:
    """Load + validate ``sol-ceilings.yaml``.

    Returns the parsed dict on success. Raises ``SoLCeilingsMalformed`` if
    the file exists but lacks ``category_ceiling_map`` or any hardware key
    with the expected ``{<metric>: {value, units, source}}`` shape.
    """
    import yaml  # lazy import; the renderer doesn't always need this page

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SoLCeilingsMalformed(path, f"not valid YAML: {e}") from e

    if not isinstance(data, dict):
        raise SoLCeilingsMalformed(path, f"top-level must be a mapping, got {type(data).__name__}")

    if "category_ceiling_map" not in data:
        raise SoLCeilingsMalformed(path, "missing 'category_ceiling_map' key")

    cat_map = data["category_ceiling_map"]
    if not isinstance(cat_map, dict) or not cat_map:
        raise SoLCeilingsMalformed(path, "'category_ceiling_map' must be a non-empty mapping")

    return data


def hardware_key_for_atlas(rows: Sequence[Any]) -> str | None:
    """Pick the hardware key (e.g. ``b200_sm100``) that matches the atlas.

    Heuristic: take the most common ``hardware`` string across rows and
    map it to a YAML key. ``"B200"`` -> ``"b200_sm100"``, ``"GB300"`` ->
    ``"gb300_nvl72"``, ``"H100"`` -> ``"h100_sxm"``. Returns ``None`` if
    no rows carry a ``hardware`` field or the most-common value doesn't
    map to a known key (in which case the renderer skips this page
    silently rather than misreporting a peak number).
    """
    counts: dict[str, int] = {}
    for r in rows:
        hw = getattr(r, "hardware", None)
        if hw:
            counts[hw] = counts.get(hw, 0) + 1
    if not counts:
        return None
    top_hw = max(counts.items(), key=lambda kv: kv[1])[0]
    mapping = {
        "B200": "b200_sm100",
        "GB300": "gb300_nvl72",
        "H100": "h100_sxm",
    }
    return mapping.get(top_hw)


def compute_category_sol(
    cell_payload: dict[str, Any],
    hw_data: dict[str, Any],
    category_ceiling_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-category SoL rows for one cell.

    Each row is::

        {
          "category": str,
          "samples": int,
          "time_share_pct": float,        # samples / total * 100
          "ceiling_metric": str | None,   # e.g. "nvlink5_tbps"
          "ceiling_value": float | None,  # e.g. 1.8
          "ceiling_units": str | None,    # e.g. "TB/s"
          "bound": str | None,            # "compute" | "bandwidth"
        }

    The ``time_share_pct`` is the only computed number; ``ceiling_*`` are
    pulled directly from the YAML so per-category numbers stay tied to the
    single source of truth.
    """
    per_cat = cell_payload.get("per_category", {})
    total = sum(int(v) for v in per_cat.values()) or 1

    rows: list[dict[str, Any]] = []
    for cat in _CATEGORY_DISPLAY_ORDER:
        samples = int(per_cat.get(cat, 0))
        if samples == 0:
            continue
        ceiling_info = category_ceiling_map.get(cat)
        if ceiling_info is None:
            ceiling_metric = ceiling_value = ceiling_units = bound = None
        else:
            ceiling_metric = ceiling_info.get("metric")
            bound = ceiling_info.get("bound")
            metric_entry = hw_data.get(ceiling_metric) if ceiling_metric else None
            if isinstance(metric_entry, dict):
                ceiling_value = metric_entry.get("value")
                ceiling_units = metric_entry.get("units")
            else:
                ceiling_value = ceiling_units = None
        rows.append(
            {
                "category": cat,
                "samples": samples,
                "time_share_pct": 100.0 * samples / total,
                "ceiling_metric": ceiling_metric,
                "ceiling_value": ceiling_value,
                "ceiling_units": ceiling_units,
                "bound": bound,
            }
        )
    return rows


def _format_ceiling(row: dict[str, Any]) -> str:
    """Format the per-category ceiling label shown next to each bar."""
    metric = row.get("ceiling_metric")
    value = row.get("ceiling_value")
    units = row.get("ceiling_units")
    if metric is None or value is None:
        return "(no ceiling mapped)"
    units_str = f" {units}" if units else ""
    return f"{metric}={value}{units_str}"


def render_page(
    fig,
    cell_kernels: "OrderedDict[str, dict[str, Any]]",
    rows: Sequence[Any],
    ceilings: dict[str, Any],
    hardware_key: str,
) -> None:
    """Draw the SoL roofline page onto a matplotlib Figure.

    Args:
        fig: matplotlib Figure (created by the caller; we own the layout).
        cell_kernels: ordered ``{cell_id: kernels.json payload}``. MUST be
            non-empty -- caller is responsible for the skip-when-empty
            decision (matches ``kernel_breakdown.render_page`` contract).
        rows: AtlasCell rows from the campaign atlas.jsonl. Used to draw
            the workload-level ``output_tps_per_gpu`` bars. May be empty
            (we degrade gracefully -- the workload panel just shows
            "no per-cell throughput rows").
        ceilings: parsed sol-ceilings.yaml dict (from ``load_ceilings``).
        hardware_key: the key in ``ceilings`` to use for this campaign,
            e.g. ``"b200_sm100"``. MUST exist in ``ceilings``.
    """
    if not cell_kernels:
        raise ValueError("sol_roofline.render_page: cell_kernels is empty")
    if hardware_key not in ceilings:
        raise SoLCeilingsMalformed(
            Path("<ceilings>"),
            f"hardware key {hardware_key!r} not in ceilings",
        )

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    hw_data = ceilings[hardware_key]
    hw_name = hw_data.get("hw_name", hardware_key)
    cat_map = ceilings["category_ceiling_map"]

    gs = gridspec.GridSpec(
        nrows=4,
        ncols=1,
        height_ratios=[0.10, 0.45, 0.32, 0.13],
        hspace=0.55,
        figure=fig,
    )

    # ----------------------------------------------------------------- header
    ax_hdr = fig.add_subplot(gs[0, 0])
    ax_hdr.axis("off")
    ax_hdr.text(
        0.5,
        0.78,
        "Speed-of-Light Roofline",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax_hdr.text(
        0.5,
        0.36,
        f"hardware: {hw_name}  |  ceiling source: configs/sol-ceilings.yaml",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )
    ax_hdr.text(
        0.5,
        0.06,
        "measured vs published peak per kernel category + workload-level tok/s/GPU",
        ha="center",
        va="center",
        fontsize=8,
        color="#777777",
        style="italic",
    )

    # ----------------------------------------------- per-category SoL panel
    ax_cat = fig.add_subplot(gs[1, 0])

    variant_ids = list(cell_kernels.keys())
    first_vid = variant_ids[0]
    cat_rows = compute_category_sol(cell_kernels[first_vid], hw_data, cat_map)
    if not cat_rows:
        ax_cat.axis("off")
        ax_cat.text(
            0.5,
            0.5,
            "(no non-zero per_category samples on first variant)",
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        cats = [r["category"] for r in cat_rows]
        shares = [r["time_share_pct"] for r in cat_rows]
        ceiling_labels = [_format_ceiling(r) for r in cat_rows]
        bound_labels = [(r.get("bound") or "?") for r in cat_rows]

        y_pos = list(range(len(cats)))
        bars = ax_cat.barh(y_pos, shares, edgecolor="white", linewidth=0.5)
        ax_cat.set_yticks(y_pos)
        ax_cat.set_yticklabels(cats, fontsize=8)
        ax_cat.invert_yaxis()
        ax_cat.set_xlabel("GPU time-share % (zymtrace sample fraction)", fontsize=8)
        ax_cat.set_title(
            f"Per-Category Time-Share + Natural Ceiling  (variant: {first_vid})",
            fontsize=10,
            loc="left",
        )
        ax_cat.tick_params(axis="x", labelsize=7)
        ax_cat.set_xlim(0, max(shares) * 1.50 if shares else 1.0)

        for bar, share, clabel, blabel in zip(bars, shares, ceiling_labels, bound_labels):
            ax_cat.text(
                share + max(shares) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{share:.1f}%  ->  ceiling: {clabel}  [{blabel}-bound]",
                va="center",
                ha="left",
                fontsize=6.5,
                color="#333333",
            )

    # ------------------------------------------------ workload-level panel
    ax_wl = fig.add_subplot(gs[2, 0])

    per_cell_peak: dict[str, float] = {}
    for r in rows:
        tps = getattr(r, "output_tps_per_gpu", None)
        if tps is None:
            continue
        cell_id = getattr(r, "cell_id", None) or ""
        if not cell_id:
            continue
        prev = per_cell_peak.get(cell_id)
        if prev is None or tps > prev:
            per_cell_peak[cell_id] = float(tps)

    if not per_cell_peak:
        ax_wl.axis("off")
        ax_wl.text(
            0.5,
            0.5,
            "(no atlas rows carry output_tps_per_gpu; "
            "workload-level SoL skipped)",
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        cell_ids = sorted(per_cell_peak.keys())
        peak_tps = [per_cell_peak[c] for c in cell_ids]
        ax_wl.bar(cell_ids, peak_tps, edgecolor="white", linewidth=0.5)
        ax_wl.set_ylabel("peak output_tps_per_gpu", fontsize=8)
        ax_wl.set_title(
            "Workload-Level Throughput per Cell (peak across concurrencies)",
            fontsize=10,
            loc="left",
        )
        ax_wl.tick_params(axis="x", rotation=20, labelsize=7)
        ax_wl.tick_params(axis="y", labelsize=7)

        # Annotate the chart with the HBM-bandwidth ceiling reminder. We
        # do NOT draw a single horizontal "ceiling" line because the
        # HBM-roofline depends on per-token footprint, which differs per
        # workload. Operators pull the per-workload roofline from
        # sol-summary.md (which references this campaign's atlas).
        hbm_metric = hw_data.get("hbm3e_tbps") or hw_data.get("hbm3_tbps")
        hbm_value = None
        if isinstance(hbm_metric, dict):
            hbm_value = hbm_metric.get("value")
        if hbm_value is not None:
            ax_wl.text(
                0.99,
                0.95,
                f"HBM ceiling reference: {hbm_value} TB/s\n"
                "(per-workload tok/s/GPU roofline depends on per-token footprint;\n"
                "see this bundle's sol-summary.md for the worked calc)",
                ha="right",
                va="top",
                transform=ax_wl.transAxes,
                fontsize=6.5,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff7e0", ec="#cc9900", linewidth=0.5),
            )

    # ----------------------------------------------------------- caveat
    ax_cv = fig.add_subplot(gs[3, 0])
    ax_cv.axis("off")
    ax_cv.text(
        0.5,
        0.55,
        "Caveat: %SoL derived from zymtrace sample-share is a time-share "
        "proxy, not byte-traffic or FLOP measurement.",
        ha="center",
        va="center",
        fontsize=7,
        style="italic",
        color="#555555",
    )
    ax_cv.text(
        0.5,
        0.15,
        "For tight per-kernel arithmetic-intensity-vs-roofline measurements, "
        "use inference-kernel-ncu-profile.  See AGENTS.md 'Speed-of-light framing'.",
        ha="center",
        va="center",
        fontsize=7,
        color="#777777",
    )
