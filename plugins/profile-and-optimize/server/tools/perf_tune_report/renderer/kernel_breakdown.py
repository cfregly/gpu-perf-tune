"""Page-3 GPU Kernel Breakdown.

Renders the zymtrace per-kernel data ingested by
``tools.perf_tune_report.importers.zymtrace_kernels`` into a single PDF page.

Layout (top to bottom):

1. Header block (campaign title + page subtitle).
2. Per-category stacked bar chart across all variants that have
   ``kernels.json``. X = variant cell_id, Y = sample count, stacked by
   category {NCCL, MoE, FMHA, BMM-NVFP4, Triton-fused, cuBLAS, Elementwise,
   Other}. Reveals "did this variant shift NCCL or MoE share?".
3. Top-20 kernels table for the first variant with kernels.json. Columns:
   rank, kernel name (truncated), samples, % of total, category.
4. Side-by-side: per-GPU sample distribution (left) + top-10 cpython frames
   during CUDA events (right). Both keyed to the first variant. Surfaces
   "which GPU is overloaded?" and "which Python code launches the hot
   kernels?".

The page is conditional on the renderer finding at least one valid
``kernels.json`` under the campaign's ``cells/`` tree. See
``render_report.discover_kernels_payloads`` for the discovery + validation
logic. If no payloads are found the page is skipped silently (correct
for non-zymtrace campaigns). If a payload is present but malformed,
``KernelsJsonMalformed`` is raised at discovery time and the renderer
aborts before drawing anything.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence


# Display order for the stacked bar's categories. Mirrors the bucketing
# in perf-tune-glm51/03f-variant-runner.sh phase 5d and in
# ``importers.zymtrace_kernels._CATEGORY_RULES``.
CATEGORY_DISPLAY_ORDER: tuple[str, ...] = (
    "NCCL",
    "MoE",
    "FMHA",
    "BMM-NVFP4",
    "Triton-fused",
    "cuBLAS",
    "Elementwise",
    "Other",
)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def render_page(fig, cell_kernels: "OrderedDict[str, dict[str, Any]]") -> None:
    """Draw the kernel-breakdown page onto a matplotlib Figure.

    Args:
        fig: matplotlib Figure (created by the caller; we own the layout).
        cell_kernels: ordered dict of ``{cell_id: kernels.json payload}``.
            MUST have at least one entry — caller is responsible for the
            "skip the page entirely when empty" decision.
    """
    if not cell_kernels:
        raise ValueError("kernel_breakdown.render_page: cell_kernels is empty")

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # 3 vertical zones: header (small), upper plot, lower table block.
    gs = gridspec.GridSpec(
        nrows=4,
        ncols=2,
        height_ratios=[0.08, 0.32, 0.30, 0.30],
        hspace=0.55,
        wspace=0.18,
        figure=fig,
    )

    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.text(
        0.5,
        0.75,
        "GPU Kernel Breakdown (zymtrace)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax_header.text(
        0.5,
        0.15,
        f"{len(cell_kernels)} variant(s) with kernels.json; "
        "samples from zymtrace_profiling.events JOIN interp_funcs",
        ha="center",
        va="center",
        fontsize=8,
        color="#666666",
    )

    # Zone 2: per-category stacked bar across variants.
    ax_cat = fig.add_subplot(gs[1, :])
    variant_ids = list(cell_kernels.keys())
    cat_totals_per_variant: dict[str, list[float]] = {
        cat: [] for cat in CATEGORY_DISPLAY_ORDER
    }
    for vid in variant_ids:
        cats = cell_kernels[vid].get("per_category", {})
        for cat in CATEGORY_DISPLAY_ORDER:
            cat_totals_per_variant[cat].append(float(cats.get(cat, 0)))

    bottom = [0.0] * len(variant_ids)
    for cat in CATEGORY_DISPLAY_ORDER:
        vals = cat_totals_per_variant[cat]
        if all(v == 0 for v in vals):
            continue
        ax_cat.bar(
            variant_ids,
            vals,
            bottom=bottom,
            label=cat,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax_cat.set_title("Per-Category CUDA Sample Share by Variant", fontsize=10)
    ax_cat.set_ylabel("samples", fontsize=8)
    ax_cat.tick_params(axis="x", rotation=20, labelsize=7)
    ax_cat.tick_params(axis="y", labelsize=7)
    ax_cat.legend(
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        fontsize=7,
        frameon=False,
    )

    # Zone 3: top-20 kernels table for the first variant.
    first_vid = variant_ids[0]
    first = cell_kernels[first_vid]
    top_kernels = first.get("top_kernels", [])[:20]
    total = sum(int(k["samples"]) for k in top_kernels) or 1

    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    ax_table.set_title(
        f"Top-20 Kernels (variant: {first_vid})", fontsize=10, loc="left"
    )
    if top_kernels:
        cell_text = [
            [
                str(i + 1),
                _truncate(k["name"], 95),
                f"{int(k['samples']):,}",
                f"{100 * int(k['samples']) / total:.1f}%",
                k.get("category", "Other"),
            ]
            for i, k in enumerate(top_kernels)
        ]
        tbl = ax_table.table(
            cellText=cell_text,
            colLabels=["#", "kernel", "samples", "% of top-20", "category"],
            colWidths=[0.04, 0.66, 0.10, 0.10, 0.10],
            cellLoc="left",
            loc="upper left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        tbl.scale(1, 1.05)
    else:
        ax_table.text(0.5, 0.5, "(no top_kernels rows)", ha="center", va="center")

    # Zone 4 left: per-GPU bar.
    ax_gpu = fig.add_subplot(gs[3, 0])
    per_gpu = first.get("per_gpu", [])
    if per_gpu:
        # Short-label each GPU as the last 6 chars of uuid (B200 boxes have
        # 8 identical names so we have to disambiguate by uuid suffix).
        labels = [g["gpu_uuid"][-6:] if g.get("gpu_uuid") else f"gpu{i}" for i, g in enumerate(per_gpu)]
        samples = [int(g["samples"]) for g in per_gpu]
        ax_gpu.bar(labels, samples)
        ax_gpu.set_title(f"Per-GPU CUDA Samples ({first_vid})", fontsize=9)
        ax_gpu.set_ylabel("samples", fontsize=7)
        ax_gpu.tick_params(axis="x", rotation=45, labelsize=6)
        ax_gpu.tick_params(axis="y", labelsize=7)
    else:
        ax_gpu.axis("off")
        ax_gpu.text(0.5, 0.5, "(no per_gpu rows)", ha="center", va="center", fontsize=8)

    # Zone 4 right: top-10 cpython frames during CUDA events.
    ax_py = fig.add_subplot(gs[3, 1])
    ax_py.axis("off")
    ax_py.set_title(f"Top-10 Python frames during CUDA events ({first_vid})", fontsize=9, loc="left")
    top_py = first.get("top_python_during_cuda", [])[:10]
    if top_py:
        tbl_py = ax_py.table(
            cellText=[
                [_truncate(p["frame"], 70), f"{int(p['samples']):,}"]
                for p in top_py
            ],
            colLabels=["python frame", "samples"],
            colWidths=[0.82, 0.18],
            cellLoc="left",
            loc="upper left",
        )
        tbl_py.auto_set_font_size(False)
        tbl_py.set_fontsize(6.5)
        tbl_py.scale(1, 1.05)
    else:
        ax_py.text(0.5, 0.5, "(no python frames)", ha="center", va="center", fontsize=8)
