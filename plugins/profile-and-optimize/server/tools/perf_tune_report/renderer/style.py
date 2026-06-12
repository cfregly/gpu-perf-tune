"""Visual encoding for the perf-report renderer.

Legend semantics (mirrors the source GLM-5.1 PDF):

- ``color = palette[(hardware, tensor_parallel)]`` -- one color per
  (hardware, TP) combo. 4 distinct colors total for the typical atlas.
- ``marker = "o" if parallel_strategy == "EP" else "x"`` -- circle markers
  are EP; X markers are TP/no-EP.
- ``markerfacecolor = "none" if mtp else color`` -- hollow markers indicate
  multi-token prediction (MTP).
- ``linestyle = ":" if parallel_strategy == "TP" else "-"`` -- dotted lines
  indicate TP/no-EP; solid lines indicate EP.
- Concurrency points are labeled inline at the operator-configurable set
  ``LABEL_CONCURRENCIES`` (default: 1, 8, 32 to mirror the PDF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.perf_tune_report.schema import AtlasCell

# Operator-configurable inline-label concurrencies. Defaults to {1, 8, 32}
# to mirror the GLM-5.1 PDF; override via the CLI ``--label-concurrencies``.
LABEL_CONCURRENCIES: tuple[int, ...] = (1, 8, 32)

# Color palette keyed by (hardware, tensor_parallel). Default scheme covers
# the four hw/TP combos the GLM-5.1 PDF uses; unknown combos fall back to a
# deterministic gray so the figure still renders without crashing.
DEFAULT_PALETTE: dict[tuple[str, int], str] = {
    ("H100", 16): "#1f77b4",   # blue
    ("B200", 8): "#2ca02c",    # green
    ("GB300", 4): "#d62728",   # red
    ("GB300", 8): "#9467bd",   # purple (NVL72 8-GPU variants)
    ("B200", 4): "#ff7f0e",    # orange
    ("H100", 8): "#8c564b",    # brown
}
_FALLBACK_COLOR = "#7f7f7f"  # neutral gray


@dataclass(frozen=True)
class CellStyle:
    """Resolved style for one (hardware, TP, EP/TP, MTP) legend group."""

    color: str
    marker: str
    markerfacecolor: str
    linestyle: str
    label: str


def color_for(hardware: str, tensor_parallel: int, palette: dict | None = None) -> str:
    p = palette or DEFAULT_PALETTE
    return p.get((hardware, tensor_parallel), _FALLBACK_COLOR)


def style_for(cell: AtlasCell, palette: dict | None = None) -> CellStyle:
    """Resolve the matplotlib style for one cell's legend group."""
    color = color_for(cell.hardware, cell.tensor_parallel, palette)
    marker = "o" if cell.parallel_strategy == "EP" else "x"
    # Hollow for MTP, filled otherwise. "x" markers don't honor markerfacecolor
    # (they're stroke-only), but setting it is harmless and keeps the rule
    # uniform across marker shapes.
    markerfacecolor = "none" if cell.mtp else color
    linestyle = ":" if cell.parallel_strategy == "TP" else "-"
    return CellStyle(
        color=color,
        marker=marker,
        markerfacecolor=markerfacecolor,
        linestyle=linestyle,
        label=cell.legend_label,
    )


def legend_groups_in_order(rows: Sequence[AtlasCell]) -> list[AtlasCell]:
    """Return one representative cell per legend group, in stable display order.

    Display order rules (mirrors the PDF):

    1. Hardware order: H100, B200, GB300 (others alphabetical).
    2. Within hardware: TP descending (TP=16 before TP=8 before TP=4).
    3. Within (hw, TP): MTP variants follow their non-MTP siblings.
    4. Within (hw, TP, MTP): EP before TP.
    """
    hardware_order = {"H100": 0, "B200": 1, "GB300": 2}

    def sort_key(cell: AtlasCell) -> tuple:
        return (
            hardware_order.get(cell.hardware, 99),
            cell.hardware,
            -cell.tensor_parallel,
            int(cell.mtp),
            0 if cell.parallel_strategy == "EP" else 1,
        )

    seen: dict[tuple, AtlasCell] = {}
    for cell in rows:
        if cell.legend_key not in seen:
            seen[cell.legend_key] = cell
    return sorted(seen.values(), key=sort_key)
