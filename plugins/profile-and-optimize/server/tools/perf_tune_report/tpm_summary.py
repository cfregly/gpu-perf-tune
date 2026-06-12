"""TPM-supported-across-hardware rollup (added v1.35.0).

A pure post-processing rollup of an already-aggregated ``atlas.jsonl``: no new
cluster runs. For each distinct ``(model, hardware, quant, tensor_parallel,
parallel_strategy, mtp)`` group it picks two operating points off the measured
concurrency sweep and reports tokens-per-minute (TPM) at three capacity bases.
The output is for pricing / capacity discussions ("what TPM does each hardware
type support for this model?").

Operating points
----------------

- **peak**: the concurrency point with the highest ``output_tps_per_gpu`` --
  the warm best-case sustained capacity the sweep observed.
- **sla**: the highest-``output_tps_per_gpu`` point that still meets the
  operator-supplied latency thresholds (TTFT and/or TPOT/ITL). This is the
  customer-commitment number. When neither threshold is supplied the SLA point
  is left unset (``None``) and surfaced as "not computed" -- nothing is
  silently invented.

Capacity bases (per operating point)
------------------------------------

- **per_gpu**: ``output_tps_per_gpu * 60``
- **per_replica**: ``* tensor_parallel`` (one model replica = TP GPUs)
- **per_node**: ``* gpus_per_node`` (default 8)

Both **output-only** TPM (decode tokens) and **total** (input+output) TPM (the
OpenAI/Azure TPM convention) are reported side by side. Total-TPM is ``None``
for any group whose rows lack ``total_tps_per_gpu`` (backends that do not emit
a "Total token throughput" line); it renders as ``n/a`` downstream.

Methodology caveat (see workspace ``AGENTS.md`` "Benchmark methodology hygiene"):
the peak point is the warm sweep best-case, NOT a cold steady-state, and TPM
inherits whatever warm/cold + ISL/OSL methodology the underlying sweep used.
The summary header carries this caveat plus the campaign's ISL/OSL context so a
pricing number is never read out of context.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tools.perf_tune_report.schema import AtlasCell

SECONDS_PER_MINUTE = 60
DEFAULT_GPUS_PER_NODE = 8

# Code-defaults (v1.49.0) so EVERY campaign gets an SLA operating point + a
# $/1M-token column without needing a per-campaign tpm:/cost: block. A campaign
# config block still overrides these (and can add hardware keys, e.g. GB300).
#
# SLA defaults are an interactive-serving lens (TTFT/TPOT below); for a
# throughput-focus campaign the SLA-TPM point is a comparison lens, not a
# ship/no-ship verdict.
DEFAULT_TTFT_SLA_MS = 2000.0
DEFAULT_TPOT_SLA_MS = 50.0
# {hardware: $/GPU-hour}. H100/H200/B200 = ASSUMED public on-demand list rate,
# per-GPU of the 8-GPU HGX node (representative GPU-cloud list pricing,
# 2026-06): HGX H100 $49.24/hr=$6.16/GPU, HGX H200 $50.44/hr=$6.31/GPU,
# HGX B200 $68.80/hr=$8.60/GPU.
# GB300 NVL72 has NO public list rate -> $12.00 is an ASSUMED ESTIMATE
# (~1.4x B200), a placeholder to replace when a real rate exists. All are LIST/
# ESTIMATE prices, not contract cost; override per-campaign.
DEFAULT_USD_PER_GPU_HOUR: dict[str, float] = {
    "H100": 6.16, "H200": 6.31, "B200": 8.60, "GB300": 12.00,
}
DEFAULT_COST_RATE_SOURCE = (
    "assumed: H100/H200/B200 = representative public on-demand list rate per-GPU "
    "of the 8-GPU HGX node (2026-06); GB300 = assumed "
    "ESTIMATE (~1.4x B200, no public list rate, placeholder). List/estimate "
    "prices, not contract cost; override per-campaign via the config "
    "cost.usd_per_gpu_hour block"
)

OPERATING_POINT_PEAK = "peak"
OPERATING_POINT_SLA = "sla"

BASES = ("per_gpu", "per_replica", "per_node")

_METHODOLOGY_CAVEAT = (
    "TPM = tokens/minute (tok/s * 60). 'peak' is the warm sweep best-case "
    "(max output tok/s/GPU observed), NOT a cold steady-state; 'sla' is the "
    "highest-throughput point still meeting the latency thresholds. TPM "
    "inherits the underlying sweep's warm/cold + ISL/OSL methodology -- read "
    "with the data-source context below."
)


@dataclass(frozen=True)
class TpmConfig:
    """Campaign-level TPM knobs, declared once in the campaign ``config.yaml``
    ``tpm:`` block and read by all three surfaces (the ``tpm_summary`` verb, the
    ``report_render`` PDF page, and ``publish_to_lake``'s ``tpm_v1`` table) so
    the peak AND sla operating points stay consistent everywhere.

    All fields optional: no ``tpm:`` block -> peak-only at the default node size.
    """

    ttft_sla_ms: float | None = DEFAULT_TTFT_SLA_MS
    tpot_sla_ms: float | None = DEFAULT_TPOT_SLA_MS
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE
    # {hardware: $/GPU-hour}. Defaults to the assumed public-list table
    # (DEFAULT_USD_PER_GPU_HOUR); a campaign ``cost:`` block is overlaid on top
    # (so a campaign can override a rate or add a hardware key like GB300).
    usd_per_gpu_hour: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_USD_PER_GPU_HOUR)
    )
    # Provenance for the cost rates in use (informational; surfaced in the
    # summary, NOT a lake column). Set to the campaign config when a cost: block
    # overrides the defaults.
    cost_rate_source: str = DEFAULT_COST_RATE_SOURCE


def _find_cost_yaml(campaign_dir: Path) -> Path | None:
    """Walk up from the campaign dir for ``perf-tune-report/configs/cost.yaml`` (the same
    walk-up ``sol-ceilings.yaml`` uses), so a published campaign picks up the fleet
    cost rates without a per-campaign ``cost:`` block. Best-effort; None if absent."""
    relpath = Path("perf-tune-report") / "configs" / "cost.yaml"
    try:
        cur = campaign_dir.resolve()
    except OSError:
        return None
    for parent in [cur, *cur.parents]:
        candidate = parent / relpath
        if candidate.is_file():
            return candidate
    return None


def discover_tpm_config(campaign_dir: Path) -> TpmConfig:
    """Read the ``tpm:`` and ``cost:`` blocks from ``<campaign_dir>/config.yaml``.

    Code-defaults (v1.49.0): when a field/block is absent the campaign still
    gets the default SLA (TTFT<=2000ms, TPOT<=50ms), ``gpus_per_node=8``, and
    the assumed public-list cost table -- so SLA-TPM + $/1M populate for every
    campaign. A per-campaign ``tpm:``/``cost:`` block overrides per field
    (cost is overlaid, so a block can override one rate or add a hardware key).
    Never raises (degrades to the defaults, matching the loud-skip pattern).
    """
    cfg_path = campaign_dir / "config.yaml"
    if not cfg_path.is_file():
        return TpmConfig()
    try:
        import yaml  # lazy: keep the module import light for the publish path

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - config is best-effort; degrade to defaults
        return TpmConfig()
    if not isinstance(data, dict):
        return TpmConfig()
    block = data.get("tpm")
    block = block if isinstance(block, dict) else {}

    def _num(key: str, default: float) -> float:
        val = block.get(key)
        return float(val) if isinstance(val, (int, float)) and val > 0 else default

    gpn = block.get("gpus_per_node")

    # cost: { usd_per_gpu_hour: { B200: 4.5, ... } } -- overlaid on the defaults.
    # Precedence: campaign cost: block > perf-tune-report/configs/cost.yaml > DEFAULT table.
    cost_block = data.get("cost")
    usd_map: dict[str, float] = dict(DEFAULT_USD_PER_GPU_HOUR)
    source = DEFAULT_COST_RATE_SOURCE
    # Fleet cost.yaml overlay (under any per-campaign block): published cost_v1 uses the
    # fleet rates without a per-campaign cost: block. `default` is a fleet_leaderboard
    # fallback key, not a tpm hardware -- skip it (it never matches an atlas hardware).
    cost_yaml = _find_cost_yaml(campaign_dir)
    if cost_yaml is not None:
        try:
            cy = yaml.safe_load(cost_yaml.read_text(encoding="utf-8")) or {}
            cy_map = cy.get("usd_per_gpu_hour") if isinstance(cy, dict) else None
            if isinstance(cy_map, dict):
                fleet = {
                    str(hw): float(v)
                    for hw, v in cy_map.items()
                    if hw != "default" and isinstance(v, (int, float)) and v > 0
                }
                if fleet:
                    usd_map.update(fleet)
                    source = f"perf-tune-report/configs/cost.yaml (rates {sorted(fleet)})"
        except Exception:  # noqa: BLE001 - cost.yaml is best-effort; degrade to defaults
            pass
    if isinstance(cost_block, dict):
        raw_map = cost_block.get("usd_per_gpu_hour")
        if isinstance(raw_map, dict):
            overrides = {
                str(hw): float(v)
                for hw, v in raw_map.items()
                if isinstance(v, (int, float)) and v > 0
            }
            if overrides:
                usd_map.update(overrides)
                source = (
                    f"campaign config cost: block (overrides {sorted(overrides)}) "
                    f"on top of {source}"
                )

    return TpmConfig(
        ttft_sla_ms=_num("ttft_sla_ms", DEFAULT_TTFT_SLA_MS),
        tpot_sla_ms=_num("tpot_sla_ms", DEFAULT_TPOT_SLA_MS),
        gpus_per_node=int(gpn) if isinstance(gpn, int) and gpn > 0 else DEFAULT_GPUS_PER_NODE,
        usd_per_gpu_hour=usd_map,
        cost_rate_source=source,
    )


@dataclass(frozen=True)
class TpmPoint:
    """One operating point's TPM at all three capacity bases."""

    operating_point: str  # "peak" | "sla"
    concurrency: int
    output_tps_per_gpu: float
    total_tps_per_gpu: float | None
    output_tpm_per_gpu: float
    output_tpm_per_replica: float
    output_tpm_per_node: float
    total_tpm_per_gpu: float | None
    total_tpm_per_replica: float | None
    total_tpm_per_node: float | None
    # The latency observed at this point (for transparency in the table).
    ttft_avg_ms: float | None = None
    tpot_median_ms: float | None = None
    itl_avg_ms: float | None = None
    # cell_id of the originating atlas row (used to join the cell's DCGM power
    # for tokens-per-watt in the economics/cost_v1 table).
    cell_id: str = ""
    # $/1M tokens (added v1.42.0): populated when a cost: config block supplies
    # the hardware's $/GPU-hour; None otherwise. Basis-independent (per-GPU
    # normalizes out GPU count).
    usd_per_1m_output_tokens: float | None = None
    usd_per_1m_total_tokens: float | None = None
    usd_per_gpu_hour: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TpmGroup:
    """TPM capacity for one (model, hardware, quant, TP, strategy, MTP) group."""

    model: str
    hardware: str
    quant: str
    tensor_parallel: int
    parallel_strategy: str
    mtp: bool
    gpus_per_node: int
    peak: TpmPoint | None
    sla: TpmPoint | None
    # Mean ISL/OSL (shape) + warm/cold label for this group (constant across the
    # group's concurrency points; from the originating atlas rows).
    mean_isl: float | None = None
    mean_osl: float | None = None
    cache_mode: str = "unknown"

    @property
    def legend_label(self) -> str:
        mtp_suffix = " MTP" if self.mtp else ""
        return (
            f"{self.hardware} {self.quant}{mtp_suffix} "
            f"TP={self.tensor_parallel} {self.parallel_strategy}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["peak"] = self.peak.to_dict() if self.peak else None
        d["sla"] = self.sla.to_dict() if self.sla else None
        return d


@dataclass
class TpmSummary:
    """The full TPM-across-hardware rollup for one campaign."""

    groups: list[TpmGroup] = field(default_factory=list)
    ttft_sla_ms: float | None = None
    tpot_sla_ms: float | None = None
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE
    context_line: str | None = None
    # Provenance for the $/GPU-hour rates behind the $/1M-token columns
    # (informational; None when no cost rates were applied).
    cost_rate_source: str | None = None

    @property
    def sla_computed(self) -> bool:
        """True iff at least one SLA threshold was supplied."""
        return self.ttft_sla_ms is not None or self.tpot_sla_ms is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tpm_summary_v1",
            "ttft_sla_ms": self.ttft_sla_ms,
            "tpot_sla_ms": self.tpot_sla_ms,
            "gpus_per_node": self.gpus_per_node,
            "sla_computed": self.sla_computed,
            "context_line": self.context_line,
            "cost_rate_source": self.cost_rate_source,
            "methodology_caveat": _METHODOLOGY_CAVEAT,
            "groups": [g.to_dict() for g in self.groups],
        }

    # --- serializers -------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_csv(self) -> str:
        """One row per (group, operating_point, basis). Flat for spreadsheets."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "model", "hardware", "quant", "tensor_parallel",
                "parallel_strategy", "mtp", "operating_point", "basis",
                "concurrency", "output_tpm", "total_tpm",
                "ttft_avg_ms", "tpot_median_ms", "itl_avg_ms",
                "mean_isl", "mean_osl", "cache_mode",
                "usd_per_1m_output_tokens", "usd_per_1m_total_tokens",
            ]
        )
        for g in self.groups:
            for point in (g.peak, g.sla):
                if point is None:
                    continue
                for basis in BASES:
                    writer.writerow(
                        [
                            g.model, g.hardware, g.quant, g.tensor_parallel,
                            g.parallel_strategy, g.mtp, point.operating_point,
                            basis, point.concurrency,
                            _fmt_num(getattr(point, f"output_tpm_{basis}")),
                            _fmt_num(getattr(point, f"total_tpm_{basis}")),
                            _fmt_num(point.ttft_avg_ms),
                            _fmt_num(point.tpot_median_ms),
                            _fmt_num(point.itl_avg_ms),
                            _fmt_num(g.mean_isl), _fmt_num(g.mean_osl), g.cache_mode,
                            _fmt_num(point.usd_per_1m_output_tokens),
                            _fmt_num(point.usd_per_1m_total_tokens),
                        ]
                    )
        return buf.getvalue()

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# TPM supported across hardware types")
        lines.append("")
        lines.append(f"_{_METHODOLOGY_CAVEAT}_")
        lines.append("")
        if self.context_line:
            lines.append(f"- Data source / shape: {self.context_line}")
        lines.append(f"- gpus_per_node basis: {self.gpus_per_node}")
        if self.sla_computed:
            lines.append(
                f"- SLA thresholds: TTFT <= {self.ttft_sla_ms} ms, "
                f"TPOT/ITL <= {self.tpot_sla_ms} ms"
            )
        else:
            lines.append(
                "- SLA thresholds: not set -> SLA columns not computed "
                "(pass --ttft-sla-ms / --tpot-sla-ms for a customer-commitment number)"
            )
        if self.cost_rate_source:
            lines.append(f"- Cost rate source: {self.cost_rate_source}")
        lines.append("")

        if not self.groups:
            lines.append("_No throughput-bearing atlas rows; nothing to roll up._")
            lines.append("")
            return "\n".join(lines)

        header = (
            "| Variant | Point | Conc | Output TPM/GPU | Output TPM/replica "
            "| Output TPM/node | Total TPM/GPU | Total TPM/replica | Total TPM/node "
            "| $/1M out | $/1M total |"
        )
        sep = "| " + " | ".join(["---"] * 11) + " |"
        # One sub-table per hardware so the pricing reader scans by hardware.
        hardwares = _ordered_unique([g.hardware for g in self.groups])
        for hw in hardwares:
            lines.append(f"## {hw}")
            lines.append("")
            # Per-hardware shape + cache caption (constant within a variant).
            shape_bits = []
            isl = next((g.mean_isl for g in self.groups if g.hardware == hw and g.mean_isl), None)
            osl = next((g.mean_osl for g in self.groups if g.hardware == hw and g.mean_osl), None)
            cmode = next((g.cache_mode for g in self.groups if g.hardware == hw), "unknown")
            if isl is not None:
                shape_bits.append(f"mean ISL ~= {isl:.0f}")
            if osl is not None:
                shape_bits.append(f"mean OSL ~= {osl:.0f}")
            shape_bits.append(f"cache: {cmode}")
            lines.append("_" + " | ".join(shape_bits) + "_")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for g in [grp for grp in self.groups if grp.hardware == hw]:
                if g.peak is not None:
                    lines.append(_md_point_row(g.legend_label, g.peak))
                if g.sla is not None:
                    lines.append(_md_point_row(g.legend_label, g.sla))
                elif self.sla_computed:
                    # SLA thresholds were set but no sweep point met them:
                    # show an explicit row so the gap is never silently absent.
                    lines.append(
                        f"| {g.legend_label} | sla | - | no point met SLA "
                        "| - | - | - | - | - | - | - |"
                    )
            lines.append("")
        return "\n".join(lines)


def _fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _md_point_row(label: str, point: TpmPoint) -> str:
    return (
        f"| {label} | {point.operating_point} | {point.concurrency} "
        f"| {_fmt_tpm(point.output_tpm_per_gpu)} "
        f"| {_fmt_tpm(point.output_tpm_per_replica)} "
        f"| {_fmt_tpm(point.output_tpm_per_node)} "
        f"| {_fmt_tpm(point.total_tpm_per_gpu)} "
        f"| {_fmt_tpm(point.total_tpm_per_replica)} "
        f"| {_fmt_tpm(point.total_tpm_per_node)} "
        f"| {_fmt_usd(point.usd_per_1m_output_tokens)} "
        f"| {_fmt_usd(point.usd_per_1m_total_tokens)} |"
    )


def _fmt_num(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def _fmt_tpm(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _ordered_unique(items: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        seen.setdefault(it, None)
    return list(seen.keys())


def _meets_sla(
    row: AtlasCell, ttft_sla_ms: float | None, tpot_sla_ms: float | None
) -> bool:
    """True iff the row satisfies every supplied latency threshold.

    A threshold of ``None`` is not applied. When a threshold is applied the
    corresponding metric must be present (a row with no TTFT cannot be
    certified against a TTFT SLA)."""
    if ttft_sla_ms is not None:
        if row.ttft_avg_ms is None or row.ttft_avg_ms > ttft_sla_ms:
            return False
    if tpot_sla_ms is not None:
        decode_ms = row.tpot_median_ms if row.tpot_median_ms is not None else row.itl_avg_ms
        if decode_ms is None or decode_ms > tpot_sla_ms:
            return False
    return True


def _point_from_row(
    row: AtlasCell,
    operating_point: str,
    tensor_parallel: int,
    gpus_per_node: int,
    usd_per_gpu_hour: float | None = None,
) -> TpmPoint:
    o_gpu = row.output_tps_per_gpu * SECONDS_PER_MINUTE
    t_gpu = (
        row.total_tps_per_gpu * SECONDS_PER_MINUTE
        if row.total_tps_per_gpu is not None
        else None
    )
    # $/1M tokens (basis-independent: cost and tokens both scale with GPU count,
    # so the per-GPU ratio is the $/token for any basis).
    usd_out = usd_tot = None
    if usd_per_gpu_hour is not None and row.output_tps_per_gpu > 0:
        usd_out = usd_per_gpu_hour * 1e6 / (row.output_tps_per_gpu * 3600.0)
        if row.total_tps_per_gpu and row.total_tps_per_gpu > 0:
            usd_tot = usd_per_gpu_hour * 1e6 / (row.total_tps_per_gpu * 3600.0)
    return TpmPoint(
        operating_point=operating_point,
        concurrency=row.concurrency,
        output_tps_per_gpu=row.output_tps_per_gpu,
        total_tps_per_gpu=row.total_tps_per_gpu,
        output_tpm_per_gpu=o_gpu,
        output_tpm_per_replica=o_gpu * tensor_parallel,
        output_tpm_per_node=o_gpu * gpus_per_node,
        total_tpm_per_gpu=t_gpu,
        total_tpm_per_replica=(t_gpu * tensor_parallel) if t_gpu is not None else None,
        total_tpm_per_node=(t_gpu * gpus_per_node) if t_gpu is not None else None,
        ttft_avg_ms=row.ttft_avg_ms,
        tpot_median_ms=row.tpot_median_ms,
        itl_avg_ms=row.itl_avg_ms,
        cell_id=row.cell_id,
        usd_per_1m_output_tokens=usd_out,
        usd_per_1m_total_tokens=usd_tot,
        usd_per_gpu_hour=usd_per_gpu_hour,
    )


def _group_attr(group_rows: Sequence[AtlasCell], attr: str) -> float | None:
    """The single non-None value of a per-group-constant numeric attr (e.g.
    mean_isl/mean_osl), or the mean if rows disagree. None when all absent."""
    vals = [getattr(r, attr) for r in group_rows if getattr(r, attr) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_tpm_summary(
    rows: Sequence[AtlasCell],
    *,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    context_line: str | None = None,
    usd_per_gpu_hour: dict[str, float] | None = None,
    cost_rate_source: str | None = None,
) -> TpmSummary:
    """Roll an atlas up into per-hardware TPM capacity points.

    Groups rows by ``(model,) + legend_key`` and, for each group with at least
    one throughput-bearing row (``output_tps_per_gpu`` present), picks the peak
    point and -- when an SLA threshold is supplied -- the SLA-bounded point.

    ``usd_per_gpu_hour`` is an optional ``{hardware: $/GPU-hour}`` map (from the
    campaign config ``cost:`` block); when a group's hardware is present, each
    point carries the derived ``$/1M tokens``.
    """
    # Group by (model, hardware, quant, TP, strategy, MTP). The legend_key
    # already keys on the last five; prepend model.
    groups: dict[tuple, list[AtlasCell]] = {}
    for row in rows:
        if row.output_tps_per_gpu is None:
            continue
        key = (row.model,) + row.legend_key
        groups.setdefault(key, []).append(row)

    out_groups: list[TpmGroup] = []
    sla_active = ttft_sla_ms is not None or tpot_sla_ms is not None
    cost_map = usd_per_gpu_hour or {}

    # Loud warning when a cost: block names hardware that matches no atlas row
    # (e.g. a `b200` vs `B200` typo) -- otherwise $/1M-token would be silently
    # null for every group. key[1] is the hardware (key = (model,) + legend_key).
    # DEFAULT_USD_PER_GPU_HOUR keys are excluded: the shipped default table
    # always carries H100/H200/B200, and a single-hardware campaign legitimately
    # matches only one of them -- warning on the others would be noise on every
    # publish. Only a NON-default (operator-supplied) unmatched key is a typo.
    if cost_map:
        present_hw = {key[1] for key in groups}
        unmatched = [
            hw
            for hw in cost_map
            if hw not in present_hw and hw not in DEFAULT_USD_PER_GPU_HOUR
        ]
        if unmatched:
            print(
                f"WARNING: cost: usd_per_gpu_hour has hardware key(s) {unmatched} "
                f"that match no atlas hardware {sorted(present_hw)} -- $/1M-token "
                "will be null for those (check for a case/spelling mismatch, e.g. "
                "'b200' vs 'B200').",
                file=sys.stderr,
            )
    for key, group_rows in groups.items():
        peak_row = max(group_rows, key=lambda r: r.output_tps_per_gpu)
        tp = peak_row.tensor_parallel
        usd_gpu_hr = cost_map.get(peak_row.hardware)
        peak = _point_from_row(
            peak_row, OPERATING_POINT_PEAK, tp, gpus_per_node, usd_gpu_hr
        )

        sla_point: TpmPoint | None = None
        if sla_active:
            sla_candidates = [
                r for r in group_rows if _meets_sla(r, ttft_sla_ms, tpot_sla_ms)
            ]
            if sla_candidates:
                sla_row = max(sla_candidates, key=lambda r: r.output_tps_per_gpu)
                sla_point = _point_from_row(
                    sla_row, OPERATING_POINT_SLA, tp, gpus_per_node, usd_gpu_hr
                )

        out_groups.append(
            TpmGroup(
                model=peak_row.model,
                hardware=peak_row.hardware,
                quant=peak_row.quant,
                tensor_parallel=tp,
                parallel_strategy=peak_row.parallel_strategy,
                mtp=peak_row.mtp,
                gpus_per_node=gpus_per_node,
                peak=peak,
                sla=sla_point,
                mean_isl=_group_attr(group_rows, "mean_input_tokens"),
                mean_osl=_group_attr(group_rows, "mean_output_tokens"),
                cache_mode=peak_row.cache_mode,
            )
        )

    # Stable display order: hardware (H100, B200, GB300, others), then TP desc,
    # then MTP, then EP-before-TP, then model -- mirrors style.legend ordering.
    hardware_order = {"H100": 0, "B200": 1, "GB300": 2}
    out_groups.sort(
        key=lambda g: (
            hardware_order.get(g.hardware, 99),
            g.hardware,
            -g.tensor_parallel,
            int(g.mtp),
            0 if g.parallel_strategy == "EP" else 1,
            g.model,
        )
    )

    return TpmSummary(
        groups=out_groups,
        ttft_sla_ms=ttft_sla_ms,
        tpot_sla_ms=tpot_sla_ms,
        gpus_per_node=gpus_per_node,
        context_line=context_line,
        cost_rate_source=cost_rate_source if cost_map else None,
    )
