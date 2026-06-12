"""champion_select: baseline vs top-X cross-engine champion selection.

The unified "it must be obvious what to push to production" synthesis. Reads a
perf-report campaign (``atlas.jsonl`` + per-cell SoL artifacts) and:

1. groups the atlas into ONE row per variant arm (``cell_id``) at the focus
   concurrency, cross-engine (vLLM + SGLang are both first-class via
   ``serving_engine``);
2. ranks the variants under the run's focus metric (throughput tok/s/GPU OR
   median TPOT) subject to a TPOT SLO ceiling, and selects the **baseline + the
   top-X** (default 3);
3. summarizes each selected variant across the **4-layer SoL ladder** (L1
   zymtrace ``kernels.json`` -> L2 ``dcgm_correlation.json``
   ``per_category_attribution`` -> L3 ``dcgm_correlation.json`` ``resources`` /
   ``roofline_sweep.json`` -> L4 ``ncu_kernels.json``) and gathers the roofline
   operating points so a single overlay can show baseline + all X variants;
4. computes a **production recommendation** tiered DRAFT vs VERDICT, where a
   VERDICT requires the variance (same-node + >=3 trials), multi-workload, and
   accuracy gates to pass and the champion to be DCGM byte-grounded (L3);
5. writes the standalone ``CHAMPION.md`` + ``champion_select.json`` (the latter
   is also the artifact the renderer's champion-selection page consumes and the
   ``champion`` rows the perf-lake records).

This is a PURE post-processing verb (no cluster runs): the gates are measured by
the bench/eval skills and passed in; this module is the synthesis + the
explainable, data-backed decision. Sibling of ``tpm_summary`` / ``value_view``.

Added in profile-and-optimize v1.66.0.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import AtlasCell, read_jsonl

SCHEMA_VERSION = "champion_select_v1"

#: The canonical multi-workload suite a throughput/latency ship VERDICT must
#: cover (the MTP-K single-workload failure this gate exists to prevent). See
#: ``docs/METHODOLOGY.md``.
CANONICAL_WORKLOADS = ("aa", "sonnet", "sharegpt", "random", "code")

_ROOFLINE_SUFFIXES = ("-decode", "-prefill")


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class SolSummary:
    """The 4-layer Speed-of-Light footprint of one variant, best-effort from the
    per-cell artifacts. Every field is nullable so a partially-captured variant
    still summarizes (the absence is the signal)."""

    l1_present: bool = False
    l1_top_categories: list[str] = field(default_factory=list)
    l2_present: bool = False
    l3_present: bool = False
    hbm_pct_sol: float | None = None
    tensor_pct_sol: float | None = None
    sm_active_pct: float | None = None
    nvlink_pct_sol: float | None = None
    l4_present: bool = False
    l4_kernel_count: int | None = None
    sol_rigor: str = "none"  # none < L1 < L2 < L3 < L4 (highest present)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VariantRow:
    cell_id: str
    engine: str
    is_baseline: bool
    focus_metric: float | None  # the ranked metric value (per-GPU tok/s OR TPOT ms)
    output_tps_per_gpu: float | None
    tpot_median_ms: float | None
    pct_win_vs_baseline: float | None  # +% better than baseline on the focus metric
    slo_verdict: str  # PASS-SLO | SLO-FAIL | NO-DATA | BASELINE
    sol: SolSummary
    has_roofline: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sol"] = self.sol.to_dict()
        return d


@dataclass
class GateResult:
    name: str
    status: str  # pass | fail | unknown
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChampionResult:
    schema_version: str
    campaign_id: str
    focus: str
    focus_c: int
    metric: str  # "tok_s_gpu" | "tpot"
    slo_ms: float | None
    hardware: str
    tensor_parallel: int
    baseline_cell: str | None
    variants: list[VariantRow]  # baseline first, then ranked top-X
    recommended_cell: str | None
    recommended_engine: str | None
    tier: str  # draft | verdict
    gates: list[GateResult]
    reasons: list[str]
    roofline_overlay: dict[str, Any]  # {cell_id: roofline_sweep.json payload}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "focus": self.focus,
            "focus_c": self.focus_c,
            "metric": self.metric,
            "slo_ms": self.slo_ms,
            "hardware": self.hardware,
            "tensor_parallel": self.tensor_parallel,
            "baseline_cell": self.baseline_cell,
            "variants": [v.to_dict() for v in self.variants],
            "recommended_cell": self.recommended_cell,
            "recommended_engine": self.recommended_engine,
            "tier": self.tier,
            "gates": [g.to_dict() for g in self.gates],
            "reasons": self.reasons,
            "roofline_overlay": self.roofline_overlay,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_roofline_cell(cell_id: str, row: AtlasCell) -> bool:
    if any(cell_id.endswith(s) for s in _ROOFLINE_SUFFIXES):
        return True
    phase = (row.extra or {}).get("phase")
    return phase in ("decode", "prefill")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _artifact_cell_id(campaign_dir: Path, row: AtlasCell | None, cell_id: str) -> str:
    """Resolve the cells/<id>/ directory that carries per-cell artifacts.

    Most campaigns use the atlas ``cell_id`` directly. Legacy variant-A/B imports
    can have logical rows suffixed with ``-Kengine`` while the artifact directory
    remains the physical arm name in ``extra["arm"]``.
    """

    arm = (row.extra or {}).get("arm") if row is not None else None
    if isinstance(arm, str) and arm and (
        (campaign_dir / "cells" / arm).is_dir()
        or (campaign_dir / "cells" / f"{arm}-decode").is_dir()
        or (campaign_dir / "cells" / f"{arm}-prefill").is_dir()
    ):
        return arm
    return cell_id


def _classify_resource(res: dict[str, Any]) -> str | None:
    """Bucket a dcgm_correlation resource into hbm / tensor / nvlink by its
    peak_key / metric (the L3 resource naming, see dcgm_correlate)."""
    key = f"{res.get('peak_key', '')} {res.get('metric', '')}".lower()
    if "hbm" in key or "dram" in key:
        return "hbm"
    if "nvlink" in key:
        return "nvlink"
    if "pflops" in key or "dense" in key or "tensor" in key or "fp16" in key or "fp8" in key:
        return "tensor"
    return None


def _summarize_sol(campaign_dir: Path, cell_id: str, focus_c: int) -> SolSummary:
    cell = campaign_dir / "cells" / cell_id
    s = SolSummary()

    # L1: zymtrace / nsys kernels.json (per_category sample/time share).
    kj = _load_json(cell / "kernels.json")
    if kj and isinstance(kj.get("per_category"), dict) and kj["per_category"]:
        s.l1_present = True
        cats = sorted(kj["per_category"].items(), key=lambda kv: -(kv[1] or 0))
        s.l1_top_categories = [c for c, _ in cats[:4]]

    # L2 + L3: dcgm_correlation.json (per_category_attribution + resources).
    dc = _load_json(cell / "dcgm_correlation.json")
    if dc:
        if dc.get("per_category_attribution"):
            s.l2_present = True
        resources = dc.get("resources") or []
        if resources:
            s.l3_present = True
            for res in resources:
                bucket = _classify_resource(res)
                pct = res.get("sol_pct")
                if pct is None:
                    continue
                if bucket == "hbm" and s.hbm_pct_sol is None:
                    s.hbm_pct_sol = float(pct)
                elif bucket == "tensor" and s.tensor_pct_sol is None:
                    s.tensor_pct_sol = float(pct)
                elif bucket == "nvlink" and s.nvlink_pct_sol is None:
                    s.nvlink_pct_sol = float(pct)

    # L3 (alt): roofline_sweep.json carries per-(c) DCGM active fractions; use the
    # decode point at the focus concurrency to fill any gap left by dcgm_correlate.
    rf = _load_json(cell / "roofline_sweep.json") or _load_json(
        campaign_dir / "cells" / f"{cell_id}-decode" / "roofline_sweep.json"
    )
    if rf and rf.get("decode"):
        pts = rf["decode"]
        pt = next((p for p in pts if p.get("c") == focus_c), None) or pts[-1]
        if pt:
            s.l3_present = True
            if s.hbm_pct_sol is None and pt.get("dram_active") is not None:
                s.hbm_pct_sol = round(float(pt["dram_active"]) * 100, 1)
            if s.tensor_pct_sol is None and pt.get("tensor_active") is not None:
                s.tensor_pct_sol = round(float(pt["tensor_active"]) * 100, 1)
            if s.sm_active_pct is None and pt.get("sm_active") is not None:
                s.sm_active_pct = round(float(pt["sm_active"]) * 100, 1)

    # L4: ncu_kernels.json (per-kernel arithmetic intensity / %SoL).
    nk = _load_json(cell / "ncu_kernels.json")
    if nk and nk.get("kernels"):
        s.l4_present = True
        s.l4_kernel_count = len(nk["kernels"])

    s.sol_rigor = (
        "L4" if s.l4_present
        else "L3" if s.l3_present
        else "L2" if s.l2_present
        else "L1" if s.l1_present
        else "none"
    )
    return s


def _load_roofline(campaign_dir: Path, cell_id: str) -> dict[str, Any] | None:
    return _load_json(campaign_dir / "cells" / cell_id / "roofline_sweep.json") or _load_json(
        campaign_dir / "cells" / f"{cell_id}-decode" / "roofline_sweep.json"
    )


def _resolve_baseline(arm_ids: list[str], rows_by_cell: dict[str, dict[int, AtlasCell]],
                      explicit: str | None) -> str | None:
    if explicit and explicit in rows_by_cell:
        return explicit
    # heuristic: a *-base / *-baseline cell, preferring vLLM (the incumbent).
    bases = [c for c in arm_ids if c.endswith(("-base", "-baseline"))]
    if bases:
        vllm_bases = [
            c for c in bases
            if any(r.serving_engine != "sglang" for r in rows_by_cell[c].values())
        ]
        return sorted(vllm_bases or bases)[0]
    return sorted(arm_ids)[0] if arm_ids else None


def _campaign_focus(campaign_dir: Path) -> str | None:
    cfg = _load_json(campaign_dir / "config.json")
    if cfg and cfg.get("focus"):
        return str(cfg["focus"])
    # config.yaml fallback (campaign_init writes config.yaml).
    yml = campaign_dir / "config.yaml"
    if yml.is_file():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(yml.read_text()) or {}
        except Exception:
            data = {}
        focus = (data.get("campaign") or {}).get("focus") or data.get("focus")
        if focus:
            return str(focus)
    return None


def _local_eval_acc(rows: list[AtlasCell]) -> dict[str, float]:
    """Serving-quality metrics from the campaign's local ``eval_acc`` cells
    (``import_model_eval``), so the variant decision can see quality without the lake."""
    out: dict[str, float] = {}
    for r in rows:
        extra = getattr(r, "extra", None) or {}
        if extra.get("metric_kind") != "eval_acc":
            continue
        qm = extra.get("quality_metrics")
        if isinstance(qm, dict):
            for k, v in qm.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[str(k)] = float(v)
    return out


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def select(
    campaign_dir: Path,
    *,
    focus: str | None = None,
    focus_c: int | None = None,
    top: int = 3,
    baseline: str | None = None,
    metric: str | None = None,
    slo_rel: float = 1.10,
    slo_abs_ms: float | None = None,
    trials: int | None = None,
    same_node: bool = False,
    require_workloads: tuple[str, ...] = CANONICAL_WORKLOADS,
    workloads_present: tuple[str, ...] | None = None,
    accuracy_gate: str = "unknown",
    accuracy_floor: float | None = None,
) -> ChampionResult:
    atlas_path = campaign_dir / "atlas.jsonl"
    if not atlas_path.is_file():
        raise FileNotFoundError(
            f"atlas.jsonl not found at {atlas_path}; run `perftunereport atlas_aggregate` first."
        )
    rows = read_jsonl(atlas_path)
    if not rows:
        raise ValueError(f"atlas.jsonl is empty at {atlas_path}")

    # Quality coupling: read the campaign's local eval_acc cells (import_model_eval). When the
    # accuracy gate was not supplied explicitly but a --accuracy-floor is, derive pass/fail from
    # the measured quality (conservative: gate on the WORST measured metric).
    local_eval_acc = _local_eval_acc(rows)
    if accuracy_gate == "unknown" and local_eval_acc and accuracy_floor is not None:
        accuracy_gate = "pass" if min(local_eval_acc.values()) >= accuracy_floor else "fail"

    resolved_focus = (focus or _campaign_focus(campaign_dir) or "throughput").lower()
    resolved_metric = metric or ("tpot" if resolved_focus == "latency" else "tok_s_gpu")
    resolved_focus_c = focus_c if focus_c is not None else (1 if resolved_focus == "latency" else 32)

    # Group atlas rows: one {concurrency: AtlasCell} per ARM cell (drop roofline cells).
    rows_by_cell: dict[str, dict[int, AtlasCell]] = {}
    for r in rows:
        if _is_roofline_cell(r.cell_id, r):
            continue
        rows_by_cell.setdefault(r.cell_id, {})[r.concurrency] = r
    arm_ids = sorted(rows_by_cell)
    if not arm_ids:
        raise ValueError(
            f"no rankable arm cells in {atlas_path} (only roofline cells found)"
        )

    hardware = rows[0].hardware
    tensor_parallel = rows[0].tensor_parallel

    baseline_cell = _resolve_baseline(arm_ids, rows_by_cell, baseline)

    def _row_at_c(cell_id: str) -> AtlasCell | None:
        by_c = rows_by_cell[cell_id]
        return by_c.get(resolved_focus_c) or (by_c[max(by_c)] if by_c else None)

    def _metric_value(cell: AtlasCell | None) -> float | None:
        if cell is None:
            return None
        if resolved_metric == "tpot":
            return cell.tpot_median_ms
        return cell.output_tps_per_gpu

    base_row = _row_at_c(baseline_cell) if baseline_cell else None
    base_tpot = base_row.tpot_median_ms if base_row else None
    base_metric = _metric_value(base_row)
    slo_ms = (
        slo_abs_ms if slo_abs_ms is not None
        else (base_tpot * slo_rel if base_tpot is not None else None)
    )

    def _slo_verdict(cell: AtlasCell | None, is_base: bool) -> str:
        if is_base:
            return "BASELINE"
        if cell is None or _metric_value(cell) is None:
            return "NO-DATA"
        if slo_ms is None or resolved_metric == "tpot":
            return "PASS-SLO"  # latency focus self-gates on TPOT
        return "PASS-SLO" if (cell.tpot_median_ms or 0) <= slo_ms else "SLO-FAIL"

    def _pct_win(val: float | None) -> float | None:
        if val is None or base_metric in (None, 0):
            return None
        if resolved_metric == "tpot":  # lower is better
            return round((base_metric / val - 1) * 100, 1) if val else None
        return round((val / base_metric - 1) * 100, 1)

    def _mk_variant(cell_id: str) -> VariantRow:
        is_base = cell_id == baseline_cell
        cell = _row_at_c(cell_id)
        engine = (cell.serving_engine if cell else "") or "vllm"
        val = _metric_value(cell)
        return VariantRow(
            cell_id=cell_id,
            engine=engine,
            is_baseline=is_base,
            focus_metric=val,
            output_tps_per_gpu=cell.output_tps_per_gpu if cell else None,
            tpot_median_ms=cell.tpot_median_ms if cell else None,
            pct_win_vs_baseline=0.0 if is_base else _pct_win(val),
            slo_verdict=_slo_verdict(cell, is_base),
            sol=_summarize_sol(
                campaign_dir, _artifact_cell_id(campaign_dir, cell, cell_id), resolved_focus_c
            ),
            has_roofline=_load_roofline(
                campaign_dir, _artifact_cell_id(campaign_dir, cell, cell_id)
            ) is not None,
        )

    # Rank the non-baseline arms: SLO-passing first, then by the focus metric
    # (tok/s/GPU desc, or TPOT asc).
    candidates = [c for c in arm_ids if c != baseline_cell]
    cand_rows = [_mk_variant(c) for c in candidates]

    def _sort_key(v: VariantRow):
        slo_rank = 0 if v.slo_verdict == "PASS-SLO" else 1
        if v.focus_metric is None:
            return (slo_rank, 1, 0.0)
        primary = v.focus_metric if resolved_metric == "tpot" else -v.focus_metric
        return (slo_rank, 0, primary)

    cand_rows.sort(key=_sort_key)
    top_rows = cand_rows[: max(0, top)]

    variants: list[VariantRow] = []
    if baseline_cell:
        variants.append(_mk_variant(baseline_cell))
    variants.extend(top_rows)

    # Recommendation: the best SLO-passing variant that actually beats the
    # baseline on the focus metric; else keep the baseline (no-change).
    champion = next(
        (v for v in top_rows if v.slo_verdict == "PASS-SLO"
         and (v.pct_win_vs_baseline or 0) > 0),
        None,
    )
    recommended_cell = champion.cell_id if champion else baseline_cell
    recommended_engine = champion.engine if champion else (
        variants[0].engine if variants else None
    )

    # Roofline overlay: baseline + every selected variant that carries a sweep.
    overlay: dict[str, Any] = {}
    for v in variants:
        cell = _row_at_c(v.cell_id)
        rf = _load_roofline(campaign_dir, _artifact_cell_id(campaign_dir, cell, v.cell_id))
        if rf:
            overlay[v.cell_id] = rf

    # Gates -> tier.
    gates = _evaluate_gates(
        recommended=champion,
        trials=trials,
        same_node=same_node,
        require_workloads=require_workloads,
        workloads_present=workloads_present,
        accuracy_gate=accuracy_gate,
        local_eval_acc=local_eval_acc,
        accuracy_floor=accuracy_floor,
    )
    tier = "verdict" if (champion is not None and all(g.status == "pass" for g in gates)) else "draft"
    reasons = (
        ["champion did not beat the baseline on the focus metric -> recommend no change"]
        if champion is None
        else [f"{g.name}: {g.detail}" for g in gates if g.status != "pass"]
    )

    return ChampionResult(
        schema_version=SCHEMA_VERSION,
        campaign_id=campaign_dir.name,
        focus=resolved_focus,
        focus_c=resolved_focus_c,
        metric=resolved_metric,
        slo_ms=round(slo_ms, 2) if slo_ms is not None else None,
        hardware=hardware,
        tensor_parallel=tensor_parallel,
        baseline_cell=baseline_cell,
        variants=variants,
        recommended_cell=recommended_cell,
        recommended_engine=recommended_engine,
        tier=tier,
        gates=gates,
        reasons=reasons,
        roofline_overlay=overlay,
    )


def _evaluate_gates(
    *,
    recommended: VariantRow | None,
    trials: int | None,
    same_node: bool,
    require_workloads: tuple[str, ...],
    workloads_present: tuple[str, ...] | None,
    accuracy_gate: str,
    local_eval_acc: dict[str, float] | None = None,
    accuracy_floor: float | None = None,
) -> list[GateResult]:
    gates: list[GateResult] = []

    # Variance: same-node + >=3 trials (the DRAFT->VERDICT rule).
    if trials is None and not same_node:
        gates.append(GateResult("variance", "unknown",
                                "trials / same-node not supplied (pass --trials >=3 --same-node)"))
    elif same_node and (trials or 0) >= 3:
        gates.append(GateResult("variance", "pass", f"same-node, {trials} trials"))
    else:
        gates.append(GateResult("variance", "fail",
                                f"need same-node + >=3 trials (got same_node={same_node}, trials={trials})"))

    # Multi-workload coverage.
    if not workloads_present:
        gates.append(GateResult("multi_workload", "unknown",
                                "no --workloads-present supplied; cannot confirm the canonical suite"))
    else:
        missing = [w for w in require_workloads if w not in set(workloads_present)]
        if missing:
            gates.append(GateResult("multi_workload", "fail",
                                    f"missing workloads: {','.join(missing)}"))
        else:
            gates.append(GateResult("multi_workload", "pass",
                                    f"covered: {','.join(require_workloads)}"))

    # Accuracy.
    # Name the measured serving-quality metrics (import_model_eval eval_acc) in the detail so
    # the variant decision shows the quality it gated on.
    eval_detail = ""
    if local_eval_acc:
        worst = min(local_eval_acc.values())
        shown = ", ".join(f"{k}={v:.3f}" for k, v in sorted(local_eval_acc.items()))
        eval_detail = (
            f" (local eval_acc: {shown}; worst={worst:.3f}"
            + (f" vs floor {accuracy_floor:.3f}" if accuracy_floor is not None else "")
            + ")"
        )
    if accuracy_gate == "pass":
        gates.append(GateResult("accuracy", "pass", "accuracy gate passed" + eval_detail))
    elif accuracy_gate == "fail":
        gates.append(GateResult("accuracy", "fail", "accuracy gate FAILED -> do-not-ship" + eval_detail))
    else:
        gates.append(GateResult(
            "accuracy", "unknown",
            "accuracy gate not run (--accuracy-gate / --accuracy-floor)" + eval_detail))

    # DCGM byte-grounding (L3) of the recommended champion.
    if recommended is None:
        gates.append(GateResult("dcgm_grounded", "unknown", "no champion selected"))
    elif recommended.sol.l3_present:
        gates.append(GateResult("dcgm_grounded", "pass",
                                f"champion sol_rigor={recommended.sol.sol_rigor}"))
    else:
        gates.append(GateResult("dcgm_grounded", "fail",
                                "champion has no L3 DCGM byte-grounding (run dcgm_correlate / roofline-sweep)"))
    return gates


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(v: float | None, nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def render_markdown(result: ChampionResult, *, title: str | None = None) -> str:
    metric_label = "TPOT ms (lower=better)" if result.metric == "tpot" else "tok/s/GPU"
    rec = result.recommended_cell or "(none)"
    tier = result.tier.upper()
    lines: list[str] = []
    lines.append(f"# {title or 'Champion selection'} -- {result.campaign_id}")
    lines.append("")
    lines.append(
        f"**RECOMMENDED FOR PRODUCTION: `{rec}` ({result.recommended_engine or '?'})  "
        f"[{tier}]**"
    )
    lines.append("")
    lines.append(
        f"- focus: `{result.focus}`  |  metric: `{metric_label}`  |  "
        f"focus concurrency c={result.focus_c}  |  hw: {result.hardware} TP={result.tensor_parallel}"
    )
    if result.slo_ms is not None:
        lines.append(f"- TPOT SLO ceiling: {result.slo_ms:.1f} ms")
    lines.append(f"- baseline: `{result.baseline_cell}`")
    lines.append("")

    # Comparison table: baseline vs top-X.
    lines.append("## Baseline vs top variants")
    lines.append("")
    lines.append(
        "| variant | engine | " + metric_label + " | %win | TPOT ms | SLO | sol_rigor | "
        "HBM% | tensor% | SM% | roofline |"
    )
    lines.append("|" + "---|" * 11)
    for v in result.variants:
        tag = " (baseline)" if v.is_baseline else ""
        win = "--" if v.is_baseline else (f"{v.pct_win_vs_baseline:+.1f}%"
                                          if v.pct_win_vs_baseline is not None else "n/a")
        lines.append(
            f"| `{v.cell_id}`{tag} | {v.engine} | {_fmt(v.focus_metric)} | {win} | "
            f"{_fmt(v.tpot_median_ms)} | {v.slo_verdict} | {v.sol.sol_rigor} | "
            f"{_fmt(v.sol.hbm_pct_sol)} | {_fmt(v.sol.tensor_pct_sol)} | "
            f"{_fmt(v.sol.sm_active_pct)} | {'yes' if v.has_roofline else 'no'} |"
        )
    lines.append("")

    # Decision + gates.
    lines.append(f"## Decision: {tier}")
    lines.append("")
    for g in result.gates:
        mark = {"pass": "[pass]", "fail": "[FAIL]", "unknown": "[unknown]"}[g.status]
        lines.append(f"- {mark} **{g.name}** -- {g.detail}")
    if result.reasons:
        lines.append("")
        lines.append("Why not a VERDICT / notes:")
        for r in result.reasons:
            lines.append(f"- {r}")
    lines.append("")
    lines.append(
        "_A VERDICT requires variance (same-node + >=3 trials), the multi-workload "
        "suite, the accuracy gate, and L3 DCGM byte-grounding of the champion. "
        "Anything short is a DRAFT recommendation._"
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: ChampionResult, campaign_dir: Path, *,
                  out_md: Path | None = None, title: str | None = None) -> tuple[Path, Path]:
    json_path = campaign_dir / "champion_select.json"
    md_path = out_md or (campaign_dir / "CHAMPION.md")
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(result, title=title))
    return json_path, md_path
