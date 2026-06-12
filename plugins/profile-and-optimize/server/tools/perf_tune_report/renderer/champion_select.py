"""Champion-selection page (the "obvious production choice" deliverable).

Consumes the ``champion_select.json`` artifact written by
``tools/perf_tune_report/champion_select.py`` (the ``champion_select`` verb) and
renders ONE page that makes the ship decision obvious:

- a RECOMMENDED-FOR-PRODUCTION banner + the DRAFT/VERDICT tier + the gate
  breakdown (variance / multi-workload / accuracy / DCGM-grounded);
- a baseline-vs-top-X table with, per variant: the focus metric, %win vs
  baseline, TPOT, the SLO verdict, the 4-layer ``sol_rigor``, and the L3 DCGM
  HBM / tensor / SM utilization;
- ONE roofline panel overlaying the baseline + ALL selected variants' decode +
  prefill operating points, so the headroom story is visual (the same per-GPU
  arithmetic-intensity vs achieved-FLOPS math as the page-7 prefill/decode
  roofline, reused so the two pages are comparable).

This page is conditional on ``champion_select.json`` being present; the renderer
records the omission loudly (``OMISSION_REASONS['champion_select']``) when it is
absent, never silently drops it.

Added in profile-and-optimize v1.66.0.
"""

from __future__ import annotations

from typing import Any

_REQUIRED_FIELDS = ("recommended_cell", "tier", "variants", "gates")

_TIER_NOTE = (
    "VERDICT requires same-node + >=3 trials, the multi-workload suite, the "
    "accuracy gate, and L3 DCGM byte-grounding of the champion. Anything short "
    "is a DRAFT recommendation."
)


class ChampionSelectJsonMalformed(Exception):
    def __init__(self, reason: str):
        super().__init__(f"champion_select.json malformed: {reason}")


def _compute_peak(ceilings_hw: dict[str, Any], quant: str) -> tuple[float, float]:
    """(compute_peak_pflops_per_gpu, hbm_peak_tbps_per_gpu) for the quant.

    Mirrors ``prefill_decode_roofline._compute_peak`` so the champion overlay
    and the page-7 roofline use identical ceilings."""
    q = (quant or "").upper()
    key = "nvfp4_dense_pflops"
    if q == "FP8":
        key = "fp8_dense_pflops"
    elif q in ("BF16", "FP16"):
        key = "bf16_dense_pflops"
    comp = (ceilings_hw.get(key) or {}).get("value")
    hbm = (ceilings_hw.get("hbm3e_tbps") or ceilings_hw.get("hbm3_tbps") or {}).get("value")
    return float(comp) if comp else 15.0, float(hbm) if hbm else 8.0


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _draw_roofline(axR, payload: dict[str, Any], ceilings: dict[str, Any] | None,
                   hardware_key: str | None) -> None:
    overlay = payload.get("roofline_overlay") or {}
    if not overlay or not ceilings or not hardware_key:
        axR.axis("off")
        axR.text(0.5, 0.5,
                 "Roofline overlay unavailable\n(no roofline_sweep.json for the "
                 "selected variants,\nor no sol-ceilings.yaml for this hardware).\n"
                 "Run roofline-sweep.sh + import_roofline_sweep on each variant.",
                 ha="center", va="center", fontsize=8, color="#a33",
                 transform=axR.transAxes)
        return

    hw = ceilings.get(hardware_key, {})
    first = next(iter(overlay.values()))
    comp_pf, hbm_tb = _compute_peak(hw, first.get("quant", "NVFP4"))
    comp_ceil = comp_pf * 1e3  # TFLOPS/GPU
    ridge = (comp_pf * 1e15) / (hbm_tb * 1e12)

    ai = [10 ** (i / 8) for i in range(0, 33)]
    axR.plot(ai, [min(a * hbm_tb, comp_ceil) for a in ai], "k-", lw=2,
             label=f"roofline ({comp_pf:.0f} PFLOPS / {hbm_tb:.0f} TB/s per GPU)")
    axR.axvline(ridge, color="#bbb", ls=":", lw=1)

    rec = payload.get("recommended_cell")
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (cid, p) in enumerate(overlay.items()):
        mk = markers[i % len(markers)]
        is_rec = cid == rec
        lab = f"{cid}{' *CHAMPION' if is_rec else ''}"
        dx, dy = [], []
        for pt in p.get("decode", []):
            t, d = pt.get("tensor_active"), pt.get("dram_active")
            if not t or not d:
                continue
            dy.append(t * comp_ceil)
            dx.append((t / d) * ridge)
        if dx:
            axR.scatter(dx, dy, s=90 if is_rec else 55, marker=mk,
                        edgecolor="k", linewidth=1.1 if is_rec else 0.4,
                        zorder=6 if is_rec else 5, label=f"decode {lab}")
        px, py = [], []
        for pt in p.get("prefill", []):
            t, d = pt.get("tensor_active"), pt.get("dram_active")
            if not t or not d:
                continue
            py.append(t * comp_ceil)
            px.append((t / d) * ridge)
        if px:
            axR.scatter(px, py, s=95 if is_rec else 60, marker=mk, facecolor="none",
                        edgecolor="#d33", linewidth=1.2, zorder=6)

    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("arithmetic intensity (FLOP/byte)", fontsize=8)
    axR.set_ylabel("achieved compute per GPU (TFLOP/s)", fontsize=8)
    axR.set_title("Roofline overlay: baseline + all selected variants (per-GPU, DCGM)",
                  fontsize=9, loc="left")
    axR.grid(True, which="both", ls=":", alpha=0.35)
    axR.legend(fontsize=6, loc="lower right")
    axR.set_ylim(comp_ceil / 1e3, comp_ceil * 2)


def render_page(fig, payload: dict[str, Any], ceilings: dict[str, Any] | None = None,
                hardware_key: str | None = None) -> None:
    """Draw the champion-selection page onto ``fig`` from a champion_select.json
    ``payload``. ``ceilings`` + ``hardware_key`` (optional) enable the roofline
    overlay panel; absent, the page renders the banner + table only."""
    missing = [f for f in _REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ChampionSelectJsonMalformed(f"missing fields: {missing}")

    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib import gridspec

    gs = gridspec.GridSpec(3, 1, height_ratios=[0.9, 1.5, 2.0], hspace=0.35, figure=fig)
    axHdr = fig.add_subplot(gs[0])
    axTbl = fig.add_subplot(gs[1])
    axRf = fig.add_subplot(gs[2])

    tier = str(payload.get("tier", "draft")).upper()
    rec = payload.get("recommended_cell") or "(none)"
    eng = payload.get("recommended_engine") or "?"
    metric = payload.get("metric", "tok_s_gpu")
    metric_label = "TPOT ms (lower=better)" if metric == "tpot" else "tok/s/GPU"

    # ---- header / banner ----
    axHdr.axis("off")
    axHdr.text(0.0, 0.92, "Production recommendation", fontsize=13, fontweight="bold",
               va="top", transform=axHdr.transAxes)
    banner_color = "#1a7a1a" if tier == "VERDICT" else "#b06a00"
    axHdr.text(0.0, 0.60, f"RECOMMENDED: {rec}  ({eng})   [{tier}]",
               fontsize=12, fontweight="bold", color=banner_color, va="top",
               transform=axHdr.transAxes)
    axHdr.text(0.0, 0.30,
               f"focus={payload.get('focus','?')}  metric={metric_label}  "
               f"c={payload.get('focus_c','?')}  hw={payload.get('hardware','?')} "
               f"TP={payload.get('tensor_parallel','?')}  "
               f"SLO={_fmt(payload.get('slo_ms'))} ms  baseline={payload.get('baseline_cell','?')}",
               fontsize=8, color="#444", va="top", transform=axHdr.transAxes)
    gates = payload.get("gates") or []
    gate_str = "   ".join(
        f"{g.get('name')}={g.get('status','?').upper()}" for g in gates
    )
    axHdr.text(0.0, 0.06, f"gates:  {gate_str}", fontsize=8, color="#333",
               va="top", transform=axHdr.transAxes)

    # ---- table: baseline + top-X ----
    axTbl.axis("off")
    cols = ["variant", "engine", metric_label, "%win", "TPOT ms", "SLO",
            "sol_rigor", "HBM%", "tensor%", "SM%", "roofline"]
    cell_text: list[list[str]] = []
    row_colors: list[str] = []
    for v in payload.get("variants", []):
        sol = v.get("sol") or {}
        is_base = v.get("is_baseline")
        is_rec = v.get("cell_id") == payload.get("recommended_cell")
        win = "--" if is_base else (
            f"{v['pct_win_vs_baseline']:+.1f}%" if v.get("pct_win_vs_baseline") is not None else "n/a"
        )
        cell_text.append([
            (v.get("cell_id", "?") + (" *" if is_rec else "")),
            v.get("engine", "?"),
            _fmt(v.get("focus_metric")),
            win,
            _fmt(v.get("tpot_median_ms")),
            v.get("slo_verdict", "?"),
            sol.get("sol_rigor", "none"),
            _fmt(sol.get("hbm_pct_sol")),
            _fmt(sol.get("tensor_pct_sol")),
            _fmt(sol.get("sm_active_pct")),
            "yes" if v.get("has_roofline") else "no",
        ])
        row_colors.append("#d6f0d6" if is_rec else ("#eef3f8" if is_base else "#ffffff"))

    if cell_text:
        tbl = axTbl.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.4)
        for (r, _c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#333333")
                cell.set_text_props(color="white", fontweight="bold")
            elif r - 1 < len(row_colors):
                cell.set_facecolor(row_colors[r - 1])
    axTbl.set_title("Baseline vs top variants (* = champion)", fontsize=9, loc="left")

    # ---- roofline overlay ----
    _draw_roofline(axRf, payload, ceilings, hardware_key)

    reasons = payload.get("reasons") or []
    note = _TIER_NOTE
    if reasons:
        note = "Not a VERDICT: " + "; ".join(reasons[:3]) + ".   " + note
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=6.5, color="#666",
             wrap=True)
    fig.suptitle(f"Champion selection -- {payload.get('campaign_id','')}", fontsize=12, y=0.985)
