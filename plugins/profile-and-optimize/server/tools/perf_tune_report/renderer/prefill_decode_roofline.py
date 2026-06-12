"""Prefill vs Decode Roofline page (page 7, always-on, L3).

Implements ``ROOFLINE-METHODOLOGY.md``. The phase-separated roofline the
per-category SoL page (page 4) and the workload-level DCGM page (page 6) do NOT
provide. Consumes ``cells/<id>/roofline_sweep.json`` (from
``profiling/roofline-sweep.sh`` -> ``importers/roofline_sweep.py``): a decode
concurrency sweep + a prefill ISL sweep, each with MEASURED bench throughput +
in-pod DCGM PROF active fractions.

Three panels, per-GPU normalized so TP2/TP4/TP8 share one ceiling:

- **A (roofline):** operating point = (analytical arithmetic intensity x,
  measured achieved-compute/GPU y). ``x`` is FLOP/byte from the model's
  ``config.json`` (``roofline_math``); ``y = flop_per_token x measured_tok/s /
  n_gpus``. Decode points (by concurrency) sit far LEFT of the ridge
  (memory-bound) and BELOW the memory diagonal -- their vertical gap below the
  diagonal IS the HBM-BW utilization. Prefill points (by ISL) climb toward the
  compute ceiling. Falls back to the DCGM tensor/dram-active proxy (clearly
  labeled) only when the model shape is unavailable.
- **B (Q2 "decode >=75% HBM"):** TWO honestly-labeled curves -- the DCGM
  ``DRAM_ACTIVE`` duty-cycle proxy AND the byte-grounded delivered-BW %
  (analytical bytes/token x measured tok/s / peak) -- plus the 75% reference.
- **C (Q1 "what C maxes the TFLOPs"):** tensor-pipe + SM active vs concurrency.

L3 (DCGM byte/FLOP-grounded). The renderer marks the campaign ``dcgm_grounded``
+ ``sol_complete`` when this page draws.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from tools.perf_tune_report import roofline_math

_REQUIRED_FIELDS = ("schema", "hardware", "tensor_parallel", "decode", "prefill")


class RooflineSweepJsonMalformed(Exception):
    def __init__(self, cell_id: str, reason: str):
        super().__init__(f"roofline_sweep.json malformed (cell {cell_id}): {reason}")


def _compute_peak(ceilings_hw: dict[str, Any], quant: str) -> tuple[float, float]:
    """(compute_peak_pflops_per_gpu, hbm_peak_tbps_per_gpu) for the quant."""
    q = (quant or "").upper()
    key = "nvfp4_dense_pflops"
    if q == "FP8":
        key = "fp8_dense_pflops"
    elif q in ("BF16", "FP16"):
        key = "bf16_dense_pflops"
    comp = (ceilings_hw.get(key) or {}).get("value")
    hbm = (ceilings_hw.get("hbm3e_tbps") or ceilings_hw.get("hbm3_tbps") or {}).get("value")
    return float(comp) if comp else 15.0, float(hbm) if hbm else 8.0


def _resolve_shape(payload: dict[str, Any]) -> "roofline_math.ModelShape | None":
    """The analytical ModelShape for this config: prefer the importer-embedded
    ``analytical_shape`` block (self-contained, captured from the in-pod
    config.json), else the registry lookup by served model name. None => the
    panel-A placement degrades to the DCGM proxy (labeled)."""
    embedded = payload.get("analytical_shape")
    if isinstance(embedded, dict) and embedded.get("hidden_size"):
        try:
            return roofline_math.shape_from_dict(embedded)
        except Exception:  # noqa: BLE001  (malformed embed -> fall through to registry)
            pass
    return roofline_math.shape_for_model(payload.get("model", ""))


def _decode_ctx(pt: dict[str, Any]) -> int:
    """Representative context length a decode step reads KV over: ISL + half the
    OSL (mid-generation). Defaults to 512 for the standard ISL=256/OSL=512 sweep."""
    isl = pt.get("isl") or 256
    osl = pt.get("osl") or 512
    return int(isl + osl // 2)


def _kv_dtype(payload: dict[str, Any]) -> str:
    return str(payload.get("kv_dtype") or "fp8")


def _decode_rate(pt: dict[str, Any]) -> float | None:
    return pt.get("output_throughput")


def _prefill_rate(pt: dict[str, Any]) -> float | None:
    inp, dur = pt.get("total_input_tokens"), pt.get("duration")
    return (inp / dur) if (inp and dur) else None


def render_page(
    fig,
    cell_roofline: "OrderedDict[str, dict[str, Any]]",
    ceilings: dict[str, Any],
    hardware_key: str,
) -> None:
    """Draw the prefill/decode roofline page. ``cell_roofline`` maps cell_id ->
    roofline_sweep.json payload (one per (model, TP) config)."""
    if not cell_roofline:
        raise ValueError("prefill_decode_roofline.render_page: cell_roofline is empty")
    for cid, p in cell_roofline.items():
        missing = [f for f in _REQUIRED_FIELDS if f not in p]
        if missing:
            raise RooflineSweepJsonMalformed(cid, f"missing fields: {missing}")

    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib import gridspec

    hw = ceilings.get(hardware_key, {})
    hw_name = hw.get("hw_name", hardware_key)

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                           hspace=0.34, wspace=0.26, figure=fig)
    axR = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # Per-GPU ceilings shared by all configs (so TP2/4/8 overlay on one roofline).
    first = next(iter(cell_roofline.values()))
    comp_pf, hbm_tb = _compute_peak(hw, first.get("quant", "NVFP4"))
    comp_ceil = comp_pf * 1e3  # TFLOPS/GPU
    ridge = (comp_pf * 1e15) / (hbm_tb * 1e12)

    # ---- Panel A: per-GPU roofline ----
    # memory diagonal in TFLOPS/GPU: y = AI * hbm_tb (the 1e12 cancels TB/s vs TFLOP/s)
    ai_axis = [10 ** (i / 8) for i in range(-8, 33)]  # ~0.1 .. ~1e4
    axR.plot(ai_axis, [min(a * hbm_tb, comp_ceil) for a in ai_axis], "k-", lw=2,
             label=f"roofline (per-GPU: {comp_pf:.0f} PFLOPS / {hbm_tb:.0f} TB/s)")
    axR.axvline(ridge, color="#bbb", ls=":", lw=1)
    axR.text(ridge * 1.05, comp_ceil * 0.35, f"ridge\nAI={ridge:.0f}", fontsize=7, color="#666")

    def _cfg_label(cid: str, tp: Any) -> str:
        base = cid
        for suf in ("-decode", "-prefill"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        return f"{base} (TP{tp})" if base else f"TP={tp}"

    markers = ["o", "s", "^", "D", "v", "P"]
    any_proxy = False
    colorbar_done = False
    for i, (cid, p) in enumerate(cell_roofline.items()):
        tp = int(p.get("tensor_parallel") or 1)
        lab = _cfg_label(cid, tp)
        mk = markers[i % len(markers)]
        shape = _resolve_shape(p)
        quant = p.get("quant", "NVFP4")
        kvd = _kv_dtype(p)

        # decode points (colored by concurrency)
        dx, dy, dc = [], [], []
        for pt in p.get("decode", []):
            c = pt.get("c")
            t, d = pt.get("tensor_active"), pt.get("dram_active")
            if shape is not None and c:
                rate = _decode_rate(pt)
                if not rate:
                    continue
                x = shape.decode_arithmetic_intensity(c, _decode_ctx(pt), quant, kvd)
                y = shape.flop_per_token * rate / tp / 1e12  # TFLOP/s per GPU
            else:  # DCGM-proxy fallback (no model shape)
                if not t or not d:
                    continue
                any_proxy = True
                x = (t / d) * ridge
                y = t * comp_ceil
            if x and y:
                dx.append(x); dy.append(y); dc.append(c)
        if dx:
            sc = axR.scatter(dx, dy, c=dc, cmap="viridis", s=70, marker=mk,
                             edgecolor="k", linewidth=0.4, zorder=5)
            axR.plot(dx, dy, "-", color="#3b7", lw=0.8, alpha=0.5, zorder=4)
            if not colorbar_done:
                cb = fig.colorbar(sc, ax=axR, pad=0.01, fraction=0.045)
                cb.set_label("decode concurrency C", fontsize=8)
                colorbar_done = True

        # prefill points (red, by ISL)
        px, py = [], []
        for pt in p.get("prefill", []):
            t, d = pt.get("tensor_active"), pt.get("dram_active")
            if shape is not None and pt.get("isl"):
                rate = _prefill_rate(pt)
                if not rate:
                    continue
                px.append(shape.prefill_arithmetic_intensity(pt["isl"], quant))
                py.append(shape.flop_per_token * rate / tp / 1e12)
            else:
                if not t or not d:
                    continue
                any_proxy = True
                px.append((t / d) * ridge); py.append(t * comp_ceil)
        if px:
            axR.scatter(px, py, marker=mk, c="#d33", s=95, edgecolor="k",
                        linewidth=0.4, zorder=6, label=f"prefill {lab}")

    axR.set_xscale("log"); axR.set_yscale("log")
    xlab = ("arithmetic intensity (FLOP/byte) [analytical, from config.json]"
            if not any_proxy else
            "arithmetic intensity (FLOP/byte) [analytical; * = DCGM-proxy fallback]")
    ylab = ("achieved compute / GPU (TFLOP/s) [flop_per_token x measured tok/s]"
            if not any_proxy else
            "achieved compute / GPU (TFLOP/s)")
    axR.set_xlabel(xlab, fontsize=8)
    axR.set_ylabel(ylab, fontsize=8)
    axR.set_title("Prefill + Decode Roofline (per-GPU; analytical AI x measured tok/s)",
                  fontsize=10, loc="left")
    axR.grid(True, which="both", ls=":", alpha=0.35)
    axR.legend(fontsize=6.5, loc="lower right")
    axR.set_ylim(comp_ceil / 1e4, comp_ceil * 2)

    # ---- Panel B: HBM-BW util vs concurrency (Q2) -- proxy AND byte-grounded ----
    peak_decode_hbm = 0.0
    for cid, p in cell_roofline.items():
        tp = int(p.get("tensor_parallel") or 1)
        shape = _resolve_shape(p)
        quant = p.get("quant", "NVFP4")
        kvd = _kv_dtype(p)
        cs = [pt.get("c") for pt in p.get("decode", []) if pt.get("dram_active") is not None]
        dr = [pt.get("dram_active") * 100 for pt in p.get("decode", []) if pt.get("dram_active") is not None]
        if cs:
            axB.plot(cs, dr, "o-", lw=1.8, label=f"DRAM-active % (DCGM) {_cfg_label(cid, tp)}")
            peak_decode_hbm = max(peak_decode_hbm, max(dr))
        # byte-grounded delivered-BW % (analytical bytes x measured tok/s / peak)
        if shape is not None:
            bg_cs, bg = [], []
            for pt in p.get("decode", []):
                c, rate = pt.get("c"), _decode_rate(pt)
                if not c or not rate:
                    continue
                union = (min(shape.n_routed_experts, shape.n_experts_per_tok * c)
                         if shape.is_moe else 0)
                wb_per_tok = shape.active_weight_bytes(union, quant) / c
                kv_per_tok = shape.kv_bytes_per_token(_decode_ctx(pt), kvd)
                delivered = (wb_per_tok + kv_per_tok) * rate / tp  # bytes/s per GPU
                bg_cs.append(c); bg.append(delivered / (hbm_tb * 1e12) * 100)
            if bg_cs:
                axB.plot(bg_cs, bg, "s--", lw=1.3, alpha=0.8,
                         label=f"delivered HBM-BW % (bytes x tok/s) {_cfg_label(cid, tp)}")
    axB.axhline(75, color="#d33", ls="--", lw=1.4, label="target 75%")
    axB.set_xscale("log", base=2); axB.set_ylim(0, 100)
    axB.set_xlabel("decode concurrency C", fontsize=8)
    axB.set_ylabel("HBM-BW utilization %", fontsize=8)
    axB.set_title("Q2: decode HBM-BW utilization vs concurrency", fontsize=9, loc="left")
    axB.grid(True, ls=":", alpha=0.35); axB.legend(fontsize=6.0, loc="upper left")

    # ---- Panel C: tensor + SM util vs concurrency (Q1) ----
    peak_decode_tensor = 0.0
    for cid, p in cell_roofline.items():
        cs = [pt.get("c") for pt in p.get("decode", []) if pt.get("tensor_active") is not None]
        te = [pt.get("tensor_active") * 100 for pt in p.get("decode", []) if pt.get("tensor_active") is not None]
        sm = [(pt.get("sm_active") or 0) * 100 for pt in p.get("decode", []) if pt.get("tensor_active") is not None]
        if cs:
            _l = _cfg_label(cid, p.get("tensor_parallel"))
            axC.plot(cs, te, "s-", lw=1.8, label=f"tensor {_l}")
            axC.plot(cs, sm, "d:", lw=1.2, alpha=0.7, label=f"SM {_l}")
            peak_decode_tensor = max(peak_decode_tensor, max(te))
    axC.set_xscale("log", base=2); axC.set_ylim(0, 100)
    axC.set_xlabel("decode concurrency C", fontsize=8)
    axC.set_ylabel("utilization %", fontsize=8)
    axC.set_title("Q1: compute (tensor) + SM utilization vs concurrency", fontsize=9, loc="left")
    axC.grid(True, ls=":", alpha=0.35); axC.legend(fontsize=6, loc="upper left")

    # ---- auto mechanism annotation (observation, not interpretation) ----
    note = (
        f"decode peak: tensor {peak_decode_tensor:.0f}% of compute, "
        f"HBM {peak_decode_hbm:.0f}% (target 75%). "
        + ("memory-bound: decode never saturates compute; prefill is the compute phase."
           if peak_decode_tensor < 25 else
           "decode engages compute -- check prefill vs decode split.")
    )
    fig.text(0.5, 0.012, note, ha="center", fontsize=7.5, color="#444", wrap=True)

    fig.suptitle(f"Prefill/Decode Roofline -- {hw_name} (measured DCGM x analytical AI; "
                 f"{len(cell_roofline)} config(s))", fontsize=12, y=0.98)
