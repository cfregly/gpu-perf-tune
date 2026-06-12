#!/usr/bin/env python3
"""TP right-size resolver: given a model's params + dtype + GPU, recommend tensor-parallel size and
flag OVER-PROVISIONING (small-active-MoE spread too thin across TP) -- an energy/OPEX lever the
throughput-only tuning misses.

This is the SINGLE SOURCE OF TRUTH for the TP right-size decision, the sibling of loader_advisor.py.
Codifies the FLEET-ENERGY-AUDIT (perf-tune-report/cross-model/FLEET-ENERGY-AUDIT-*.md) right-size rule:

  Size TP so ACTIVE-params/GPU is loaded (>= ~3B). A small-active MoE served at a TP higher than
  memory requires spreads ~3B active over N GPUs -> each GPU near-idle (low W/GPU, low SM) AND worse
  per-GPU throughput (sharding comm overhead, underfill). Right-size to the lowest TP that fits memory.

MEASURED basis (GB300, NVFP4, mns=192, c=192, ISL4096/OSL512, 2026-06-09):
  - Qwen3-30B-A3B TP1 = 11,968 tok/s/GPU @ ~820 W/GPU  (right-sized: 3B active / 1 GPU)
  - Nemotron-30B-A3B TP4 = 4,378 tok/s/GPU @ ~362 W/GPU (over-provisioned: 3.6B active / 4 GPU = 0.9B/GPU)
  => TP=1 is 2.73x more GPU-efficient for the small-active-MoE class; ~$15.7k/standing-replica/month
     at $8.6/GPU-hour (matched throughput, ~2.7x fewer GPUs). Power-confirmed also on Qwen3-Next-80B-A3B
     (TP4 ~400 W/GPU). Large-active tiers (V4-Flash 13B-act, EXAONE-236B-A23B) at TP4 keep >=3B/GPU and
     are NOT flagged.

Confidence: MEASURED for small-total (<= ~40B) small-active (<= ~4B) MoE; EXTRAPOLATED for larger-total
models (they fit a low TP by memory, but prefill compute / KV may still favor higher TP -- verify with
a same-node TP A/B before shipping). Pure-Python, no cluster calls (the decision is structural).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

VALID_TP = [1, 2, 4, 8]
BYTES_PER_PARAM = {"nvfp4": 0.5, "fp4": 0.5, "mxfp4": 0.5, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0}
TARGET_ACTIVE_PER_GPU_B = 3.0   # >= this = a loaded GPU (from the measured Qwen3-30B TP1 = 3B/GPU @ 820W)
SAFE_ACTIVE_PER_GPU_B = 6.0     # right-size target (loaded but headroom); active/rec_tp ~ this
WEIGHT_FRAC_OF_USABLE = 0.55    # weights must fit in this fraction of usable mem (rest = KV + activations)
GPU_HOURLY_USD = 8.6


@dataclasses.dataclass
class Gate:
    name: str
    status: str  # pass | warn | info
    detail: str


@dataclasses.dataclass
class TpResult:
    recommended_tp: int
    over_provisioned: bool
    current_tp: int | None
    min_tp_memory: int
    active_per_gpu_current_b: float | None
    active_per_gpu_recommended_b: float
    gpu_reduction_factor: float | None  # current_tp / recommended_tp (matched-throughput GPU saving)
    monthly_usd_saved_per_replica: float | None
    confidence: str  # measured | extrapolated
    gates: list[Gate]
    reasons: list[str]


def _nearest_valid_tp_le(x: float) -> int:
    cand = [t for t in VALID_TP if t <= max(1.0, x)] or [1]
    return cand[-1]


def min_tp_for_memory(weights_gb: float, gpu_mem_gb: float, gmu: float) -> int:
    cap = gpu_mem_gb * gmu * WEIGHT_FRAC_OF_USABLE
    for tp in VALID_TP:
        if weights_gb / tp <= cap:
            return tp
    return VALID_TP[-1]  # may need PP / multi-node beyond TP8


def resolve(
    *,
    total_params_b: float,
    active_params_b: float,
    dtype: str = "nvfp4",
    gpu_mem_gb: float = 284.0,   # GB300
    gmu: float = 0.9,
    current_tp: int | None = None,
) -> TpResult:
    bpp = BYTES_PER_PARAM.get(dtype.lower(), 0.5)
    weights_gb = total_params_b * bpp
    min_tp_mem = min_tp_for_memory(weights_gb, gpu_mem_gb, gmu)

    active_cur = (active_params_b / current_tp) if current_tp else None
    over = bool(current_tp and current_tp > min_tp_mem and active_cur is not None and active_cur < TARGET_ACTIVE_PER_GPU_B)

    if over:
        # right-size: lowest TP that fits memory + keeps active/GPU loaded (~SAFE target)
        rec = max(min_tp_mem, _nearest_valid_tp_le(active_params_b / SAFE_ACTIVE_PER_GPU_B))
        rec = min(rec, current_tp)  # never recommend MORE TP than current when reducing
    else:
        rec = current_tp if current_tp else max(min_tp_mem, _nearest_valid_tp_le(active_params_b / SAFE_ACTIVE_PER_GPU_B))

    active_rec = active_params_b / rec
    reduction = (current_tp / rec) if (current_tp and rec) else None
    monthly = (reduction - 1.0) * rec * GPU_HOURLY_USD * 730 if (reduction and reduction > 1.0) else None
    # GPUs saved = current_tp - rec; cost = (current_tp - rec) * $/hr * 730. (reduction-1)*rec = current-rec.

    confidence = "measured" if (total_params_b <= 40 and active_params_b <= 4) else "extrapolated"

    gates = [
        Gate("memory_fit", "info", f"{weights_gb:.0f} GB weights ({dtype}) fit at min TP={min_tp_mem} "
                                   f"({gpu_mem_gb:.0f} GB/GPU, gmu={gmu}, weights<= {WEIGHT_FRAC_OF_USABLE:.0%} usable)"),
        Gate("active_density", "warn" if over else "pass",
             (f"active {active_params_b:.1f}B / current TP{current_tp} = {active_cur:.2f}B/GPU "
              f"< {TARGET_ACTIVE_PER_GPU_B:.0f}B target -> GPUs underfilled" if over
              else (f"active {active_params_b:.1f}B / TP{current_tp} = {active_cur:.2f}B/GPU (loaded)" if current_tp
                    else f"active {active_params_b:.1f}B; recommend TP keeps >= {TARGET_ACTIVE_PER_GPU_B:.0f}B/GPU"))),
    ]
    reasons: list[str] = []
    if over:
        reasons.append(
            f"OVER-PROVISIONED: {active_params_b:.1f}B active across TP{current_tp} = {active_cur:.2f}B/GPU "
            f"(< {TARGET_ACTIVE_PER_GPU_B:.0f}B). The small active-param MoE does not benefit from {current_tp}-way "
            f"sharding -> near-idle GPUs + worse per-GPU throughput (measured 2.73x on the 30B-A3B class).")
        reasons.append(
            f"RIGHT-SIZE to TP{rec} (active {active_rec:.1f}B/GPU): ~{reduction:.1f}x fewer GPUs at matched "
            f"throughput" + (f", ~${monthly:,.0f}/standing-replica/month at ${GPU_HOURLY_USD}/GPU-hr." if monthly else "."))
        if confidence == "extrapolated":
            reasons.append(
                "CONFIDENCE=extrapolated: total params > ~40B, so the low-TP weights fit but prefill compute / "
                "KV may still favor a higher TP -- verify with a same-node TP A/B before shipping.")
    else:
        if current_tp:
            reasons.append(f"TP{current_tp} is right-sized: active {active_cur:.2f}B/GPU >= {TARGET_ACTIVE_PER_GPU_B:.0f}B "
                           f"(loaded). No change.")
        else:
            reasons.append(f"Recommend TP{rec} (active {active_rec:.1f}B/GPU; min TP for memory = {min_tp_mem}).")
    return TpResult(rec, over, current_tp, min_tp_mem, active_cur, active_rec, reduction, monthly, confidence, gates, reasons)


def render_md(r: TpResult) -> str:
    lines = ["# TP right-size", ""]
    head = f"**RECOMMENDED TP: {r.recommended_tp}**"
    if r.over_provisioned:
        head += f"  [OVER-PROVISIONED at TP{r.current_tp} -> right-size, ~{r.gpu_reduction_factor:.1f}x fewer GPUs]"
    head += f"  ({r.confidence})"
    lines.append(head)
    lines += ["", "## Gates"]
    for g in r.gates:
        mark = {"pass": "[pass]", "warn": "[warn]", "info": "[info]"}[g.status]
        lines.append(f"- {mark} **{g.name}** -- {g.detail}")
    lines += ["", "## Rationale"]
    for x in r.reasons:
        lines.append(f"- {x}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recommend TP + flag over-provisioning (energy/OPEX right-size).")
    p.add_argument("--total-params-b", type=float, required=True, help="total params in billions")
    p.add_argument("--active-params-b", type=float, required=True, help="active params/token in billions (=total for dense)")
    p.add_argument("--dtype", default="nvfp4", choices=sorted(set(BYTES_PER_PARAM)))
    p.add_argument("--gpu-mem-gb", type=float, default=284.0, help="per-GPU memory (GB300=284)")
    p.add_argument("--gmu", type=float, default=0.9)
    p.add_argument("--current-tp", type=int, default=None, help="current TP (to flag over-provisioning)")
    p.add_argument("--emit", default="", help="directory to write tp_rightsize.json + TP-RIGHTSIZE.md")
    p.add_argument("--print-tp", action="store_true", help="print only the recommended TP and exit")
    args = p.parse_args(argv)

    r = resolve(total_params_b=args.total_params_b, active_params_b=args.active_params_b, dtype=args.dtype,
                gpu_mem_gb=args.gpu_mem_gb, gmu=args.gmu, current_tp=args.current_tp)
    if args.print_tp:
        print(r.recommended_tp)
        return 1 if r.over_provisioned else 0
    md = render_md(r)
    sys.stdout.write(md)
    if args.emit:
        d = Path(args.emit); d.mkdir(parents=True, exist_ok=True)
        (d / "tp_rightsize.json").write_text(json.dumps(dataclasses.asdict(r), indent=2) + "\n")
        (d / "TP-RIGHTSIZE.md").write_text(md)
        sys.stdout.write(f"\nwrote {d / 'tp_rightsize.json'} + {d / 'TP-RIGHTSIZE.md'}\n")
    return 1 if r.over_provisioned else 0


if __name__ == "__main__":
    raise SystemExit(main())
