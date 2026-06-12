#!/usr/bin/env python3
"""SLA-first capacity sizing: given a TPM target + an interactivity SLA (tokens/s/user) + a
model's measured tok/s/user-vs-concurrency curve, compute the pods/GPUs needed.

This is the SINGLE SOURCE OF TRUTH for the "how many GPUs for N TPM" decision, the sibling of
tp_rightsize_advisor.py. It encodes the rule that sizing MUST start from the interactivity SLA,
not the TPM number alone (a TPM target is underdetermined: 3M TPM can be few users at high
tok/s/user or many users at low tok/s/user). The chain is:

  SLA S (tok/s/user)  -> c* = max concurrency per replica where tok/s/user(c*) >= S
                      -> per-replica throughput at c*
                      -> replicas = ceil( (TPM/60) / (per-replica * util) )
                      -> GPUs = replicas * gpus_per_pod

The curve is supplied as measured anchors (c, tok/s/user); per-replica throughput at an anchor
is c * tok/s/user(c). Intermediate concurrencies are interpolated piecewise-linearly in log(c).
Below the lowest-tok/s/user anchor (past the throughput knee) per-replica throughput plateaus,
so the pod count is flat; above the highest anchor (tighter than c=1) the SLA is infeasible.

DRAFT vs VERDICT: with only the c=1 / c=10 / knee anchors this is a DRAFT (anchored + interp);
feeding the full roofline sweep (c=8/16/32/64/128) makes it a VERDICT. Pure-Python, no cluster
calls (the math is structural); fail-loud on missing/degenerate inputs (asset-validation rule).

Worked cross-check (the "3M TPM, 18 pods" thread): per-replica 2916 tok/s at c=128 (tok/s/user
22.78), 8-GPU pod, util=1.0 -> ceil(50000/2916) = 18 pods = 144 GPUs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

GPU_HOURLY_USD = 8.6
HOURS_PER_MONTH = 730.0


@dataclasses.dataclass
class Anchor:
    concurrency: float
    toks_per_user: float

    @property
    def per_replica_tps(self) -> float:
        return self.concurrency * self.toks_per_user


@dataclasses.dataclass
class SizingRow:
    sla_toks_per_user: float
    concurrency: float          # c* (the max concurrency meeting the SLA)
    per_replica_tps: float
    replicas: int
    gpus: int
    usd_per_month: float
    regime: str                 # measured-anchor | interp | below-knee-plateau | latency-tier-infeasible


@dataclasses.dataclass
class SizingResult:
    model: str
    tpm: float
    tok_per_s_target: float
    gpus_per_pod: int
    util: float
    gpu_hourly_usd: float
    anchors: list[Anchor]
    rows: list[SizingRow]
    notes: list[str]


def parse_anchors(spec: str) -> list[Anchor]:
    """Parse 'c:tok_per_user,c:tok_per_user,...' into sorted, validated anchors."""
    anchors: list[Anchor] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"anchor '{part}' must be 'concurrency:toks_per_user'")
        c_s, tpu_s = part.split(":", 1)
        c, tpu = float(c_s), float(tpu_s)
        if c <= 0 or tpu <= 0:
            raise ValueError(f"anchor '{part}': concurrency and toks_per_user must be > 0")
        anchors.append(Anchor(c, tpu))
    if len(anchors) < 1:
        raise ValueError("need at least one (concurrency:toks_per_user) anchor")
    anchors.sort(key=lambda a: a.concurrency)
    # tok/s/user must be non-increasing in concurrency (more concurrent streams -> <= per-user rate)
    for lo, hi in zip(anchors, anchors[1:]):
        if hi.toks_per_user > lo.toks_per_user + 1e-9:
            raise ValueError(
                f"non-monotonic curve: tok/s/user rose from {lo.toks_per_user} @c{lo.concurrency} "
                f"to {hi.toks_per_user} @c{hi.concurrency} (must be non-increasing in concurrency)")
    return anchors


def _interp_concurrency_for_sla(anchors: list[Anchor], sla: float) -> tuple[float, float, str]:
    """Return (c*, per_replica_tps, regime) for a target SLA via log-c piecewise interpolation."""
    hi_tpu = anchors[0].toks_per_user   # largest tok/s/user (lowest concurrency)
    lo_tpu = anchors[-1].toks_per_user  # smallest tok/s/user (highest concurrency = knee)

    if sla > hi_tpu + 1e-9:
        # tighter than the c=1 (lowest-concurrency) anchor can deliver: latency tier, clamp to that anchor
        a = anchors[0]
        return a.concurrency, a.per_replica_tps, "latency-tier-infeasible"
    if sla <= lo_tpu + 1e-9:
        # at or below the knee: per-replica throughput plateaus, pod count is flat
        a = anchors[-1]
        return a.concurrency, a.per_replica_tps, "below-knee-plateau"

    # bracket: find adjacent anchors with tpu[i] >= sla >= tpu[i+1]
    for lo, hi in zip(anchors, anchors[1:]):
        if lo.toks_per_user + 1e-9 >= sla >= hi.toks_per_user - 1e-9:
            if abs(lo.toks_per_user - sla) < 1e-9:
                return lo.concurrency, lo.per_replica_tps, "measured-anchor"
            if abs(hi.toks_per_user - sla) < 1e-9:
                return hi.concurrency, hi.per_replica_tps, "measured-anchor"
            frac = (lo.toks_per_user - sla) / (lo.toks_per_user - hi.toks_per_user)
            log_c = math.log10(lo.concurrency) + frac * (math.log10(hi.concurrency) - math.log10(lo.concurrency))
            c = 10.0 ** log_c
            per_replica = c * sla
            return c, per_replica, "interp"
    # unreachable given the guards above
    raise RuntimeError(f"could not bracket SLA {sla} in anchors")


def resolve(
    *,
    tpm: float,
    sla_list: list[float],
    anchors: list[Anchor],
    gpus_per_pod: int = 4,
    util: float = 0.70,
    gpu_hourly_usd: float = GPU_HOURLY_USD,
    model: str = "model",
) -> SizingResult:
    if tpm <= 0:
        raise ValueError("--tpm must be > 0")
    if not sla_list:
        raise ValueError("need at least one SLA (tokens/s/user)")
    if not (0 < util <= 1.0):
        raise ValueError("--util must be in (0, 1]")
    if gpus_per_pod <= 0:
        raise ValueError("--gpus-per-pod must be > 0")

    tok_per_s = tpm / 60.0
    rows: list[SizingRow] = []
    for sla in sorted(sla_list):
        if sla <= 0:
            raise ValueError(f"SLA {sla} must be > 0")
        c, per_replica, regime = _interp_concurrency_for_sla(anchors, sla)
        if per_replica <= 0:
            raise ValueError(f"degenerate per-replica throughput at SLA {sla}")
        replicas = math.ceil(tok_per_s / (per_replica * util))
        gpus = replicas * gpus_per_pod
        usd = gpus * gpu_hourly_usd * HOURS_PER_MONTH
        rows.append(SizingRow(sla, round(c, 1), round(per_replica, 1), replicas, gpus, round(usd, 2), regime))

    notes: list[str] = []
    if any(r.regime == "interp" for r in rows):
        notes.append("interp rows are log-c interpolated between measured anchors (DRAFT); feed the full "
                     "roofline sweep (c=8..128) for a VERDICT.")
    if any(r.regime == "below-knee-plateau" for r in rows):
        notes.append("below-knee SLAs share the knee per-replica throughput (throughput plateaus past the "
                     "knee; the workload is host/KV-bound there), so the pod count is flat.")
    if any(r.regime == "latency-tier-infeasible" for r in rows):
        notes.append("an SLA above the c=1 anchor is a latency-tier point; serving a large TPM there needs "
                     "an impractical pod count.")
    return SizingResult(model, tpm, tok_per_s, gpus_per_pod, util, gpu_hourly_usd, anchors, rows, notes)


def render_md(r: SizingResult) -> str:
    lines = [f"# Capacity sizing (SLA-first): {r.model}", ""]
    lines.append(f"TPM target {r.tpm:,.0f} output tok/min ({r.tok_per_s_target:,.0f} tok/s); "
                 f"pod = {r.gpus_per_pod} GPUs; util {r.util:.0%}; ${r.gpu_hourly_usd}/GPU-hour.")
    lines.append("")
    lines.append("| SLA tok/s/user | c* | per-pod tok/s | pods | GPUs | $/month | regime |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in r.rows:
        lines.append(f"| {row.sla_toks_per_user:g} | {row.concurrency:g} | {row.per_replica_tps:,.0f} | "
                     f"{row.replicas} | {row.gpus} | ${row.usd_per_month:,.0f} | {row.regime} |")
    if r.notes:
        lines += ["", "## Notes"]
        lines += [f"- {n}" for n in r.notes]
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SLA-first capacity sizing (TPM + tok/s/user SLA -> pods/GPUs).")
    p.add_argument("--tpm", type=float, required=True, help="total output tokens per minute target (e.g. 3000000)")
    p.add_argument("--sla", required=True,
                   help="tokens/s/user SLA(s), comma-separated (e.g. '20,50,100' or a single '50')")
    p.add_argument("--anchors", required=True,
                   help="measured curve as 'c:tok_per_user,...' (e.g. '1:215.5,10:122.2,256:41.8')")
    p.add_argument("--gpus-per-pod", type=int, default=4, help="GPUs per replica/pod (GB300 TP4 node = 4)")
    p.add_argument("--util", type=float, default=0.70, help="target utilization headroom (default 0.70)")
    p.add_argument("--gpu-hourly-usd", type=float, default=GPU_HOURLY_USD)
    p.add_argument("--model", default="model", help="model label for the output")
    p.add_argument("--emit", default="", help="directory to write CAPACITY-SIZING.md + capacity_sizing.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    anchors = parse_anchors(args.anchors)
    sla_list = [float(s) for s in str(args.sla).split(",") if s.strip()]
    r = resolve(tpm=args.tpm, sla_list=sla_list, anchors=anchors, gpus_per_pod=args.gpus_per_pod,
                util=args.util, gpu_hourly_usd=args.gpu_hourly_usd, model=args.model)
    md = render_md(r)
    sys.stdout.write(md)
    if args.emit:
        d = Path(args.emit)
        d.mkdir(parents=True, exist_ok=True)
        (d / "capacity_sizing.json").write_text(json.dumps(dataclasses.asdict(r), indent=2) + "\n")
        (d / "CAPACITY-SIZING.md").write_text(md)
        sys.stdout.write(f"\nwrote {d / 'capacity_sizing.json'} + {d / 'CAPACITY-SIZING.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
