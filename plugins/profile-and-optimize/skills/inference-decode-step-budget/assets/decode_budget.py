#!/usr/bin/env python3
"""decode_budget.py -- compute the c=1 (low-concurrency) decode-step budget
from a profiler trace, for the inference-decode-step-budget skill.

Auto-detects the trace format:
  * nsys SQLite  (.sqlite)            -- from `nsys export --type sqlite <rep>`
                                         or `nsys stats` (which builds it).
  * kineto JSON  (.json / .json.gz)   -- from vLLM's torch profiler
                                         (--profiler-config.profiler=torch).

It enforces the skill's correctness gates:
  Gate 2  true GPU-busy = union(KERNEL u GRAPH_TRACE u MEMCPY); never KERNEL-only.
  Gate 3  reconcile step cadence / tokens_per_step against driver TPOT (--tpot-ms).
  Gate 4  capture-quality: must have CUDA-kernel data AND a repeating decode-step
          pattern (a dominant inter-step idle-gap bucket); else FAIL the trace.

Output: per-device true-busy %, idle-gap histogram, inter-step cadence, the
per-step busy-vs-idle split, a host/kernel/comm hint, and a GREEN/RED verdict.

Stdlib only (sqlite3, json, gzip). Usage:
  python3 decode_budget.py TRACE [--tpot-ms 6.21] [--tokens-per-step 2]
"""
import argparse
import collections
import gzip
import json
import os
import sqlite3
import statistics
import sys


def union_busy(intervals):
    """Total covered length + merged intervals of (start,end) pairs (ns)."""
    if not intervals:
        return 0, []
    iv = sorted([list(x) for x in intervals])
    merged = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return sum(e - s for s, e in merged), merged


def gap_histogram(merged):
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    buckets = collections.Counter()
    bsum = collections.Counter()
    for g in gaps:
        gm = g / 1e6  # ns -> ms
        k = ("<0.5ms" if gm < 0.5 else "0.5-3ms" if gm < 3 else
             "3-8ms" if gm < 8 else "8-16ms" if gm < 16 else ">16ms")
        buckets[k] += 1
        bsum[k] += g
    return gaps, buckets, bsum


def report(label, span_ns, busy_ns, merged, comm_ns=None, parts=None,
           tpot_ms=None, tokens_per_step=None):
    span_ms = span_ns / 1e6
    busy_ms = busy_ns / 1e6
    idle_ms = span_ms - busy_ms
    print(f"\n=== {label} ===")
    if parts:
        ps = "  ".join(f"{k}={v/1e6:.0f}ms" for k, v in parts.items())
        print(f"components: {ps}")
    print(f"span={span_ms:.0f}ms  TRUE-busy={busy_ms:.0f}ms ({100*busy_ms/span_ms:.1f}%)  "
          f"idle={idle_ms:.0f}ms ({100*idle_ms/span_ms:.1f}%)")
    if comm_ns is not None and busy_ns:
        print(f"comm (allreduce/allgather) = {comm_ns/1e6:.0f}ms ({100*comm_ns/busy_ns:.1f}% of busy)")

    gaps, buckets, bsum = gap_histogram(merged)
    print("idle-gap buckets (count, total_ms):",
          {k: (buckets[k], round(bsum[k] / 1e6, 1)) for k in buckets})

    # Gate 4: repeating decode-step pattern -> a dominant 3-16ms inter-step bucket
    stepgaps = [g for g in gaps if 3e6 <= g <= 16e6]
    quality = "GREEN"
    notes = []
    if busy_ns == 0:
        quality = "RED"
        notes.append("no GPU-kernel data (load-time / lull / wrong window)")
    if not stepgaps:
        quality = "RED"
        notes.append("no repeating decode-step gaps (3-16ms) -> not steady decode")
    if stepgaps:
        med_gap = statistics.median(stepgaps) / 1e6
        n = len(stepgaps)
        per_busy_us = busy_ns / 1e3 / max(n, 1)
        per_idle_us = sum(stepgaps) / 1e3 / max(n, 1)
        step_ms = med_gap + per_busy_us / 1e3
        print(f"decode steps~{n}  inter-step idle med={med_gap:.2f}ms  "
              f"per-step: busy~{per_busy_us:.0f}us idle~{per_idle_us:.0f}us  cadence~{step_ms:.2f}ms/step")
        verdict = ("HOST-BOUND" if per_busy_us / 1e3 < 0.5 * med_gap else
                   "BALANCED/KERNEL-BOUND")
        print(f"classification hint: {verdict} (GPU-busy {per_busy_us:.0f}us vs host-idle {per_idle_us:.0f}us per step)")
        # Gate 3: TPOT reconciliation
        if tpot_ms:
            tps = tokens_per_step or 1
            modeled = step_ms / tps
            ok = abs(modeled - tpot_ms) / tpot_ms <= 0.15
            print(f"Gate 3 reconcile: step {step_ms:.2f}ms / {tps} tok = {modeled:.2f} ms/tok "
                  f"vs driver TPOT {tpot_ms:.2f} ms/tok -> {'OK' if ok else 'MISMATCH (capture suspect)'}")
            if not ok:
                quality = "RED"
                notes.append("TPOT reconciliation mismatch >15%")
    print(f"CAPTURE QUALITY: {quality}" + (f"  [{'; '.join(notes)}]" if notes else ""))
    return quality


# ---------------------------------------------------------------- nsys sqlite
def analyze_nsys(path, tpot_ms, tokens_per_step):
    c = sqlite3.connect(path)
    cur = c.cursor()

    def tables():
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {r[0] for r in cur.fetchall()}

    tabs = tables()
    KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"
    if KERNEL not in tabs:
        print("RED: sqlite has no CUPTI_ACTIVITY_KIND_KERNEL (no CUDA-kernel data).")
        print("     Gate 0: on GB300 <org-id> this is usually the CUDA 12.x-image vs 13.x-driver "
              "CUPTI skew (CUPTI_ERROR_INVALID_DEVICE), NOT capture hygiene -- grep the kineto/nsys "
              "log for 'CUDA versions. CUPTI/Runtime/Driver'; fix = CUDA-13 image or zymtrace.")
        return "RED"
    GRAPH = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tabs else None
    MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY" if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else None

    # graph-usage signal (Gate 2 awareness)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs:
        cur.execute("""SELECT
            SUM(CASE WHEN s.value LIKE '%cudaGraphLaunch%' THEN 1 ELSE 0 END),
            SUM(CASE WHEN s.value LIKE '%cudaLaunchKernel%' THEN 1 ELSE 0 END)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id""")
        gl, kl = cur.fetchone()
        print(f"[graphs] cudaGraphLaunch={gl or 0} cudaLaunchKernel={kl or 0} "
              f"-> {'CUDA graphs IN PLAY (GRAPH_TRACE counted below)' if (gl or 0) else 'no graphs'}")

    cur.execute(f"SELECT DISTINCT deviceId FROM {KERNEL} ORDER BY deviceId")
    devs = [r[0] for r in cur.fetchall()]
    overall = "GREEN"
    rep_dev = devs[0] if devs else None
    for dev in devs:
        def grab(tbl):
            if not tbl:
                return []
            cur.execute(f"SELECT start,end FROM {tbl} WHERE deviceId=?", (dev,))
            return cur.fetchall()
        k = grab(KERNEL); g = grab(GRAPH); mc = grab(MEMCPY)
        kb, _ = union_busy(k); gb, _ = union_busy(g)
        allb, merged = union_busy(k + g + mc)
        span = merged[-1][1] - merged[0][0]
        # comm = allreduce/allgather eager kernels
        cur.execute(f"""SELECT SUM(x.e-x.s) FROM (SELECT k.start s,k.end e FROM {KERNEL} k
            JOIN StringIds s ON k.shortName=s.id WHERE k.deviceId=? AND
            (s.value LIKE '%allreduce%' OR s.value LIKE '%ncclDevKernel%'
             OR s.value LIKE '%all_reduce%' OR s.value LIKE '%cross_device_reduce%')) x""", (dev,))
        comm = cur.fetchone()[0] or 0
        only_rep = (dev == rep_dev)
        q = report(f"nsys dev{dev}", span, allb, merged, comm_ns=comm,
                   parts={"KERNEL": kb, "GRAPH": gb},
                   tpot_ms=tpot_ms if only_rep else None,
                   tokens_per_step=tokens_per_step)
        if q == "RED":
            overall = "RED"
    return overall


# ---------------------------------------------------------------- kineto json
def analyze_kineto(path, tpot_ms, tokens_per_step):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        trace = json.load(f)
    ev = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu = [(int(e["ts"] * 1000), int((e["ts"] + e.get("dur", 0)) * 1000))  # us -> ns
           for e in ev if e.get("ph") == "X" and e.get("cat") in GPU_CATS and "ts" in e]
    if not gpu:
        print("RED: kineto trace has no GPU kernel/memcpy events (no CUDA-kernel data).")
        return "RED"
    graph_launches = sum(1 for e in ev if "cudaGraphLaunch" in (e.get("name") or ""))
    kern_launches = sum(1 for e in ev if "cudaLaunchKernel" in (e.get("name") or ""))
    print(f"[graphs] cudaGraphLaunch={graph_launches} cudaLaunchKernel={kern_launches} "
          f"-> {'graphs in play (kineto attributes graph kernels as events)' if graph_launches else 'no graphs'}")
    comm = sum(e1 - s1 for s1, e1 in
               [(int(e["ts"] * 1000), int((e["ts"] + e.get("dur", 0)) * 1000))
                for e in ev if e.get("ph") == "X" and e.get("cat") in GPU_CATS
                and any(x in (e.get("name") or "").lower()
                        for x in ("allreduce", "nccl", "all_reduce", "reduce_scatter", "all_gather", "allgather"))])
    print("\n[!] KINETO OVERHEAD CAVEAT: the torch profiler slows decode (~6x on")
    print("    TP=8 NVFP4+graph deploys) and inflates kernel durations, so the")
    print("    ABSOLUTE GPU-busy% below is NOT trustworthy (reads high vs reality).")
    print("    The ROBUST signal is the inter-step host-gap median (host gaps are")
    print("    not distorted). For an accurate absolute budget use the nsys-cuda")
    print("    backend (--profiler-config.profiler=cuda + nsys). See the skill.")
    busy, merged = union_busy(gpu)
    span = merged[-1][1] - merged[0][0]
    return report("kineto (GPU stream union; busy% overhead-distorted)", span, busy,
                  merged, comm_ns=comm, tpot_ms=tpot_ms, tokens_per_step=tokens_per_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="nsys .sqlite or kineto .json[.gz]")
    ap.add_argument("--tpot-ms", type=float, default=None,
                    help="driver-measured TPOT (ms/tok) for the reconciliation gate")
    ap.add_argument("--tokens-per-step", type=float, default=1,
                    help="tokens emitted per decode step (MTP accept_len; ~2 for K=1)")
    a = ap.parse_args()
    p = a.trace
    if not os.path.exists(p):
        sys.exit(f"no such trace: {p}")
    if p.endswith(".sqlite"):
        q = analyze_nsys(p, a.tpot_ms, a.tokens_per_step)
    elif p.endswith(".json") or p.endswith(".json.gz"):
        q = analyze_kineto(p, a.tpot_ms, a.tokens_per_step)
    else:
        sys.exit("unknown trace type (want .sqlite or .json[.gz]). For nsys: "
                 "`nsys export --type sqlite <rep>` first.")
    print(f"\nOVERALL: {q}")
    sys.exit(0 if q == "GREEN" else 2)


if __name__ == "__main__":
    main()
