#!/usr/bin/env python3
"""mirage megakernel graph-coverage auditor (read-only).

Catches the "feature defined in code but never wired into the generated task
graph" bug class -- e.g. RoPE cos/sin tables that are allocated + attached but
never consumed by any task (GLM-5.1 GB300 coherence bug, COHERENCE-ROPE). A
numerics harness silently PASSES on this class; graph-coverage catches it.

For each TP rank it cross-references:
  - DECLARED tensors:  all_tensors["NAME"] = NAME;   in kernel_<rank>.cu
  - CONSUMED tensors:  all_tasks[].inputs[].base_ptr / .outputs[].base_ptr
                       in task_graph_<rank>.json  (-> consuming task_type set)
and flags any tensor declared but never consumed as an INPUT by any task.

Read-only: parses generated artifacts only; no GPU, no kernel-source edits.

Usage:
  python3 mk-graph-coverage-audit.py [GENDIR] [--critical a,b,...] [--ranks 0,1]
GENDIR default: /work/mirage-perop2/demo/deepseek_v3
Exit: 0 = PASS (every critical tensor consumed as input), 1 = FAIL, 2 = no artifacts.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
from collections import defaultdict

DECL_RE = re.compile(r'all_tensors\[\s*"([^"]+)"\s*\]\s*=')
DEFAULT_CRITICAL = ["rope_cos", "rope_sin", "cos_pos_embed", "sin_pos_embed"]


def parse_declared(kernel_cu):
    names = set()
    with open(kernel_cu, errors="replace") as f:
        for line in f:
            m = DECL_RE.search(line)
            if m:
                names.add(m.group(1))
    return names


def parse_consumed(task_graph_json):
    d = json.load(open(task_graph_json))
    in_refs, out_refs = defaultdict(int), defaultdict(int)
    in_tt, out_tt = defaultdict(set), defaultdict(set)
    for t in d.get("all_tasks", []):
        tt = t.get("task_type")
        for io, refs, tts in (("inputs", in_refs, in_tt), ("outputs", out_refs, out_tt)):
            for e in (t.get(io) or []):
                bp = e.get("base_ptr") if isinstance(e, dict) else None
                if bp:
                    refs[bp] += 1
                    tts[bp].add(tt)
    return in_refs, out_refs, in_tt, out_tt, len(d.get("all_tasks", []))


def audit_rank(gendir, rank, critical):
    kcu = os.path.join(gendir, f"kernel_{rank}.cu")
    tgj = os.path.join(gendir, f"task_graph_{rank}.json")
    if not (os.path.exists(kcu) and os.path.exists(tgj)):
        return None
    declared = parse_declared(kcu)
    in_refs, out_refs, in_tt, out_tt, ntasks = parse_consumed(tgj)
    dead = sorted(t for t in declared if in_refs.get(t, 0) == 0 and out_refs.get(t, 0) == 0)
    write_only = sorted(t for t in declared if out_refs.get(t, 0) > 0 and in_refs.get(t, 0) == 0)
    crit = {c: {"declared": c in declared, "in_refs": in_refs.get(c, 0),
                "consumers": sorted(x for x in in_tt.get(c, set()) if x is not None),
                "out_refs": out_refs.get(c, 0)} for c in critical}
    return {"rank": rank, "ntasks": ntasks, "n_declared": len(declared),
            "dead": dead, "write_only": write_only, "crit": crit,
            "mtime": os.path.getmtime(tgj)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gendir", nargs="?", default="/work/mirage-perop2/demo/deepseek_v3")
    ap.add_argument("--critical", default=",".join(DEFAULT_CRITICAL))
    ap.add_argument("--ranks", default="")
    a = ap.parse_args()
    critical = [c for c in a.critical.split(",") if c]
    if a.ranks:
        ranks = [int(x) for x in a.ranks.split(",") if x.strip()]
    else:
        ranks = sorted(int(re.search(r"task_graph_(\d+)\.json", p).group(1))
                       for p in glob.glob(os.path.join(a.gendir, "task_graph_*.json")))
    if not ranks:
        print(f"NO ARTIFACTS in {a.gendir}")
        sys.exit(2)
    print(f"# mirage graph-coverage audit  gendir={a.gendir}  ranks={ranks}")
    print(f"# critical (must be consumed as INPUT by >=1 task): {critical}")
    fail = False
    for r in ranks:
        res = audit_rank(a.gendir, r, critical)
        if res is None:
            print(f"\n## rank {r}: MISSING artifacts")
            fail = True
            continue
        mt = datetime.datetime.fromtimestamp(res["mtime"], datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n## rank {r}  tasks={res['ntasks']}  declared_tensors={res['n_declared']}  task_graph_mtime={mt}")
        for c in critical:
            ci = res["crit"][c]
            if ci["declared"] and ci["in_refs"] == 0:
                status = "MISSING-WIRING"
                fail = True
            elif ci["in_refs"] > 0:
                status = "OK"
            else:
                status = "not-declared"
            print(f"  [{status}] {c}: declared={ci['declared']} input_refs={ci['in_refs']} "
                  f"consumer_task_types={ci['consumers']} output_refs={ci['out_refs']}")
        if res["write_only"]:
            print(f"  INFO write-only (produced, 0 input refs) [{len(res['write_only'])}]: {res['write_only'][:20]}")
        if res["dead"]:
            print(f"  INFO dead (declared, 0 refs) [{len(res['dead'])}]: {res['dead'][:20]}")
    print(f"\n# VERDICT: {'FAIL - a critical tensor is declared but never consumed as input (missing wiring)' if fail else 'PASS - every critical tensor is consumed as input by >=1 task'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
