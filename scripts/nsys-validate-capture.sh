#!/usr/bin/env bash
# nsys-validate-capture.sh — the mechanical "empty != blind spot" gate.
#
# An empty nsys cuda_gpu_kern_sum on a cudagraph-on vLLM deploy is almost ALWAYS a
# capture-hygiene bug, not a "cudagraph blind spot". This script enforces the 4-point
# gate from docs/METHODOLOGY.md capture hygiene + `docs/METHODOLOGY.md`
# BEFORE anyone concludes nsys cannot resolve kernels on this stack.
#
# Worked failure mode: an idle capture window yields an empty kernel table and reads
# as "blocked"; a driven window on the same stack resolves tens of millions of kernels.
#
# Usage:
#   # In-pod (rep + nsys both inside an analyzer/serving pod):
#   NS=<namespace> POD=<pod> CONT=<container> REP=/models/capture.nsys-rep \
#     bash nsys-validate-capture.sh
#   # Local (rep + nsys on this host):
#   REP=./capture.nsys-rep NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys \
#     LOCAL=1 bash nsys-validate-capture.sh
#
# Env:
#   REP            (required) path to the .nsys-rep (in-pod path if NS/POD set, else local)
#   NS, POD, CONT  exec target (omit + set LOCAL=1 to run against a local rep)
#   NSYS           nsys binary path (default: autodetect /opt/nvidia/nsight-systems/*/bin/nsys)
#   MIN_REP_MB     rep-size floor in MB (default 10; real c>=64 reps are 100s of MB-GB)
#   DEPLOY_ARGS    optional: the live deploy nsys argv (string) to check for --cuda-graph-trace=node
#
# Exit 0 = PASS (rep has kernel data; safe to analyze). Exit 1 = RETRY (with reason).
set -uo pipefail

REP="${REP:?set REP=/path/to/capture.nsys-rep}"
MIN_REP_MB="${MIN_REP_MB:-10}"
LOCAL="${LOCAL:-0}"

# run a command either locally or via kubectl exec into the pod
run() {
  if [ "$LOCAL" = "1" ]; then bash -lc "$1"
  else kubectl -n "${NS:?set NS or LOCAL=1}" exec "${POD:?set POD or LOCAL=1}" -c "${CONT:-vllm}" -- bash -lc "$1"; fi
}

fail() { echo "RETRY: $1"; echo "  -> see docs/METHODOLOGY.md 'capture hygiene' + `docs/METHODOLOGY.md`"; exit 1; }

echo "== nsys capture-validation gate =="
echo "rep=$REP  min_rep_mb=$MIN_REP_MB  mode=$([ "$LOCAL" = 1 ] && echo local || echo "exec $NS/$POD")"

# ---- Check 1: --cuda-graph-trace=node (best-effort; pass DEPLOY_ARGS or it warns) ----
if [ -n "${DEPLOY_ARGS:-}" ]; then
  case "$DEPLOY_ARGS" in
    *"--cuda-graph-trace=node"*) echo "[1/4] flag      OK   (--cuda-graph-trace=node present)";;
    *) fail "[1/4] flag MISSING: nsys argv has no --cuda-graph-trace=node -> graph-resident kernels are opaque GRAPH_TRACE at c>=64 (empty kern_sum). Re-capture with the flag.";;
  esac
else
  echo "[1/4] flag      WARN (DEPLOY_ARGS not provided; confirm --cuda-graph-trace=node was in the nsys argv)"
fi

# ---- Check 2 (size) + Check 3 (rep exists) ----
SZ=$(run "stat -c %s '$REP' 2>/dev/null || echo 0" | tr -dc '0-9')
SZ="${SZ:-0}"
[ "$SZ" -gt 0 ] || fail "[2/4] rep MISSING or 0 bytes at $REP (capture did not finalize, or wrong path)."
SZ_MB=$(( SZ / 1048576 ))
echo "[2/4] rep-size  ${SZ_MB} MB"
if [ "$SZ_MB" -lt "$MIN_REP_MB" ]; then
  fail "[2/4] rep too small (${SZ_MB} MB < ${MIN_REP_MB} MB) -> idle/untrafficked window (Check: was a bench DRIVING c>=64 load during [delay,delay+duration]?). RETRY the capture with driven in-window traffic; do NOT run stats on this rep."
fi

# ---- Check 4: sqlite KERNEL row count (the decisive probe) ----
NSYS="${NSYS:-$(run "ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1")}"
[ -n "$NSYS" ] || fail "[4/4] nsys binary not found (set NSYS=...)."
SQ="/tmp/nsys-validate-$$.sqlite"
echo "[3/4] exporting sqlite (large reps take minutes; KERNEL table is what matters)..."
run "$NSYS export --type sqlite --force-overwrite=true --output='$SQ' '$REP' >/dev/null 2>&1 || true"
KROWS=$(run "python3 -c \"import sqlite3;print(sqlite3.connect('$SQ').execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0])\" 2>/dev/null || echo ERR" | tr -dc '0-9A-Za-z')
run "rm -f '$SQ' 2>/dev/null || true"
case "$KROWS" in
  ERR|"") fail "[4/4] could not read CUPTI_ACTIVITY_KIND_KERNEL (export failed or no python3/sqlite). Re-run export manually before concluding empty.";;
  0)      fail "[4/4] KERNEL rows = 0 EVEN WITH a >=${MIN_REP_MB}MB rep. Re-verify Check 1 (--cuda-graph-trace=node) + Check 2 (driven traffic). Only escalate to a genuine tooling limit after all 4 hold and it is still 0.";;
  *)      echo "[4/4] KERNEL rows = ${KROWS}  -> capture has per-kernel data; NOT a blind spot.";;
esac

echo "== PASS: capture is valid for per-kernel analysis (kernels=${KROWS}, ${SZ_MB} MB) =="
