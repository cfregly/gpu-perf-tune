#!/usr/bin/env bash
# nsys-validate-capture.sh is the mechanical "empty != blind spot" gate.
#
# An empty nsys cuda_gpu_kern_sum on a cudagraph-on vLLM deploy is almost ALWAYS a
# capture-hygiene bug, not a "cudagraph blind spot". This script enforces the 4-point
# gate from the capture hygiene section in docs/METHODOLOGY.md
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

# Run an argv vector locally or through kubectl. User-controlled paths never
# enter shell source text.
run_argv() {
  if [ "$LOCAL" = "1" ]; then
    "$@"
  else
    kubectl -n "${NS:?set NS or LOCAL=1}" exec \
      "${POD:?set POD or LOCAL=1}" -c "${CONT:-vllm}" -- "$@"
  fi
}

fail() {
  printf 'RETRY: %s\n' "$1"
  printf '%s\n' "  -> see docs/METHODOLOGY.md, capture hygiene."
  exit 1
}

case "$LOCAL" in
  0|1) ;;
  *) fail "LOCAL must be 0 or 1." ;;
esac
case "$MIN_REP_MB" in
  ''|*[!0-9]*) fail "MIN_REP_MB must be a positive integer." ;;
esac
[ "$MIN_REP_MB" -gt 0 ] || fail "MIN_REP_MB must be a positive integer."

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
if ! SZ=$(run_argv stat -c %s -- "$REP" 2>/dev/null); then
  if ! SZ=$(run_argv stat -f %z -- "$REP" 2>/dev/null); then
    SZ=0
  fi
fi
SZ=$(printf '%s' "$SZ" | tr -dc '0-9')
SZ="${SZ:-0}"
[ "$SZ" -gt 0 ] || fail "[2/4] rep MISSING or 0 bytes at $REP (capture did not finalize, or wrong path)."
SZ_MB=$(( SZ / 1048576 ))
echo "[2/4] rep-size  ${SZ_MB} MB"
if [ "$SZ_MB" -lt "$MIN_REP_MB" ]; then
  fail "[2/4] rep too small (${SZ_MB} MB < ${MIN_REP_MB} MB) -> idle/untrafficked window (Check: was a bench DRIVING c>=64 load during [delay,delay+duration]?). RETRY the capture with driven in-window traffic; do NOT run stats on this rep."
fi

# ---- Check 4: sqlite KERNEL row count (the decisive probe) ----
if [ -z "${NSYS:-}" ]; then
  NSYS=$(run_argv find /opt/nvidia/nsight-systems -type f -path '*/bin/nsys' -print -quit 2>/dev/null || true)
fi
[ -n "$NSYS" ] || fail "[4/4] nsys binary not found (set NSYS=...)."
SQ="/tmp/nsys-validate-$$.sqlite"
echo "[3/4] exporting sqlite (large reps take minutes; KERNEL table is what matters)..."
run_argv "$NSYS" export --type sqlite --force-overwrite=true \
  "--output=$SQ" "$REP" >/dev/null 2>&1 || true
if ! KROWS=$(run_argv python3 -c \
  'import sqlite3, sys; print(sqlite3.connect(sys.argv[1]).execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0])' \
  "$SQ" 2>/dev/null); then
  KROWS=ERR
fi
KROWS=$(printf '%s' "$KROWS" | tr -dc '0-9A-Za-z')
run_argv rm -f -- "$SQ" 2>/dev/null || true
case "$KROWS" in
  ERR|"") fail "[4/4] could not read CUPTI_ACTIVITY_KIND_KERNEL (export failed or no python3/sqlite). Re-run export manually before concluding empty.";;
  0)      fail "[4/4] KERNEL rows = 0 EVEN WITH a >=${MIN_REP_MB}MB rep. Re-verify Check 1 (--cuda-graph-trace=node) + Check 2 (driven traffic). Only escalate to a genuine tooling limit after all 4 hold and it is still 0.";;
  *)      echo "[4/4] KERNEL rows = ${KROWS}  -> capture has per-kernel data; NOT a blind spot.";;
esac

echo "== PASS: capture is valid for per-kernel analysis (kernels=${KROWS}, ${SZ_MB} MB) =="
