#!/usr/bin/env bash
# clean-driver.sh -- workload-agnostic CLEAN single-stream decode driver for
# the inference-decode-step-budget skill (Gate 1).
#
# Drives ONE in-flight request at a time with a tiny prompt + long generation
# + ignore_eos, so the captured window is PURE steady-state decode: no prefill
# bursts, no inter-request lulls. Reports per-round wall-time so TPOT falls out
# directly (feeds the reconciliation gate, Gate 3).
#
# Run this INSIDE the vllm pod (localhost:8000) between /start_profile and
# /stop_profile, e.g.:
#   curl -XPOST localhost:8000/start_profile
#   MODEL=zai-org/GLM-5.1 ROUNDS=1 ./clean-driver.sh
#   curl -XPOST localhost:8000/stop_profile
#
# Why single-stream long-gen and not `vllm bench serve --random-input-len N`:
# the bench driver mixes ISL prefill into the decode profile and leaves gaps
# between requests -> contaminated budget. One long generation = continuous
# decode for the whole window.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:?set MODEL to the served model name, e.g. zai-org/GLM-5.1}"
PROMPT="${PROMPT:-Count slowly:}"      # tiny prompt -> negligible prefill
MAX_TOKENS="${MAX_TOKENS:-3000}"        # long gen -> spans the capture window
CONCURRENCY="${CONCURRENCY:-1}"         # c=1 by default; raise for c=2..8 budgets
ROUNDS="${ROUNDS:-1}"                    # back-to-back generations
ENDPOINT="${ENDPOINT:-/v1/completions}"

req() {
  curl -sS "${BASE_URL}${ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"ignore_eos\":true,\"temperature\":0.0,\"stream\":false}" \
    -o /dev/null -w "http=%{http_code} t=%{time_total}s\n"
}

echo "[clean-driver] model=${MODEL} c=${CONCURRENCY} max_tokens=${MAX_TOKENS} rounds=${ROUNDS} start=$(date -u +%H:%M:%SZ)"
for r in $(seq 1 "${ROUNDS}"); do
  if [ "${CONCURRENCY}" -le 1 ]; then
    OUT=$(req)
  else
    # c>1: launch CONCURRENCY identical long generations in parallel, wait all
    pids=(); outs=()
    for _ in $(seq 1 "${CONCURRENCY}"); do req & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || true; done
    OUT="(${CONCURRENCY} parallel streams)"
  fi
  TOK_PER_S=""
  if [ "${CONCURRENCY}" -le 1 ]; then
    T=$(printf '%s' "$OUT" | sed -n 's/.*t=\([0-9.]*\)s/\1/p')
    if [ -n "$T" ]; then TOK_PER_S=$(python3 -c "print(f'{${MAX_TOKENS}/${T}:.1f} tok/s, TPOT {1000*${T}/${MAX_TOKENS}:.2f} ms/tok')" 2>/dev/null || true); fi
  fi
  echo "[clean-driver] round ${r}/${ROUNDS} ${OUT} ${TOK_PER_S} $(date -u +%H:%M:%SZ)"
done
echo "[clean-driver] done $(date -u +%H:%M:%SZ)"
echo "[clean-driver] NOTE: feed the reported TPOT into decode_budget.py's --tpot-ms for Gate 3 (reconciliation)."
