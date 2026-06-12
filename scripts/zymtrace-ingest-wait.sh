#!/usr/bin/env bash
# zymtrace-ingest-wait.sh -- shared "wait for ClickHouse ingest, then requery" helper.
#
# Zymtrace profiling frames land in ClickHouse ASYNCHRONOUSLY: the per-pod CUDA
# implant accumulates samples in-process and flushes to the backend on an
# interval (seconds to ~minutes). So a query issued AT or JUST-AFTER the bench
# window can return ZERO ROWS even though capture succeeded -- the data is not
# queryable YET. That empty-now is INGEST LAG, NOT a telemetry gap.
#
# This is the sibling of nsys-validate-capture.sh's "empty != blind spot" gate
# (`docs/METHODOLOGY.md`): there an empty rep is a capture-hygiene
# bug; here an empty query is ingest lag, and the fix is to WAIT for the flush and
# REQUERY before concluding a gap. Canon: . "Zymtrace data
# is not instantaneous (ClickHouse ingest lag)".
#
# Source this file and call the two functions it defines:
#   zym_wait_for_rows <probe_fn> [label]  -- poll a row-count probe until it
#       returns a positive integer (data present) or the attempts are exhausted.
#       <probe_fn> is the NAME of a shell function that echoes a non-negative
#       integer row count (kept as a function, not an eval'd string, to avoid SQL
#       quoting hell). Returns 0 the moment rows appear; non-zero if every attempt
#       came back 0/blank (a genuine gap -- the caller decides what to do).
#   zym_ingest_lag_hint [context]         -- print the standardized "this empty may
#       be ingest lag, requery before calling it a gap" guidance to stderr. Used by
#       capture + verify scripts so the message is identical everywhere.
#
# Env knobs (all optional; tuned so the default poll outlasts the ~60s flush the
# PROFILING-RUNBOOK documents):
#   ZYM_INGEST_WAIT_SEC      per-attempt poll interval, seconds (default 20)
#   ZYM_INGEST_MAX_ATTEMPTS  max poll attempts before giving up (default 6 -> ~120s)
#   ZYM_INGEST_BACKOFF       multiply the interval each attempt (default 1.0 = linear poll)
#   ZYM_INGEST_DISABLE       set to 1 to skip the wait entirely (e.g. backfilling an
#                            old, already-flushed window from retained telemetry)
#
# No `set -e` here on purpose: the wait is advisory. A caller that sources this
# while running the rest of its capture must not abort just because the wait timed
# out -- the existing empty-output check is what decides the real gap.

zym_wait_for_rows() {
  local probe_fn="${1:?zym_wait_for_rows needs the NAME of a probe function that echoes a row count}"
  local label="${2:-zymtrace rows}"
  local wait_sec="${ZYM_INGEST_WAIT_SEC:-20}"
  local max="${ZYM_INGEST_MAX_ATTEMPTS:-6}"
  local backoff="${ZYM_INGEST_BACKOFF:-1.0}"

  if [[ "${ZYM_INGEST_DISABLE:-0}" == "1" ]]; then
    echo "[zym-ingest-wait] ZYM_INGEST_DISABLE=1 -> skipping ingest wait for ${label}" >&2
    return 0
  fi
  if ! declare -F "$probe_fn" >/dev/null 2>&1; then
    echo "[zym-ingest-wait] WARN: probe '$probe_fn' is not a defined function; skipping wait" >&2
    return 0
  fi

  local attempt=1 count interval="$wait_sec"
  while (( attempt <= max )); do
    count="$("$probe_fn" 2>/dev/null | tr -dc '0-9' | head -c 18)"
    count="${count:-0}"
    if [[ "$count" =~ ^[0-9]+$ ]] && (( count > 0 )); then
      echo "[zym-ingest-wait] ${label}: ${count} rows on attempt ${attempt}/${max} -> data present." >&2
      return 0
    fi
    if (( attempt >= max )); then break; fi
    local s="${interval%.*}"; s="${s:-$wait_sec}"
    echo "[zym-ingest-wait] ${label}: 0 rows (attempt ${attempt}/${max}); likely ClickHouse ingest lag, sleeping ${s}s then requerying for the freshest data..." >&2
    sleep "$s"
    interval="$(awk -v i="$interval" -v b="$backoff" 'BEGIN{printf "%.2f", i*b}')"
    (( attempt++ ))
  done
  echo "[zym-ingest-wait] ${label}: still 0 rows after ${max} attempt(s) (~$(( wait_sec * max ))s of polling)." >&2
  echo "[zym-ingest-wait] Past the flush window now -> treat this as a REAL telemetry gap, not ingest lag." >&2
  return 1
}

zym_ingest_lag_hint() {
  local ctx="${1:-}"
  {
    echo "NOTE: an empty zymtrace result is NOT always a telemetry gap. Zymtrace flushes to"
    echo "      ClickHouse asynchronously (~seconds-to-minutes of ingest lag), so a query run at"
    echo "      or just after the bench window can read empty before the data is queryable."
    [[ -n "$ctx" ]] && echo "      Context: ${ctx}"
    echo "      First requery (capture polls automatically via zymtrace-ingest-wait.sh; tune"
    echo "      ZYM_INGEST_WAIT_SEC / ZYM_INGEST_MAX_ATTEMPTS). Re-run capture-sol-window.sh after"
    echo "      the flush interval before concluding empty; query by host=<node>+window (not the"
    echo "      hash-suffixed pod name) and only --ack-telemetry-gap if it STAYS empty afterward."
  } >&2
}
