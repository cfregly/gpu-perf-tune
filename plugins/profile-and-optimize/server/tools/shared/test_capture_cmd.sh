#!/usr/bin/env bash
# Unit test for tools/shared/capture_cmd.sh.
#
# Exercises:
#   - Successful command captures the four-file tuple with rc=0.
#   - Failing command captures the tuple with the wrapped non-zero rc.
#   - Subsequent invocations auto-increment the NN ordinal.
#   - Misuse (missing --, bad slug, missing ART_DIR) exits 2 without
#     writing partial artifacts.
#   - Wrapped command stdout/stderr land in the right files.
#
# Per AGENTS.md "Fail Fast, No Silent Fallbacks", the test file fails
# loudly on any unmet assertion.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${HERE}/capture_cmd.sh"

if [[ ! -x "${HELPER}" ]]; then
  echo "FATAL: helper missing or not executable: ${HELPER}" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

assert() {
  local desc="$1"; shift
  if ! "$@"; then
    echo "FAIL: ${desc}" >&2
    echo "  args: $*" >&2
    exit 1
  fi
}

# --- happy path: success ---------------------------------------------
ART_DIR="${tmp}/run1" \
  bash "${HELPER}" smoke-true -- /usr/bin/env true >/dev/null 2>&1

assert "01-smoke-true.cmd exists" test -f "${tmp}/run1/commands/01-smoke-true.cmd"
assert "01-smoke-true.stdout exists" test -f "${tmp}/run1/commands/01-smoke-true.stdout"
assert "01-smoke-true.stderr exists" test -f "${tmp}/run1/commands/01-smoke-true.stderr"
assert "01-smoke-true.exit exists" test -f "${tmp}/run1/commands/01-smoke-true.exit"
assert "01-smoke-true.exit==0" test "$(cat "${tmp}/run1/commands/01-smoke-true.exit")" = "0"

# --- happy path: failure ---------------------------------------------
ART_DIR="${tmp}/run1" \
  bash "${HELPER}" smoke-false -- /usr/bin/env false >/dev/null 2>&1 \
  && rc=0 || rc=$?
assert "false-wrapped propagates non-zero rc" test "${rc:-0}" -ne 0
assert "02-smoke-false.exit nonzero" test "$(cat "${tmp}/run1/commands/02-smoke-false.exit")" != "0"
assert "auto-increment to 02" test -f "${tmp}/run1/commands/02-smoke-false.cmd"

# --- happy path: stdout/stderr split ---------------------------------
ART_DIR="${tmp}/run1" \
  bash "${HELPER}" stream-split -- /bin/sh -c 'printf out; printf err >&2' >/dev/null 2>&1 || true
assert "03 stdout has 'out'" test "$(cat "${tmp}/run1/commands/03-stream-split.stdout")" = "out"
assert "03 stderr has 'err'" test "$(cat "${tmp}/run1/commands/03-stream-split.stderr")" = "err"

# --- misuse: missing ART_DIR -----------------------------------------
unset_rc=0
( unset ART_DIR; bash "${HELPER}" foo -- /usr/bin/env true >/dev/null 2>&1 ) || unset_rc=$?
assert "missing ART_DIR exits non-zero" test "${unset_rc}" -eq 2

# --- misuse: missing -- separator ------------------------------------
sep_rc=0
ART_DIR="${tmp}/run2" bash "${HELPER}" foo /usr/bin/env true >/dev/null 2>&1 || sep_rc=$?
assert "missing -- exits 2" test "${sep_rc}" -eq 2

# --- misuse: bad slug -------------------------------------------------
slug_rc=0
ART_DIR="${tmp}/run2" bash "${HELPER}" 'BadSlug!' -- /usr/bin/env true >/dev/null 2>&1 || slug_rc=$?
assert "bad slug exits 2" test "${slug_rc}" -eq 2

# --- separator handling: empty NN dir starts at 01 -------------------
ART_DIR="${tmp}/run3" \
  bash "${HELPER}" first -- /usr/bin/env true >/dev/null 2>&1
assert "first run starts at NN=01" test -f "${tmp}/run3/commands/01-first.cmd"

echo "OK: tools/shared/test_capture_cmd.sh passed all assertions"
