#!/usr/bin/env bash
# capture_cmd.sh
#
# Capture an arbitrary CLI command into the four-file tuple required by
# AGENTS.md "Reproducibility-Grade Evidence":
#
#   ${ART_DIR}/commands/NN-<slug>.cmd       # exact argv (one line)
#   ${ART_DIR}/commands/NN-<slug>.stdout    # captured stdout
#   ${ART_DIR}/commands/NN-<slug>.stderr    # captured stderr
#   ${ART_DIR}/commands/NN-<slug>.exit      # integer exit code
#
# Usage:
#   ART_DIR=experiments/artifacts/cluster-health/<run-id> \
#     bash tools/shared/capture_cmd.sh <slug> -- <cmd> [args...]
#
# The NN ordinal auto-increments from the existing entries under
# ${ART_DIR}/commands/ so the natural sort matches run order. The
# command runs with the helper's exit propagated, so an outer script
# can chain on success/failure.
#
# Per AGENTS.md "Fail Fast, No Silent Fallbacks", the helper exits
# non-zero (rc=2) on misuse without ever swallowing the wrapped command:
# bad usage shows up before the four-file tuple is written.

set -uo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  ART_DIR=<bundle-dir> bash tools/shared/capture_cmd.sh <slug> -- <cmd> [args...]

Required:
  ART_DIR (env)   path to the artifact bundle root (e.g.
                  experiments/artifacts/cluster-health/<run-id>).
                  The helper writes ${ART_DIR}/commands/NN-<slug>.{cmd,stdout,stderr,exit}.

Args:
  slug            kebab-case description; used in the filename after the NN ordinal.
  --              literal separator before the wrapped command argv.
  cmd [args...]   the command to run.
USAGE
}

if [[ "${ART_DIR:-}" == "" ]]; then
  echo "FATAL: ART_DIR env var is required." >&2
  usage
  exit 2
fi

if [[ $# -lt 3 ]]; then
  echo "FATAL: need <slug> -- <cmd> [args...]" >&2
  usage
  exit 2
fi

slug="$1"
shift

if [[ "$1" != "--" ]]; then
  echo "FATAL: missing -- separator before the wrapped command." >&2
  usage
  exit 2
fi
shift

if [[ ! "${slug}" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "FATAL: slug '${slug}' must be kebab-case lowercase alphanumeric." >&2
  exit 2
fi

cmd_dir="${ART_DIR}/commands"
mkdir -p "${cmd_dir}"

# Determine the next NN ordinal by counting existing .cmd files. The
# match is for `NN-...cmd` only so unrelated files in the dir do not
# inflate the count.
shopt -s nullglob
existing=("${cmd_dir}"/[0-9][0-9]-*.cmd)
shopt -u nullglob
next_index=$(( ${#existing[@]} + 1 ))
nn=$(printf '%02d' "${next_index}")
prefix="${cmd_dir}/${nn}-${slug}"

# Record the argv as a single shell-quoted line so a reviewer can
# paste-and-run. printf '%q ' double-quotes whitespace + special chars.
printf '%q ' "$@" > "${prefix}.cmd"
printf '\n' >> "${prefix}.cmd"

# Run the command, capture stdout + stderr separately, propagate exit.
# Use `set +e` locally so the trap-style fail-fast does not eat the
# wrapped command's non-zero exit.
#
# Export the currently-in-flight tuple so any tool that walks
# experiments/artifacts/ (e.g. audit_evidence_bundle.py) can skip the
# self-reference. The audit reports tuples whose `.exit` is missing as
# `incomplete-tuple`; the wrapped command's `.exit` is written only
# AFTER the command returns, so without this hint a capture-wrapped
# audit-of-self always fails. The hint is "<art-dir>:NN-<slug>" so the
# audit only skips this exact tuple in this exact bundle. Per
# AGENTS.md "Self-audit caveat".
export CAPTURE_CMD_IN_FLIGHT="${ART_DIR}:${nn}-${slug}"

set +e
"$@" > "${prefix}.stdout" 2> "${prefix}.stderr"
rc=$?
set -e

unset CAPTURE_CMD_IN_FLIGHT

printf '%d\n' "${rc}" > "${prefix}.exit"

# Print artifact paths to stdout so a chaining caller can pick them up.
# Print rc to stderr so callers redirecting stdout still see the result.
echo "${prefix}.cmd"
echo "${prefix}.stdout"
echo "${prefix}.stderr"
echo "${prefix}.exit"
echo "capture_cmd: nn=${nn} slug=${slug} rc=${rc}" >&2

exit "${rc}"
