#!/usr/bin/env bash
# Cached wrapper around `claude plugin validate`.
#
# Why: `claude plugin validate` is deterministic on the plugin-manifest tree
# (plugin.json + .mcp.json + SKILL.md frontmatter). It is the slowest of the
# four gates in `make -j4 all` (~1.5s) and rarely needs to re-run between
# commits. This script hashes the manifest tree, looks up
# `.cache/plugin-validate-<sha>.ok`, and skips the real invocation on a hit.
#
# Cache invalidation is automatic: any byte change in plugin.json, .mcp.json,
# or any SKILL.md content yields a different SHA -> cache miss -> real
# validation. Stale cache entries can be wiped via `rm -rf .cache/`.
#
# Environment overrides:
#   CLAUDE_PLUGIN_VALIDATE_CMD  command to invoke instead of `claude plugin
#                               validate ...`. CI sets this to a lightweight
#                               python3 -m json.tool sanity check because the
#                               `claude` CLI is not on the runner.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PLUGIN_DIR="${REPO_ROOT}/plugins/profile-and-optimize"
CACHE_DIR="${REPO_ROOT}/.cache"

mkdir -p "${CACHE_DIR}"

# Compute a stable SHA over the manifest-tree files: plugin.json, .mcp.json,
# every SKILL.md, marketplace.json. Use the sorted-find -> shasum pattern so
# the same set of files in the same content always yields the same SHA, and
# any byte change yields a different SHA.
SHA="$(
  {
    cat "${PLUGIN_DIR}/.claude-plugin/plugin.json" 2>/dev/null || true
    cat "${PLUGIN_DIR}/.mcp.json" 2>/dev/null || true
    cat "${REPO_ROOT}/.claude-plugin/marketplace.json" 2>/dev/null || true
    find "${PLUGIN_DIR}/skills" -name SKILL.md -print0 2>/dev/null | sort -z | xargs -0 cat 2>/dev/null || true
  } | shasum -a 256 | awk '{print $1}' | cut -c1-16
)"

CACHE_FILE="${CACHE_DIR}/plugin-validate-${SHA}.ok"

if [[ -f "${CACHE_FILE}" ]]; then
  TS="$(head -1 "${CACHE_FILE}" 2>/dev/null || echo 'unknown')"
  printf '[cached] claude plugin validate (manifest sha=%s; validated at %s; bypass with `make validate-uncached`)\n' \
    "${SHA}" "${TS}"
  exit 0
fi

# Cache miss: run the real validation.
if [[ -n "${CLAUDE_PLUGIN_VALIDATE_CMD:-}" ]]; then
  printf '[validate-cached] cache miss (sha=%s); running override: %s\n' "${SHA}" "${CLAUDE_PLUGIN_VALIDATE_CMD}"
  if eval "${CLAUDE_PLUGIN_VALIDATE_CMD}"; then
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${CACHE_FILE}"
    printf 'manifest_sha=%s\n' "${SHA}" >> "${CACHE_FILE}"
    printf 'validator=%s\n' "${CLAUDE_PLUGIN_VALIDATE_CMD}" >> "${CACHE_FILE}"
  else
    exit_code=$?
    printf '[FAIL] override validator failed (exit=%d); cache NOT written\n' "${exit_code}" >&2
    exit "${exit_code}"
  fi
elif command -v claude >/dev/null 2>&1; then
  printf '[validate-cached] cache miss (sha=%s); running: claude plugin validate %s\n' "${SHA}" "${PLUGIN_DIR}"
  if claude plugin validate "${PLUGIN_DIR}"; then
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${CACHE_FILE}"
    printf 'manifest_sha=%s\n' "${SHA}" >> "${CACHE_FILE}"
    printf 'validator=claude plugin validate\n' >> "${CACHE_FILE}"
  else
    exit_code=$?
    printf '[FAIL] claude plugin validate failed (exit=%d); cache NOT written\n' "${exit_code}" >&2
    exit "${exit_code}"
  fi
else
  printf 'FATAL: `claude` CLI not on PATH and CLAUDE_PLUGIN_VALIDATE_CMD unset.\n' >&2
  printf '       Install Claude Code or export CLAUDE_PLUGIN_VALIDATE_CMD to a substitute (e.g.\n' >&2
  printf '       `python3 -m json.tool < %s/.claude-plugin/plugin.json`).\n' "${PLUGIN_DIR}" >&2
  exit 2
fi
