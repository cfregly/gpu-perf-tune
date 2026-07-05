#!/usr/bin/env bash
# Install the bundled profile_and_optimize MCP server into a venv that Claude Code
# (or Cursor / Codex CLI / Gemini CLI / Antigravity) can launch via the
# plugin's .mcp.json entry. Adapted from the upstream
# tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh.
#
# Default install path: ${CLAUDE_PLUGIN_ROOT}/server/.venv, where
# CLAUDE_PLUGIN_ROOT is the directory containing the plugin's
# .claude-plugin/plugin.json. Pass --venv to override.

set -euo pipefail

# Resolve script + server root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="${SCRIPT_DIR}"

# Defaults.
VENV="${SERVER_ROOT}/.venv"
LOGIN_HOST="${PROFILE_AND_OPTIMIZE_LOGIN_HOST:-${USER:-operator}@192.0.2.10}"
DRY_RUN=0
WITH_DEV=0
WITH_FULL=0
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: server/install.sh [options]

Installs the bundled profile_and_optimize MCP server in editable mode into a venv,
ready for the plugin's .mcp.json to launch.

Options:
  --venv PATH         Python venv path. Default: <server>/.venv
  --python PATH       Python interpreter. Default: python3 from PATH.
  --login-host HOST   PROFILE_AND_OPTIMIZE_LOGIN_HOST value used by profile_and_optimize_mcp at
                      runtime. Default: $USER@192.0.2.10.
  --with-dev          Also install the `dev` extras (pytest, pyright, ruff,
                      pre-commit, pytest-xdist). Lets you run
                      `.venv/bin/python -m pytest tools/` without a
                      separate `pip install pytest` side-channel.
  --full              Also install the `perf_tune_report` + `leaderboard` extras
                      (matplotlib, pandas, pyarrow, boto3, tiktoken, openpyxl)
                      so the inference-perf-tune-report / Speed-of-Light report
                      pages render without a separate pip step. Uses
                      constraints-aa.txt to pin the AA-workload deps.
  --dry-run           Print actions without writing anything.
  -h, --help          Show this help.

After install, the bundled server is reachable at:
    <server>/.venv/bin/python -m profile_and_optimize_mcp serve

with PROFILE_AND_OPTIMIZE_REPO_ROOT=<server> set in the launch environment. The
plugin's .mcp.json already encodes this.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --login-host)
      LOGIN_HOST="$2"
      shift 2
      ;;
    --with-dev)
      WITH_DEV=1
      shift
      ;;
    --full)
      WITH_FULL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown arg: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Sanity-check the server tree.
if [[ ! -f "${SERVER_ROOT}/CLAUDE.md" || ! -d "${SERVER_ROOT}/tools" || ! -f "${SERVER_ROOT}/mcp_surface.py" ]]; then
  printf 'FATAL: server tree looks wrong at %s (missing CLAUDE.md / tools/ / mcp_surface.py)\n' "${SERVER_ROOT}" >&2
  exit 2
fi

if [[ ! -d "${SERVER_ROOT}/tools/profile_and_optimize_mcp" ]]; then
  printf 'FATAL: bundled profile_and_optimize_mcp package not found at %s/tools/profile_and_optimize_mcp\n' "${SERVER_ROOT}" >&2
  exit 2
fi

# Compose the extras list from --with-dev / --full. Empty means the minimal
# runtime install (mcp + PyYAML only).
EXTRAS=""
if [[ "${WITH_DEV}" -eq 1 ]]; then
  EXTRAS="${EXTRAS:+${EXTRAS},}dev"
fi
if [[ "${WITH_FULL}" -eq 1 ]]; then
  EXTRAS="${EXTRAS:+${EXTRAS},}perf_tune_report,leaderboard"
fi
# --full pins the AA-workload deps via constraints-aa.txt so the team resolves
# identical versions.
CONSTRAINTS_ARG=()
if [[ "${WITH_FULL}" -eq 1 && -f "${SERVER_ROOT}/constraints-aa.txt" ]]; then
  CONSTRAINTS_ARG=(-c "${SERVER_ROOT}/constraints-aa.txt")
fi
if [[ -n "${EXTRAS}" ]]; then
  SERVER_SPEC="${SERVER_ROOT}[${EXTRAS}]"
else
  SERVER_SPEC="${SERVER_ROOT}"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '[dry-run] would create venv: %s\n' "${VENV}"
  printf '[dry-run] would pip install -e %s %s\n' "${SERVER_SPEC}" "${CONSTRAINTS_ARG[*]+${CONSTRAINTS_ARG[*]}}"
  printf '[dry-run] would pip install -e %s\n' "${SERVER_ROOT}/tools/profile_and_optimize_mcp"
  printf '[dry-run] launch envelope: %s -m profile_and_optimize_mcp serve (PROFILE_AND_OPTIMIZE_REPO_ROOT=%s, PROFILE_AND_OPTIMIZE_LOGIN_HOST=%s)\n' \
    "${VENV}/bin/python" "${SERVER_ROOT}" "${LOGIN_HOST}"
  exit 0
fi

# Create the venv. Use --copies so the interpreter is a real file rather than a
# symlink: the plugin cache is populated by a directory-marketplace tree copy of
# this server dir, and symlinked interpreters get dropped on copy (-> spawn
# .venv/bin/python ENOENT). --copies keeps the bundled venv launchable after copy.
"${PYTHON}" -m venv --copies "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip wheel setuptools

# Install the 22 stubs + tools namespace from server/pyproject.toml,
# then the profile_and_optimize_mcp package from server/tools/profile_and_optimize_mcp/pyproject.toml.
# EXTRAS (composed above) adds dev (--with-dev) and/or perf_tune_report+leaderboard
# (--full) so cache-side tests / the perftunereport renderer don't need a
# side-channel pip step.
"${VENV}/bin/python" -m pip install -e "${SERVER_SPEC}" "${CONSTRAINTS_ARG[@]+${CONSTRAINTS_ARG[@]}}"
"${VENV}/bin/python" -m pip install -e "${SERVER_ROOT}/tools/profile_and_optimize_mcp"

# Smoke-check: derive the MCP surface and confirm it matches the canonical
# count constants in mcp_surface.py. The `counts` subcommand raises an
# AssertionError on drift, so a non-zero exit here means a real bug.
if "${VENV}/bin/python" "${SERVER_ROOT}/mcp_surface.py" counts; then
  printf '[ok] mcp_surface.py canonical-counts verification passed\n'
else
  printf '[warn] mcp_surface.py canonical-counts verification FAILED. Run\n'
  printf '       %s/bin/python %s/mcp_surface.py counts\n' "${VENV}" "${SERVER_ROOT}"
  printf '       to inspect; either a library was added without updating _TOTAL_CONTRACT_TOOLS\n'
  printf '       in mcp_surface.py, or LIBRARIES drifted away from _TOTAL_LIBRARIES.\n' >&2
fi

# v1.13.0 smoke: confirm perftunereport CLI is on PATH and report_smoke renders a PDF
# from the bundled synthetic fixture. Skipped if matplotlib is missing.
if "${VENV}/bin/python" -c 'import matplotlib' >/dev/null 2>&1; then
  SMOKE_PDF="$(mktemp -t perftunereport-smoke-XXXXXX.pdf)"
  if "${VENV}/bin/perftunereport" report_smoke --out "${SMOKE_PDF}" >/dev/null 2>&1; then
    SIZE="$(wc -c <"${SMOKE_PDF}" | tr -d ' ')"
    printf '[ok] perftunereport report_smoke produced %s bytes at %s\n' "${SIZE}" "${SMOKE_PDF}"
    rm -f "${SMOKE_PDF}"
  else
    printf '[warn] perftunereport report_smoke failed; run %s/bin/perftunereport report_smoke --out /tmp/smoke.pdf to inspect.\n' \
      "${VENV}" >&2
  fi
else
  printf '\n'
  printf '  ====================================================================\n'
  printf '  [ACTION NEEDED] perf_tune_report renderer deps (matplotlib/pandas/pyarrow)\n'
  printf '  are NOT installed, so the inference-perf-tune-report skill + the\n'
  printf '  Speed-of-Light report pages will fail. Enable them with EITHER:\n'
  printf '      bash %s/install.sh --full\n' "${SERVER_ROOT}"
  printf '      %s/bin/pip install -e "%s[perf_tune_report,leaderboard]" -c %s/constraints-aa.txt\n' \
    "${VENV}" "${SERVER_ROOT}" "${SERVER_ROOT}"
  printf '  (constraints-aa.txt exact-pins the AA-workload deps so the team\n'
  printf '  resolves identical versions.)\n'
  printf '  ====================================================================\n'
fi

printf '\n[done] profile_and_optimize MCP server installed at %s\n' "${VENV}"
printf '       launch: %s -m profile_and_optimize_mcp serve\n' "${VENV}/bin/python"
printf '       env:    PROFILE_AND_OPTIMIZE_REPO_ROOT=%s PROFILE_AND_OPTIMIZE_LOGIN_HOST=%s\n' "${SERVER_ROOT}" "${LOGIN_HOST}"
printf '       The plugin .mcp.json already encodes this; restart your client to load the server.\n'
if [[ "${WITH_DEV}" -eq 1 ]]; then
  printf '       dev extras installed: pytest available at %s/bin/pytest\n' "${VENV}"
fi
