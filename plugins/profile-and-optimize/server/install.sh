#!/usr/bin/env bash
# Install the bundled profile_and_optimize MCP server into a venv that an MCP
# client can launch. Adapted from the
# tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh.
#
# The default install path is .venv under this server directory. Pass --venv
# to override it.

set -euo pipefail

# Resolve script + server root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="${SCRIPT_DIR}"

# Defaults.
VENV="${SERVER_ROOT}/.venv"
DRY_RUN=0
WITH_DEV=0
WITH_FULL=0
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: server/install.sh [options]

Installs the bundled profile_and_optimize MCP server in editable mode into a venv
that a supported MCP client can launch.

Options:
  --venv PATH         Python venv path. Default: <server>/.venv
  --python PATH       Python interpreter. Default: python3 from PATH.
  --with-dev          Also install the `dev` extras required by the default
                      pytest target, plus pyright, ruff, and pre-commit.
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
client configuration must provide this environment variable.
EOF
}

require_value() {
  local option="$1"
  local value="${2-}"
  if [[ -z "${value}" || "${value}" == -* ]]; then
    printf 'FATAL: %s requires a value\n' "${option}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      require_value "$1" "${2-}"
      VENV="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2-}"
      PYTHON="$2"
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
if [[ ! -f "${SERVER_ROOT}/pyproject.toml" || ! -d "${SERVER_ROOT}/tools" || ! -f "${SERVER_ROOT}/mcp_surface.py" ]]; then
  printf 'FATAL: server tree looks wrong at %s (missing pyproject.toml / tools/ / mcp_surface.py)\n' "${SERVER_ROOT}" >&2
  exit 2
fi

if [[ ! -d "${SERVER_ROOT}/tools/profile_and_optimize_mcp" ]]; then
  printf 'FATAL: bundled profile_and_optimize_mcp package not found at %s/tools/profile_and_optimize_mcp\n' "${SERVER_ROOT}" >&2
  exit 2
fi

# Compose the extras list from --with-dev and --full. Empty means the base
# runtime dependencies only.
EXTRAS=""
if [[ "${WITH_DEV}" -eq 1 ]]; then
  EXTRAS="${EXTRAS:+${EXTRAS},}dev"
fi
if [[ "${WITH_FULL}" -eq 1 ]]; then
  EXTRAS="${EXTRAS:+${EXTRAS},}perf_tune_report,leaderboard"
fi
# --full pins the AA-workload deps via constraints-aa.txt so installs resolve
# the same versions.
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
  if [[ "${#CONSTRAINTS_ARG[@]}" -gt 0 ]]; then
    printf '[dry-run] would pip install -e %s %s\n' "${SERVER_SPEC}" "${CONSTRAINTS_ARG[*]}"
  else
    printf '[dry-run] would pip install -e %s\n' "${SERVER_SPEC}"
  fi
  printf '[dry-run] would pip install -e %s\n' "${SERVER_ROOT}/tools/profile_and_optimize_mcp"
  printf '[dry-run] launch envelope: %s -m profile_and_optimize_mcp serve (PROFILE_AND_OPTIMIZE_REPO_ROOT=%s)\n' \
    "${VENV}/bin/python" "${SERVER_ROOT}"
  exit 0
fi

# Create the venv. Use --copies so the interpreter remains launchable if a
# client or installer copies the server directory without preserving symlinks.
"${PYTHON}" -m venv --copies "${VENV}"
"${VENV}/bin/python" -m pip --version >/dev/null

# Install the 8 CLI libraries and tools namespace from server/pyproject.toml,
# then the profile_and_optimize_mcp package from server/tools/profile_and_optimize_mcp/pyproject.toml.
# EXTRAS (composed above) adds dev (--with-dev) and/or perf_tune_report+leaderboard
# (--full) so cache-side tests / the perftunereport renderer don't need a
# side-channel pip step.
"${VENV}/bin/python" -m pip install -e "${SERVER_SPEC}" "${CONSTRAINTS_ARG[@]+${CONSTRAINTS_ARG[@]}}"
"${VENV}/bin/python" -m pip install -e "${SERVER_ROOT}/tools/profile_and_optimize_mcp"

# Smoke-check: derive the MCP surface and confirm it matches the canonical
# count constants in mcp_surface.py. The `counts` subcommand raises an
# AssertionError on drift, so a non-zero exit here means a real bug.
if ! "${VENV}/bin/python" "${SERVER_ROOT}/mcp_surface.py" counts; then
  printf 'FATAL: mcp_surface.py canonical-count verification failed\n' >&2
  printf '       inspect with: %s/bin/python %s/mcp_surface.py counts\n' \
    "${VENV}" "${SERVER_ROOT}" >&2
  exit 1
fi
printf '[ok] mcp_surface.py canonical-count verification passed\n'

# Confirm the perftunereport CLI is on PATH and report_smoke renders a PDF
# from the bundled synthetic fixture. A minimal install omits these optional
# renderer dependencies.
if "${VENV}/bin/python" -c 'import matplotlib' >/dev/null 2>&1; then
  SMOKE_PDF="$(mktemp -t perftunereport-smoke-XXXXXX.pdf)"
  if ! "${VENV}/bin/perftunereport" report_smoke --out "${SMOKE_PDF}" >/dev/null 2>&1; then
    rm -f "${SMOKE_PDF}"
    printf 'FATAL: perftunereport report_smoke failed\n' >&2
    printf '       inspect with: %s/bin/perftunereport report_smoke --out /tmp/smoke.pdf\n' \
      "${VENV}" >&2
    exit 1
  fi
  if [[ ! -s "${SMOKE_PDF}" ]]; then
    rm -f "${SMOKE_PDF}"
    printf 'FATAL: perftunereport report_smoke produced no PDF data\n' >&2
    exit 1
  fi
  SIZE="$(wc -c <"${SMOKE_PDF}" | tr -d ' ')"
  printf '[ok] perftunereport report_smoke produced %s bytes\n' "${SIZE}"
  rm -f "${SMOKE_PDF}"
else
  if [[ "${WITH_DEV}" -eq 1 || "${WITH_FULL}" -eq 1 ]]; then
    printf 'FATAL: selected extras did not install the report renderer dependencies\n' >&2
    exit 1
  fi
  printf '[note] Optional report rendering is not installed. Re-run with --full when needed.\n'
fi

printf '\n[done] profile_and_optimize MCP server installed at %s\n' "${VENV}"
printf '       launch: %s -m profile_and_optimize_mcp serve\n' "${VENV}/bin/python"
printf '       env:    PROFILE_AND_OPTIMIZE_REPO_ROOT=%s\n' "${SERVER_ROOT}"
printf '       Restart your configured MCP client to load the server.\n'
if [[ "${WITH_DEV}" -eq 1 ]]; then
  printf '       dev extras installed: pytest available at %s/bin/pytest\n' "${VENV}"
fi
