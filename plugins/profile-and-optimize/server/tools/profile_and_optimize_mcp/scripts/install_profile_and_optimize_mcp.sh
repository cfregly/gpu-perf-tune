#!/usr/bin/env bash
# Install the repo-local profile_and_optimize MCP server and wire it into local MCP clients.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
VENV="${HOME}/.local/share/profile-and-optimize-mcp-venv"
LOGIN_HOST="${PROFILE_AND_OPTIMIZE_LOGIN_HOST:-${USER:-operator}@192.0.2.10}"
CLIENT_ARGS=()
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh [options]

Options:
  --client NAME       cursor | claude | codex | gemini | antigravity | all
                      May be passed multiple times. Default: cursor.
  --repo-root PATH    mlperf-6.0-training repo root. Default: auto-detected.
  --venv PATH         Python venv path. Default: ~/.local/share/profile-and-optimize-mcp-venv.
  --login-host HOST   PROFILE_AND_OPTIMIZE_LOGIN_HOST value. Default: $USER@192.0.2.10.
  --dry-run           Print config changes without writing client configs.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --client)
      CLIENT_ARGS+=(--client "$2")
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --venv)
      VENV="$2"
      shift 2
      ;;
    --login-host)
      LOGIN_HOST="$2"
      shift 2
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

if [[ ! -f "${REPO_ROOT}/AGENTS.md" || ! -d "${REPO_ROOT}/tools/profile_and_optimize_mcp" ]]; then
  printf 'FATAL: --repo-root is not mlperf-6.0-training: %s\n' "${REPO_ROOT}" >&2
  exit 2
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -e "${REPO_ROOT}/tools/profile_and_optimize_mcp"

CONFIG_ARGS=(
  --repo-root "${REPO_ROOT}"
  --python "${VENV}/bin/python"
  --login-host "${LOGIN_HOST}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  CONFIG_ARGS+=(--dry-run)
fi
if [[ "${#CLIENT_ARGS[@]}" -gt 0 ]]; then
  CONFIG_ARGS+=("${CLIENT_ARGS[@]}")
fi

"${VENV}/bin/python" "${REPO_ROOT}/tools/profile_and_optimize_mcp/scripts/configure_clients.py" "${CONFIG_ARGS[@]}"

printf 'profile_and_optimize MCP installed. Restart your client to load the server.\n'
