#!/usr/bin/env bash
# Install the bundled MCP server and wire it into supported local clients.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
VENV="${HOME}/.local/share/profile-and-optimize-mcp-venv"
PYTHON="${PYTHON:-python3}"
REGISTRATION="auto"
CLIENT_ARGS=()
CONFIG_ARGS=()
INSTALL_ARGS=()
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh [options]

Installs the complete bundled server, then registers it with selected clients.

Options:
  --client NAME       cursor | claude | codex | gemini | antigravity | all
                      May be passed multiple times. Default: cursor.
  --repo-root PATH    Bundled server root. Default: auto-detected server directory.
  --venv PATH         Python venv path. Default: ~/.local/share/profile-and-optimize-mcp-venv.
  --python PATH       Python interpreter used to create the venv. Default: python3.
  --registration MODE auto | file. Default: auto.
                      Auto uses Claude or Codex CLI registration for a new entry,
                      with atomic config-file fallback. File skips client CLIs.
  --cursor-config PATH       Override ~/.cursor/mcp.json.
  --claude-config PATH       Override ~/.claude.json.
  --codex-config PATH        Override ~/.codex/config.toml.
  --gemini-config PATH       Override ~/.gemini/settings.json.
  --antigravity-config PATH  Override the Antigravity MCP config path. Default:
                             ~/.gemini/config/mcp_config.json.
  --full              Install report-renderer and leaderboard extras.
  --with-dev          Install test and development extras.
  --dry-run           Print install and registration actions. Write nothing,
                      create no venv, install no package, and run no client CLI.
  -h, --help          Show this help.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
    printf 'missing value for %s\n' "$1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --client)
      require_value "$@"
      case "$2" in
        cursor|claude|codex|gemini|antigravity|all) ;;
        *)
          printf 'invalid client: %s\n' "$2" >&2
          usage >&2
          exit 2
          ;;
      esac
      CLIENT_ARGS+=(--client "$2")
      shift 2
      ;;
    --repo-root)
      require_value "$@"
      SERVER_ROOT="$2"
      shift 2
      ;;
    --venv)
      require_value "$@"
      VENV="$2"
      shift 2
      ;;
    --python)
      require_value "$@"
      PYTHON="$2"
      shift 2
      ;;
    --registration)
      require_value "$@"
      REGISTRATION="$2"
      shift 2
      ;;
    --cursor-config|--claude-config|--codex-config|--gemini-config|--antigravity-config)
      require_value "$@"
      CONFIG_ARGS+=("$1" "$2")
      shift 2
      ;;
    --full|--with-dev)
      INSTALL_ARGS+=("$1")
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

if [[ "${REGISTRATION}" != "auto" && "${REGISTRATION}" != "file" ]]; then
  printf 'invalid --registration mode: %s. Expected auto or file.\n' "${REGISTRATION}" >&2
  exit 2
fi

if [[ ! -f "${SERVER_ROOT}/pyproject.toml" || ! -f "${SERVER_ROOT}/mcp_surface.py" || ! -d "${SERVER_ROOT}/tools/profile_and_optimize_mcp" || ! -f "${SERVER_ROOT}/install.sh" ]]; then
  printf 'FATAL: bundled server root is incomplete: %s\n' "${SERVER_ROOT}" >&2
  printf 'Expected pyproject.toml, mcp_surface.py, install.sh, and tools/profile_and_optimize_mcp/.\n' >&2
  exit 2
fi

CONFIG_COMMAND_ARGS=(
  --repo-root "${SERVER_ROOT}"
  --python "${VENV}/bin/python"
  --registration "${REGISTRATION}"
  "${CONFIG_ARGS[@]}"
)
if [[ "${#CLIENT_ARGS[@]}" -gt 0 ]]; then
  CONFIG_COMMAND_ARGS+=("${CLIENT_ARGS[@]}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%s\n' '=== [dry-run] bundled server install ==='
  bash "${SERVER_ROOT}/install.sh" \
    --venv "${VENV}" \
    --python "${PYTHON}" \
    "${INSTALL_ARGS[@]}" \
    --dry-run

  printf '\n%s\n' '=== [dry-run] client registration ==='
  "${PYTHON}" "${SCRIPT_DIR}/configure_clients.py" \
    "${CONFIG_COMMAND_ARGS[@]}" \
    --dry-run
  printf '\n%s\n' '[done] dry run complete. No files or client settings were changed.'
  exit 0
fi

bash "${SERVER_ROOT}/install.sh" \
  --venv "${VENV}" \
  --python "${PYTHON}" \
  "${INSTALL_ARGS[@]}"

"${VENV}/bin/python" "${SCRIPT_DIR}/configure_clients.py" "${CONFIG_COMMAND_ARGS[@]}"

printf '\n%s\n' "[done] profile_and_optimize MCP installed at ${VENV}"
printf '%s\n' 'Restart or reload each configured client to load the server.'
