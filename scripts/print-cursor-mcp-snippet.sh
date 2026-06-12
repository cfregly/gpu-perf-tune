#!/usr/bin/env bash
# Print the `profile_and_optimize` MCP server block for ~/.cursor/mcp.json, with the venv
# path resolved against THIS checkout's bundled server.
#
# Cursor (unlike Claude Code) does not read the plugin's .mcp.json; the operator
# must add the `profile_and_optimize` server to ~/.cursor/mcp.json by hand. This script
# emits a ready-to-paste block so that step isn't guesswork. It is READ-ONLY:
# it prints to stdout and never edits ~/.cursor/mcp.json (use cursor-mcp-doctor.sh
# --fix for that).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVER_ROOT="${REPO_ROOT}/plugins/profile-and-optimize/server"
VENV_PY="${SERVER_ROOT}/.venv/bin/python"
LOGIN_HOST="${PROFILE_AND_OPTIMIZE_LOGIN_HOST:-${USER:-operator}@192.0.2.10}"

if [[ ! -x "${VENV_PY}" ]]; then
  printf '[warn] %s not found; run `make bootstrap` (or `bash %s/install.sh`) first.\n' \
    "${VENV_PY}" "${SERVER_ROOT}" >&2
fi

cat <<EOF
# Add this to the "mcpServers" object in ~/.cursor/mcp.json, then reload Cursor.
# Resolved against this checkout: ${SERVER_ROOT}
    "profile_and_optimize": {
      "type": "stdio",
      "command": "${VENV_PY}",
      "args": ["-m", "profile_and_optimize_mcp", "serve"],
      "env": {
        "PROFILE_AND_OPTIMIZE_REPO_ROOT": "${SERVER_ROOT}",
        "PROFILE_AND_OPTIMIZE_LOGIN_HOST": "${LOGIN_HOST}"
      }
    }
EOF
