#!/usr/bin/env bash
# One-shot bootstrap for the Cursor / dev-clone path.
#
# Collapses the multi-step setup ritual into one command:
#   1. Install the bundled profile_and_optimize MCP server venv (server/install.sh).
#   2. Symlink every skill into ~/.cursor/skills/ (refresh-symlinks).
#   3. Print the ~/.cursor/mcp.json `profile_and_optimize` block (read-only; never edits the file).
#
# Claude-Code-only operators don't need this — `claude plugin install` +
# `bash <cache>/server/install.sh` is enough. This is for the Cursor path,
# where skills + the MCP server entry are wired up by hand otherwise.
#
# Flags are passed straight through to server/install.sh, so:
#   bash scripts/bootstrap.sh --full         # also installs perf_tune_report renderer deps
#   bash scripts/bootstrap.sh --with-dev     # also installs pytest etc.
#   bash scripts/bootstrap.sh --full --with-dev

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVER_ROOT="${REPO_ROOT}/plugins/profile-and-optimize/server"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [install.sh options]

Runs server/install.sh, then refreshes the Cursor skill symlinks, then prints
the ~/.cursor/mcp.json `profile_and_optimize` block. All options are forwarded to
server/install.sh; the common ones:

  --full        Also install the perf_tune_report renderer deps (matplotlib, pandas,
                pyarrow, boto3, tiktoken) needed by inference-perf-tune-report + the
                Speed-of-Light report pages.
  --with-dev    Also install the dev extras (pytest, ruff, pyright, pre-commit).
  -h, --help    Show this help.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
  esac
done

printf '=== [1/3] installing bundled profile_and_optimize MCP server venv ===\n'
bash "${SERVER_ROOT}/install.sh" "$@"

printf '\n=== [2/3] refreshing Cursor skill symlinks ===\n'
bash "${SCRIPT_DIR}/install-skills-into-cursor.sh"

printf '\n=== [3/3] Cursor ~/.cursor/mcp.json snippet (read-only) ===\n'
bash "${SCRIPT_DIR}/print-cursor-mcp-snippet.sh"

printf '\n[done] bootstrap complete. Add the snippet above to ~/.cursor/mcp.json and reload Cursor.\n'
printf '       Already wired up but red after a version bump? Run: make doctor\n'
