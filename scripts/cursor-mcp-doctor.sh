#!/usr/bin/env bash
# Diagnose (and optionally repair) the `profile_and_optimize` MCP server entry in
# ~/.cursor/mcp.json.
#
# The recurring failure mode (OPERATOR-TODO.md item #7 + AGENTS.md
# "Cockpit-retirement repoint"): after an `profile-and-optimize` version bump or a
# checkout move, the `profile_and_optimize` entry's `command` points at a venv path that
# no longer exists, so the next Cursor MCP reload spawns a child that fails with
# `spawn ... ENOENT` / `No module named profile_and_optimize_mcp`. The morning's healthy
# stdio child can mask it for hours; only the next reload surfaces it.
#
# Default: READ-ONLY report. With --fix, repoints ONLY the `profile_and_optimize` entry's
# `command` + `env.PROFILE_AND_OPTIMIZE_REPO_ROOT` to THIS checkout's bundled venv, after
# backing up ~/.cursor/mcp.json to a timestamped .bak. No other entry is touched.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVER_ROOT="${REPO_ROOT}/plugins/profile-and-optimize/server"
VENV_PY="${SERVER_ROOT}/.venv/bin/python"
MCP_JSON="${HOME}/.cursor/mcp.json"
FIX=0

usage() {
  cat <<'EOF'
Usage: scripts/cursor-mcp-doctor.sh [--fix]

  (no args)   Read-only: report whether the profile_and_optimize entry in ~/.cursor/mcp.json
              points at a venv python that actually exists.
  --fix       Repoint ONLY the profile_and_optimize entry to this checkout's bundled venv,
              backing up ~/.cursor/mcp.json to a timestamped .bak first.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix) FIX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${MCP_JSON}" ]]; then
  printf '[doctor] %s does not exist. Add the profile_and_optimize block with:\n' "${MCP_JSON}"
  printf '         make print-mcp-snippet\n'
  exit 0
fi

PROFILE_AND_OPTIMIZE_DOCTOR_MCP_JSON="${MCP_JSON}" \
PROFILE_AND_OPTIMIZE_DOCTOR_VENV_PY="${VENV_PY}" \
PROFILE_AND_OPTIMIZE_DOCTOR_SERVER_ROOT="${SERVER_ROOT}" \
PROFILE_AND_OPTIMIZE_DOCTOR_FIX="${FIX}" \
python3 - <<'PY'
import datetime
import json
import os
import shutil
import sys

mcp_json = os.environ["PROFILE_AND_OPTIMIZE_DOCTOR_MCP_JSON"]
want_py = os.environ["PROFILE_AND_OPTIMIZE_DOCTOR_VENV_PY"]
want_root = os.environ["PROFILE_AND_OPTIMIZE_DOCTOR_SERVER_ROOT"]
fix = os.environ["PROFILE_AND_OPTIMIZE_DOCTOR_FIX"] == "1"

try:
    with open(mcp_json) as fh:
        data = json.load(fh)
except json.JSONDecodeError as exc:
    print(f"[doctor] {mcp_json} is not valid JSON ({exc}); not touching it.", file=sys.stderr)
    sys.exit(2)

servers = data.get("mcpServers") or data.get("servers") or {}
entry = servers.get("profile_and_optimize")

if entry is None:
    print("[doctor] no `profile_and_optimize` entry in ~/.cursor/mcp.json.")
    print("         Add it with: make print-mcp-snippet")
    sys.exit(0)

cur_cmd = entry.get("command", "")
cur_root = (entry.get("env") or {}).get("PROFILE_AND_OPTIMIZE_REPO_ROOT", "")
cmd_ok = bool(cur_cmd) and os.path.exists(cur_cmd) and os.access(cur_cmd, os.X_OK)
root_ok = bool(cur_root) and os.path.isdir(cur_root)
want_py_exists = os.path.exists(want_py)

print(f"[doctor] profile_and_optimize.command          = {cur_cmd or '(unset)'}")
print(f"[doctor]   -> exists & executable?  = {cmd_ok}")
print(f"[doctor] profile_and_optimize.PROFILE_AND_OPTIMIZE_REPO_ROOT = {cur_root or '(unset)'}")
print(f"[doctor]   -> directory exists?     = {root_ok}")
print(f"[doctor] this checkout's venv python = {want_py} (exists={want_py_exists})")

healthy = cmd_ok and root_ok
if healthy:
    print("[doctor] OK: profile_and_optimize entry points at a venv python that exists.")
    if not fix:
        sys.exit(0)
    if cur_cmd == want_py and cur_root == want_root:
        print("[doctor] --fix: already pointed at this checkout; nothing to change.")
        sys.exit(0)

if not fix:
    print("[doctor] STALE: profile_and_optimize entry is broken. Re-run with --fix to repoint it,", file=sys.stderr)
    print("         or `make doctor FIX=1`.", file=sys.stderr)
    sys.exit(1)

if not want_py_exists:
    print(f"[doctor] --fix refused: this checkout's venv {want_py} does not exist yet.", file=sys.stderr)
    print("         Run `make bootstrap` (or `bash plugins/profile-and-optimize/server/install.sh`) first.", file=sys.stderr)
    sys.exit(2)

ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup = f"{mcp_json}.bak-{ts}-pre-doctor"
shutil.copy2(mcp_json, backup)
entry["command"] = want_py
entry.setdefault("env", {})["PROFILE_AND_OPTIMIZE_REPO_ROOT"] = want_root
with open(mcp_json, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
print(f"[doctor] --fix: backed up to {backup}")
print(f"[doctor] --fix: repointed profile_and_optimize.command -> {want_py}")
print(f"[doctor] --fix: repointed PROFILE_AND_OPTIMIZE_REPO_ROOT -> {want_root}")
print("[doctor] Reload Cursor (toggle the profile_and_optimize MCP off/on, or restart) to pick it up.")
PY
