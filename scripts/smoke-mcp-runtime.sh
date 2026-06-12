#!/usr/bin/env bash
# End-to-end MCP runtime smoke test for the bundled profile_and_optimize MCP server.
#
# Spawns `python -m profile_and_optimize_mcp serve` over stdio, sends MCP JSON-RPC
# requests (initialize / tools/list / tools/call for search_runbooks),
# verifies the expected envelope shape + tool count.
#
# This tightens the "only mcp_surface.py was exercised, not the runtime"
# caveat from earlier release outcome tables: this script actually starts
# the FastMCP stdio server and confirms the wire-format works.
#
# Usage:
#   bash scripts/smoke-mcp-runtime.sh                # prefers ${CLAUDE_PLUGIN_ROOT}/server, else in-repo
#   bash scripts/smoke-mcp-runtime.sh --server PATH  # explicit server directory
#
# Exit codes:
#   0 = green (server started + tools/list returned expected count + tools/call returned envelope)
#   1 = warn (server started but one or more assertions failed)
#   2 = red  (server failed to start, missing venv, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SERVER=""
# Default expected tool count is read at runtime from
# `mcp_surface.py counts` so this script never drifts from the canonical
# constants. Override via `--expected-tools N` or `EXPECTED_TOOLS=N` for
# pinned-count testing.
EXPECTED_TOOLS="${EXPECTED_TOOLS:-}"
QUIET=0

usage() {
  cat <<'EOF'
Usage: scripts/smoke-mcp-runtime.sh [options]

Options:
  --server PATH         Bundled MCP server directory. Default: prefer
                        ${CLAUDE_PLUGIN_ROOT}/server, else the in-repo
                        plugins/profile-and-optimize/server/.
  --expected-tools N    Override the expected MCP tool count. Default:
                        read from `mcp_surface.py counts --json` (the
                        canonical source-of-truth in the bundled server).
  --quiet               Suppress the per-step progress lines; only print the
                        final verdict and exit code semantics.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --expected-tools) EXPECTED_TOOLS="$2"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

log() {
  if [[ "${QUIET}" -eq 0 ]]; then
    printf '%s\n' "$*"
  fi
}

# Resolve the server directory.
if [[ -z "${SERVER}" ]]; then
  if [[ -n "${CLAUDE_PLUGIN_ROOT:-}" && -d "${CLAUDE_PLUGIN_ROOT}/server" ]]; then
    SERVER="${CLAUDE_PLUGIN_ROOT}/server"
  else
    SERVER="${REPO_ROOT}/plugins/profile-and-optimize/server"
  fi
fi

if [[ ! -f "${SERVER}/AGENTS.md" || ! -d "${SERVER}/tools" ]]; then
  printf 'FATAL: server directory does not look right: %s\n' "${SERVER}" >&2
  exit 2
fi

VENV_PY="${SERVER}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  printf 'FATAL: bundled venv not installed at %s\n' "${VENV_PY}" >&2
  printf 'Run: bash %s/install.sh\n' "${SERVER}" >&2
  exit 2
fi

# Resolve EXPECTED_TOOLS from mcp_surface.py if the caller didn't pin
# it. This keeps this script in lockstep with the canonical constants in
# mcp_surface.py instead of drifting independently. Note `--json` is a
# top-level flag, before the subcommand.
if [[ -z "${EXPECTED_TOOLS}" ]]; then
  EXPECTED_TOOLS="$("${VENV_PY}" "${SERVER}/mcp_surface.py" --json counts 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['canonical']['total_mcp_tools'])" 2>/dev/null || true)"
  if [[ -z "${EXPECTED_TOOLS}" ]]; then
    printf 'FATAL: could not resolve canonical tool count from %s/mcp_surface.py --json counts\n' "${SERVER}" >&2
    printf '       Pass --expected-tools N or run %s -m profile_and_optimize_mcp serve manually to debug.\n' "${VENV_PY}" >&2
    exit 2
  fi
fi

log "[1/4] server:         ${SERVER}"
log "[1/4] venv python:    ${VENV_PY}"
log "[1/4] expected tools: ${EXPECTED_TOOLS} (from mcp_surface.py canonical-counts)"

# Run the round-trip via an embedded Python program. Sending raw JSON-RPC over
# stdio works regardless of which MCP client we're invoking; the FastMCP server
# implements the MCP wire protocol.
SMOKE_OUTPUT="$(PROFILE_AND_OPTIMIZE_REPO_ROOT="${SERVER}" "${VENV_PY}" - <<'PYEOF'
import json
import os
import subprocess
import sys
import time

server_root = os.environ["PROFILE_AND_OPTIMIZE_REPO_ROOT"]
venv_py = os.path.join(server_root, ".venv", "bin", "python")

proc = subprocess.Popen(
    [venv_py, "-m", "profile_and_optimize_mcp", "serve"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env={**os.environ, "PROFILE_AND_OPTIMIZE_REPO_ROOT": server_root},
)


def send_request(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    req = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        req["params"] = params
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed stdout before responding")
    return json.loads(line)


def send_notification(method: str, params: dict | None = None) -> None:
    note = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        note["params"] = params
    proc.stdin.write(json.dumps(note) + "\n")
    proc.stdin.flush()


result = {"start_ok": False, "tools_list": None, "tools_call": None, "errors": []}

try:
    # 1. initialize handshake.
    init_resp = send_request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "profile-and-optimize-smoke", "version": "0.6.0"},
    }, req_id=1)
    if "error" in init_resp:
        result["errors"].append({"step": "initialize", "error": init_resp["error"]})
    else:
        result["start_ok"] = True
        send_notification("notifications/initialized", {})

    # 2. tools/list.
    if result["start_ok"]:
        tl_resp = send_request("tools/list", {}, req_id=2)
        if "error" in tl_resp:
            result["errors"].append({"step": "tools/list", "error": tl_resp["error"]})
        else:
            tools = tl_resp.get("result", {}).get("tools", [])
            result["tools_list"] = {
                "tool_count": len(tools),
                "sample_names": sorted({t.get("name") for t in tools})[:6],
            }

    # 3. tools/call for search_runbooks (read_only; minimal payload).
    if result["start_ok"]:
        call_resp = send_request("tools/call", {
            "name": "search_runbooks",
            "arguments": {"params": {"args": ["b200 8b", "--limit", "1"]}},
        }, req_id=3)
        if "error" in call_resp:
            result["errors"].append({"step": "tools/call", "error": call_resp["error"]})
        else:
            content = call_resp.get("result", {}).get("content", [])
            result["tools_call"] = {
                "ok": True,
                "content_chunks": len(content),
                "first_chunk_type": content[0].get("type") if content else None,
            }
finally:
    try:
        proc.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    result["exit_code"] = proc.returncode

print(json.dumps(result, indent=2, sort_keys=True))
PYEOF
)"

# Parse the result envelope (the Python program prints JSON to stdout).
log "[2/4] runtime round-trip output:"
if [[ "${QUIET}" -eq 0 ]]; then
  printf '%s\n' "${SMOKE_OUTPUT}" | sed 's/^/    /'
fi

# Assertions.
START_OK="$(printf '%s' "${SMOKE_OUTPUT}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('start_ok'))")"
TOOL_COUNT="$(printf '%s' "${SMOKE_OUTPUT}" | python3 -c "import json,sys; r=json.load(sys.stdin); print((r.get('tools_list') or {}).get('tool_count') or 0)")"
CALL_OK="$(printf '%s' "${SMOKE_OUTPUT}" | python3 -c "import json,sys; r=json.load(sys.stdin); print((r.get('tools_call') or {}).get('ok'))")"

ASSERT_FAIL=0
if [[ "${START_OK}" != "True" ]]; then
  printf '[FAIL] server did not return a successful initialize handshake\n' >&2
  ASSERT_FAIL=1
fi
if [[ "${TOOL_COUNT}" != "${EXPECTED_TOOLS}" ]]; then
  printf '[FAIL] tools/list returned %s tools (expected %s)\n' "${TOOL_COUNT}" "${EXPECTED_TOOLS}" >&2
  ASSERT_FAIL=1
fi
if [[ "${CALL_OK}" != "True" ]]; then
  printf '[FAIL] tools/call for search_runbooks did not return a successful envelope\n' >&2
  ASSERT_FAIL=1
fi

if [[ "${ASSERT_FAIL}" -eq 1 ]]; then
  printf '[FAIL] mcp-runtime smoke test had assertion failures (see above)\n' >&2
  exit 1
fi

log "[3/4] initialize handshake:      ok"
log "[3/4] tools/list count:          ${TOOL_COUNT} (expected ${EXPECTED_TOOLS})"
log "[3/4] tools/call search_runbooks: ok"
log "[4/4] [ok] mcp-runtime smoke test passed"

exit 0
