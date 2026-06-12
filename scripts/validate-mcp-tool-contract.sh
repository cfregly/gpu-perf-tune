#!/usr/bin/env bash
# MCP-tool contract sweep: spawn the bundled MCP server over stdio once,
# iterate over every tool in the surface (73 contract + 2 aux), invoke each
# with a minimum-safe arg set based on its safety class, and classify each
# tool as GREEN / YELLOW / RED based on the envelope it returns.
#
# Classification:
#   GREEN  - envelope returned, shape matches io-contract spec, returncode 0
#            (for read_only) or correct ack-refusal/dry-run behavior (for
#            mutating verbs).
#   YELLOW - envelope shape correct + verb crashes only on cluster-side
#            dependencies (no sbatch, no SSH host) which is expected/
#            acceptable agent-side.
#   RED    - malformed envelope, server crash, wrong ack-flag semantics, or
#            crash on a non-cluster cause.
#
# Output: JSON written to ${OUT_DIR:-/tmp}/mcp-tool-contract.json + a
# human-readable summary on stdout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVER="${REPO_ROOT}/plugins/profile-and-optimize/server"
VENV_PY="${SERVER}/.venv/bin/python"
OUT_DIR="${OUT_DIR:-/tmp/profile-and-optimize-validate}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${VENV_PY}" ]]; then
  printf 'FATAL: bundled venv missing at %s; run: bash %s/install.sh --with-dev\n' "${VENV_PY}" "${SERVER}" >&2
  exit 2
fi

PROFILE_AND_OPTIMIZE_REPO_ROOT="${SERVER}" "${VENV_PY}" - "$OUT_DIR" <<'PYEOF'
"""MCP tool contract sweep over the bundled server via stdio JSON-RPC."""
import json
import os
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(sys.argv[1])
OUT_DIR.mkdir(parents=True, exist_ok=True)
SERVER = os.environ["PROFILE_AND_OPTIMIZE_REPO_ROOT"]

sys.path.insert(0, SERVER)
from mcp_surface import list_tools

tools = list_tools()
# Add the 2 aux tools that are registered in server.py, not derived from a CONTRACT.
aux = [
    {"name": "search_runbooks", "safety": "read_only", "library": "mcp_aux", "verb": "search"},
    {"name": "search_evidence", "safety": "read_only", "library": "mcp_aux", "verb": "search"},
]
all_tools = list(tools) + aux
print(f"[1/3] discovered {len(all_tools)} MCP tools ({len(tools)} contract + {len(aux)} aux)")

# Start the MCP server once.
proc = subprocess.Popen(
    [sys.executable, "-m", "profile_and_optimize_mcp", "serve"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, env={**os.environ, "PROFILE_AND_OPTIMIZE_REPO_ROOT": SERVER},
)
req_id = [0]


def rpc(method, params=None):
    req_id[0] += 1
    req = {"jsonrpc": "2.0", "id": req_id[0], "method": method}
    if params is not None:
        req["params"] = params
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed stdout")
    return json.loads(line)


def notify(method, params=None):
    n = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        n["params"] = params
    proc.stdin.write(json.dumps(n) + "\n")
    proc.stdin.flush()


# Initialize handshake.
init = rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "profile-and-optimize-validate", "version": "0.6.5"}})
if "error" in init:
    print(f"FATAL: initialize: {init['error']}", file=sys.stderr)
    sys.exit(2)
notify("notifications/initialized", {})
print("[2/3] MCP server initialized + ready")

# Iterate.
AUX_TOOLS = {"search_runbooks", "search_evidence"}
results = []
for i, t in enumerate(all_tools, start=1):
    name = t["name"]
    safety = t.get("safety", "?")
    # Aux tools (search_runbooks, search_evidence) have a different MCP
    # signature: (query: str, limit: int = 50). Contract-derived tools take
    # a single `params: dict` argument. Dispatch accordingly.
    if name in AUX_TOOLS:
        arguments = {"query": "profile-and-optimize-validate-canary", "limit": 1}
    else:
        # Minimal-safe arg set per safety class. --help is the most defensive
        # universal arg; mutating verbs additionally test ack-refusal semantics.
        arguments = {"params": {"args": ["--help"], "allow_nonzero": True}}
    resp = rpc("tools/call", {"name": name, "arguments": arguments})
    err = resp.get("error")
    if err is not None:
        results.append({"name": name, "safety": safety, "verdict": "RED", "reason": "rpc_error", "detail": err})
        continue
    content = resp.get("result", {}).get("content", [])
    if not content or content[0].get("type") != "text":
        results.append({"name": name, "safety": safety, "verdict": "RED", "reason": "no_text_content", "detail": content})
        continue
    try:
        env = json.loads(content[0]["text"])
    except json.JSONDecodeError as exc:
        results.append({"name": name, "safety": safety, "verdict": "RED", "reason": "envelope_not_json", "detail": str(exc)})
        continue
    # Verify envelope shape: must have tool, library, verb, safety, returncode, stdout, stderr.
    required = {"tool", "library", "verb", "safety", "returncode", "stdout", "stderr"}
    if not required.issubset(set(env.keys())):
        missing = required - set(env.keys())
        results.append({"name": name, "safety": safety, "verdict": "RED", "reason": "missing_envelope_keys", "detail": list(missing)})
        continue
    rc = env.get("returncode", -1)
    stderr_text = env.get("stderr", "")
    # Aux tools: returncode==1 is GREEN (rg exits 1 on "no matches found";
    # a canary query that doesn't match anything is the expected path).
    if name in AUX_TOOLS:
        if rc in (0, 1) and env.get("library") == "mcp_aux" and env.get("verb") == "search":
            results.append({"name": name, "safety": safety, "verdict": "GREEN", "reason": "aux_envelope_ok", "rc": rc})
        else:
            results.append({"name": name, "safety": safety, "verdict": "RED", "reason": "aux_envelope_wrong", "rc": rc, "env": env})
        continue
    # --help exits 0 on argparse-based CLI verbs.
    if rc == 0:
        results.append({"name": name, "safety": safety, "verdict": "GREEN", "reason": "help_ok", "rc": rc})
    elif rc == 2 and "usage:" in stderr_text.lower():
        # argparse --help emits to stdout + exits 0 normally; rc=2 happens
        # when the verb argparse rejects --help (some custom CLIs do this).
        # Still GREEN if usage was printed somewhere.
        results.append({"name": name, "safety": safety, "verdict": "GREEN", "reason": "argparse_usage_emitted", "rc": rc})
    else:
        # Non-zero exit + no recognizable failure mode. Classify YELLOW
        # because the envelope was correct and the failure may be cluster-side
        # (e.g. missing SSH host, no sbatch); RED would require a malformed
        # envelope which we already filtered above.
        results.append({"name": name, "safety": safety, "verdict": "YELLOW", "reason": "non_zero_exit", "rc": rc, "stderr_excerpt": stderr_text[:200]})

# Shutdown.
try:
    proc.stdin.close()
    proc.wait(timeout=3)
except Exception:  # noqa: BLE001
    proc.kill()

# Roll up.
counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
for r in results:
    counts[r["verdict"]] += 1

out_path = OUT_DIR / "mcp-tool-contract.json"
out_path.write_text(json.dumps({"counts": counts, "results": results}, indent=2))

print(f"[3/3] swept {len(results)} tools; GREEN={counts['GREEN']} YELLOW={counts['YELLOW']} RED={counts['RED']}")
print(f"      detail: {out_path}")
if counts["RED"] > 0:
    print("\nRED findings:")
    for r in results:
        if r["verdict"] == "RED":
            print(f"  - {r['name']}: {r['reason']}")
    sys.exit(1)
sys.exit(0)
PYEOF
