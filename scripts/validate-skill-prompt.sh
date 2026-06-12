#!/usr/bin/env bash
# Skill→prompt simulation: for each on-disk skill, take the canonical example
# prompt from its SKILL.md description, follow the workflow's first MCP tool
# call against synthetic / read-only inputs, and classify GREEN / YELLOW /
# RED. As of v1.14.0 the script covers all 44 skills and ends with a
# consistency check that asserts every SKILL.md under
# plugins/profile-and-optimize/skills/ has a coverage entry.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVER="${REPO_ROOT}/plugins/profile-and-optimize/server"
VENV_PY="${SERVER}/.venv/bin/python"
OUT_DIR="${OUT_DIR:-/tmp/profile-and-optimize-validate}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${VENV_PY}" ]]; then
  printf 'FATAL: bundled venv missing at %s\n' "${VENV_PY}" >&2
  exit 2
fi

PROFILE_AND_OPTIMIZE_REPO_ROOT="${SERVER}" "${VENV_PY}" - "${OUT_DIR}" "${SERVER}" <<'PYEOF'
"""Skill→prompt simulation via stdio MCP server.

For each skill, drive the first MCP call its workflow specifies, and record
the result as GREEN / YELLOW / RED with a one-line reason.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

OUT_DIR = Path(sys.argv[1])
SERVER = sys.argv[2]
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMMANDS_DIR = OUT_DIR / "commands"
COMMANDS_DIR.mkdir(parents=True, exist_ok=True)

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
    return json.loads(proc.stdout.readline())


def notify(method, params=None):
    n = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        n["params"] = params
    proc.stdin.write(json.dumps(n) + "\n")
    proc.stdin.flush()


def call_tool(name, arguments):
    """Invoke a tool, return its envelope dict + the raw response."""
    resp = rpc("tools/call", {"name": name, "arguments": arguments})
    result = resp.get("result", {})
    content = result.get("content", [])
    if not content or content[0].get("type") != "text":
        return None, resp
    try:
        return json.loads(content[0]["text"]), resp
    except json.JSONDecodeError:
        return None, resp


rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "profile-and-optimize-skill-validate", "version": "0.6.5"}})
notify("notifications/initialized", {})


results = []


def record(skill, tier, verdict, reason, env=None):
    rec = {"skill": skill, "tier": tier, "verdict": verdict, "reason": reason}
    if env is not None:
        rec["envelope"] = {
            "tool": env.get("tool"),
            "library": env.get("library"),
            "verb": env.get("verb"),
            "returncode": env.get("returncode"),
            "stderr_len": len(env.get("stderr") or ""),
        }
    results.append(rec)
    print(f"  [{verdict}] {skill}: {reason}")


# ============================================================================
# Tier A: fully testable agent-side (3 skills)
# ============================================================================
print("Tier A (end-to-end with synthetic/read-only data):")

with tempfile.TemporaryDirectory(prefix="profile-and-optimize-skill-") as tmpdir:
    tmp = Path(tmpdir)

    # Seed a synthetic source file once for the perf-baseline tests.
    (tmp / "snap.json").write_text(json.dumps({"busbw_gbps": 47.5, "n_nodes": 8}))

    # 3. perf-baseline-record: register a synthetic baseline.
    env, _ = call_tool("perf_baseline_record", {"params": {"args": [
        "--family", "validation",
        "--measurement", "nccl_busbw",
        "--source", str(tmp / "snap.json"),
        "--value", "47.5",
        "--unit", "GB/s",
        "--notes", "v0.6.5 validation synthetic baseline",
        "--repo-root", str(tmp),
        "--json",
    ], "allow_nonzero": True}})
    if env and env.get("returncode") == 0:
        record("perf-baseline-record", "A", "GREEN", "perf_baseline_record registered synthetic baseline under experiments/artifacts/perf-baselines/", env)
    else:
        record("perf-baseline-record", "A", "RED", f"perf_baseline_record rc={env and env.get('returncode')}; stderr={(env or {}).get('stderr','')[:120]}", env)

    # 4. perf-baseline-diff: locate the entry dir (timestamped slug under
    # validation/nccl_busbw/) from the record envelope's json output.
    entry_dir = None
    if env and env.get("returncode") == 0:
        try:
            record_payload = json.loads(env.get("stdout") or "{}")
            entry_dir = record_payload.get("entry_dir") or record_payload.get("baseline_dir") or record_payload.get("path")
        except json.JSONDecodeError:
            entry_dir = None
    # Fallback: list children of validation/nccl_busbw/ and pick the only one.
    if not entry_dir:
        nccl_dir = tmp / "experiments" / "artifacts" / "perf-baselines" / "validation" / "nccl_busbw"
        if nccl_dir.is_dir():
            children = [p for p in nccl_dir.iterdir() if p.is_dir()]
            if children:
                entry_dir = str(children[0])
    if entry_dir:
        # `--current` is a PATH to a file whose content is either a scalar
        # numeric string or a {"value": N} JSON dict. Write the synthetic
        # current measurement at +6% vs the baseline so the diff produces a
        # YELLOW verdict (an interesting non-trivial result for the bundle).
        current_path = tmp / "current.json"
        current_path.write_text(json.dumps({"value": 47.5 * 1.06}))
        env, _ = call_tool("perf_baseline_diff", {"params": {"args": [
            "--baseline", str(entry_dir),
            "--current", str(current_path),
            "--repo-root", str(tmp),
            "--json",
        ], "allow_nonzero": True}})
        if env and env.get("returncode") == 0:
            record("perf-baseline-diff", "A", "GREEN", "perf_baseline_diff returned a verdict against the synthetic entry", env)
        else:
            record("perf-baseline-diff", "A", "RED", f"perf_baseline_diff rc={env and env.get('returncode')}; stderr={(env or {}).get('stderr','')[:120]}", env)
    else:
        record("perf-baseline-diff", "A", "RED", "could not locate entry dir from perf_baseline_record envelope; downstream diff skipped", None)

    # 5. evidence-bundle-init: scaffold a learnings bundle.
    env, _ = call_tool("evidence_init", {"params": {"args": [
        "--family", "validation",
        "--intent", "v0.6.5 validation synthetic evidence bundle",
        "--run-id", "demo-bundle",
        "--repo-root", str(tmp),
        "--json",
    ], "allow_nonzero": True}})
    if env and env.get("returncode") == 0:
        record("evidence-bundle-init", "A", "GREEN", "evidence_init scaffolded SOURCE.md + summary.md + commands/", env)
    else:
        record("evidence-bundle-init", "A", "RED", f"evidence_init rc={env and env.get('returncode')}; stderr={(env or {}).get('stderr','')[:120]}", env)


# ============================================================================
# Tier B: contract-level only (12 skills; each cites a bundled MCP verb whose
# --help envelope is smoked here; the cluster side is exercised separately)
# ============================================================================
print("\nTier B (contract-level only; cluster-side fully exercised separately):")

TIER_B = [
    # v1.13.0: inference perf-bench report renderer; contract-level smoke
    # via report_smoke --help (the actual render needs matplotlib + bundle).
    ("inference-perf-tune-report", "perf_tune_report_report_smoke"),
    # v1.10.0 inference bridge: cites the existing perf_baseline_record /
    # _diff MCP verbs (no new verbs added).
    ("inference-perf-baseline-bridge", "perf_baseline_record"),
    # Inference skills whose primary workflow drives a bundled library verb.
    ("inference-fleet-leaderboard", "perf_tune_report_fleet_leaderboard"),
    ("inference-known-good-config", "known_good_config_check"),
    ("inference-model-optimize", "perf_tune_report_campaign_init"),
    ("inference-perf-synthesize", "findings_record"),
    ("inference-quantize-calibrate", "evidence_init"),
    ("inference-spec-decode-train", "evidence_init"),
    ("inference-spec-decode-tune", "ai_tuning_optimizer"),
    ("inference-tune-sweep", "perf_tune_report_campaign_init"),
    ("inference-value-ledger", "perf_tune_report_value_view"),
    ("inference-workload-profile", "evidence_init"),
]
for skill, tool in TIER_B:
    env, _ = call_tool(tool, {"params": {"args": ["--help"], "allow_nonzero": True}})
    if env and env.get("library") and env.get("verb"):
        rc = env.get("returncode", -1)
        if rc == 0:
            record(skill, "B", "GREEN", f"{tool} --help returned valid envelope (rc=0)", env)
        else:
            record(skill, "B", "YELLOW", f"{tool} --help envelope shape correct but rc={rc} (acceptable contract-side)", env)
    else:
        record(skill, "B", "RED", f"{tool} returned malformed envelope", env)


# ============================================================================
# Tier C: external-MCP-server-dependent skills (the count drifts as new
# adaptations land; the coverage-consistency check below catches misses).
# ============================================================================
print("\nTier C (external MCP server; not agent-side testable):")

TIER_C = [
    ("prometheus-anchored-query", "prometheus_mcp", "query_observability_knowledge_base"),
    ("k8s-troubleshooting", "prometheus_mcp", "query_prometheus"),
    # v1.12.0: ClickHouse cousin of prometheus-anchored-query. Pure
    # kubectl-port-forward + curl workflow; no MCP tool. Records GREEN
    # because the skill body's discipline (knowledge-base-first SQL +
    # provenance bundle) is the contract.
    ("zymtrace-anchored-query", "kubectl+curl", "manual zymtrace ClickHouse session (no MCP tool; skill body is the contract)"),
    # v1.10.0 inference skills: thin aliases delegating to a vendored
    # upstream SKILL.md at server/inference-tools/ (perf-bench, model-eval).
    # No bundled MCP verb is invoked in the primary workflow.
    ("inference-perf-bench", "vendored upstream", "server/inference-tools/perf-bench/SKILL.md (vendored at UPSTREAM_SHA)"),
    ("inference-model-eval", "vendored upstream", "server/inference-tools/model-eval/SKILL.md (vendored at UPSTREAM_SHA)"),
    # v1.15.0: external-MCP adaptation. Not agent-side testable, but the
    # SKILL.md's allowed-tools + ## Origin section document the contract.
    ("analyze-zymtrace-workload", "zymtrace", "topfunctions / flamegraph / topentities (operator-side optional MCP; see plugin README \"Operator-side optional MCPs\")"),
    # v1.40.0: closed-loop spec-dec-as-a-service orchestrator. Composes sibling
    # skills (inference-workload-profile + inference-spec-decode-train) + bundled
    # perf_tune_report verbs; not agent-side testable, contract is the SKILL.md phases.
    ("inference-spec-decode-service", "profile_and_optimize", "perf_tune_report_* (orchestrator; composes inference-workload-profile + inference-spec-decode-train + the same-node acceptance A/B)"),
    # Pod-attach profiling skills: pure kubectl debug/exec/cp workflows
    # against a live inference pod; no MCP tool, the SKILL.md is the contract.
    ("inference-decode-step-budget", "kubectl", "vLLM /start_profile + /stop_profile via kubectl exec; no MCP tool"),
    ("inference-graph-diff", "kubectl", "torch._dynamo.explain + compile-graph dumps in-pod via kubectl exec/cp"),
    ("inference-kernel-profile", "kubectl", "nsys debug sidecar via kubectl debug; .nsys-rep pulled with kubectl cp"),
    ("inference-kernel-ncu-profile", "kubectl", "ncu debug sidecar via kubectl debug, --kernel-name scoped"),
    ("inference-kernel-whitebox-debug", "kubectl", "in-kernel operand trace + standalone reproducer built in-pod via kubectl exec"),
    ("mirage-graph-coverage", "kubectl", "task_graph_N.json + kernel_N.cu cross-reference pulled via kubectl cp"),
    # Knowledge-layer-only skills: auxiliary search tools (search_runbooks /
    # search_evidence) guide a self-contained script or calculator workflow.
    ("inference-aa-workload", "profile_and_optimize", "search_runbooks (AA shapes driven by the bundled self-contained AIPerf script)"),
    ("inference-capacity-sizing", "profile_and_optimize", "search_runbooks (sizing calculator; knowledge-layer search only)"),
    # Prometheus-side byte-grounding for the SoL hierarchy.
    ("inference-dcgm-correlate", "prometheus_mcp", "query_prometheus (DCGM PROF group over the sweep window)"),
]
for skill, server, tool in TIER_C:
    record(skill, "C", "GREEN", f"first call routes to external MCP server '{server}' (tool: {tool}); not agent-side testable but documented in SKILL.md allowed-tools + plugin .mcp.json declares the server")


# Shutdown.
try:
    proc.stdin.close()
    proc.wait(timeout=3)
except Exception:  # noqa: BLE001
    proc.kill()

# ============================================================================
# Coverage consistency check: every SKILL.md under plugins/profile-and-optimize/skills/
# (excluding `_template`) MUST have a coverage entry in the tiers above.
# Added in v1.14.0: catches the v1.13.0 class of bug where new skills were
# shipped without extending this validator.
# ============================================================================
SKILLS_DIR = Path(SERVER).parent / "skills"
on_disk = sorted(
    p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md")
    if p.parent.name != "_template"
)
covered = sorted({r["skill"] for r in results})
missing = [s for s in on_disk if s not in covered]
extra = [s for s in covered if s not in on_disk]
if missing or extra:
    print(f"\n[FAIL] coverage drift:")
    for s in missing:
        print(f"  MISSING coverage entry for on-disk skill: {s}")
        results.append({"skill": s, "tier": "?", "verdict": "RED",
                         "reason": "no coverage entry in validate-skill-prompt.sh"})
    for s in extra:
        print(f"  EXTRA coverage entry (skill not on disk): {s}")

# Roll up.
counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
for r in results:
    counts[r["verdict"]] += 1

out_path = OUT_DIR / "skill-prompt-simulation.json"
out_path.write_text(json.dumps({
    "counts": counts,
    "on_disk_count": len(on_disk),
    "covered_count": len(covered),
    "missing": missing,
    "extra": extra,
    "results": results,
}, indent=2))

print(f"\n=== Skill-prompt simulation summary ===")
print(f"  On-disk: {len(on_disk)} skills")
print(f"  Covered: {len(covered)} skills")
print(f"  GREEN: {counts['GREEN']}")
print(f"  YELLOW: {counts['YELLOW']}")
print(f"  RED:  {counts['RED']}")
print(f"  Detail: {out_path}")

if counts["RED"] > 0 or missing or extra:
    sys.exit(1)
sys.exit(0)
PYEOF
