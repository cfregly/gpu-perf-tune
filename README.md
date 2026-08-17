# gpu-perf-tune

[![ci](https://github.com/cfregly/gpu-perf-tune/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/cfregly/gpu-perf-tune/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

GPU performance engineering for agents and MCP clients, with an
inference-focused skills package and reusable cluster utilities. The project
ships 32 task-oriented [Agent Skills](https://agentskills.io/), a bundled MCP
server, safety guards, workload proof schemas, and local validation tools.
Together they cover first estimates, benchmark sweeps, kernel profiling,
speed-of-light analysis, optimization, and evidence-backed reports.

The work comes from GPU fleet performance practice. Every result stays a
candidate until the workload, baseline, hardware, precision, parallelism, and
engine version are recorded and the result survives a skeptical check.

## Names and boundaries

| Name | Meaning |
| --- | --- |
| `gpu-perf-tune` | The project and repository. It owns the skills, MCP server, guards, schemas, examples, and validation. |
| `profile-and-optimize` | The skills package under `plugins/profile-and-optimize/`. Claude Code consumes it as a plugin. Codex installs the same skills from a clone. |
| `profile_and_optimize` | The configured MCP server key. It is usable without the Claude Code plugin. |
| `profile_and_optimize_mcp` | The Python module that serves the MCP tools and resources. |

Claude Code is one supported client, not the project boundary. Client support
depends on the surface being installed.

## Client support

| Client | Agent Skills | MCP server | Safety guards |
| --- | --- | --- | --- |
| Claude Code | First-class marketplace plugin | Plugin or repo installer | Provenance hook is installed. Enforcement is opt-in |
| Codex | First-class repo installer into `~/.agents/skills` | Repo installer | No client hook adapter. MCP acknowledgement gates still apply |
| Cursor, Gemini CLI, and Antigravity | Best-effort helpers | Best-effort repo installer | No release-tested adapter |
| Other stdio MCP clients | Client-owned discovery | Documented stdio command | No packaged adapter |

The `SKILL.md` sources follow the open Agent Skills standard. This repository
maintains and tests the Claude Code and Codex paths. The MCP protocol remains
client-neutral. Configuration helpers for other clients are available, but
they are not part of the release bar.

## What it covers

1. **Estimate, benchmark, and sweep:** `inference-performance-hints` for rough
   performance bounds, `inference-perf-bench` for load sweeps,
   `inference-tune-sweep` for engine knobs, `inference-model-eval` for quality
   gates, and `perf-baseline-record` with `perf-baseline-diff` for regressions.
2. **Profile:** `inference-workload-profile`, `inference-kernel-profile` for
   nsys, `inference-kernel-ncu-profile` for per-kernel roofline work,
   `inference-dcgm-correlate`, `analyze-zymtrace-workload`,
   `inference-graph-diff`, and `mirage-graph-coverage`.
3. **Optimize:** `inference-model-optimize`, `inference-quantize-calibrate`,
   the speculative decode train, tune, and service skills,
   `inference-decode-step-budget`, `inference-capacity-sizing`, and
   `inference-known-good-config`.
4. **Report and track:** `inference-perf-tune-report`,
   `inference-perf-synthesize`, `inference-fleet-leaderboard`,
   `inference-value-ledger`, `evidence-bundle-init`, and the anchored
   Prometheus and zymtrace query skills.

The documented command-line path remains available when an external
observability MCP server is absent.

## Quickstart

### Inspect the project without a GPU

```bash
git clone https://github.com/cfregly/gpu-perf-tune.git
cd gpu-perf-tune
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/check.txt
make demo
make check
make workload-proof-check
```

`make demo` prints the skill and MCP tool surface. A real performance run needs
the bundled server, the target workload, and suitable GPU hardware.

### Codex: skills and MCP

From a clone of this repository:

```bash
make install-skills CLIENT=codex
make install-mcp CLIENT=codex
codex mcp get profile_and_optimize
```

The registration check should report `enabled: true`. Run
`make smoke-mcp-runtime` for a live server handshake, then restart Codex after
installation. Codex discovers the linked skills under
`~/.agents/skills`. Use `/skills`, invoke a skill with `$skill-name`, or
describe the task and let Codex select one. The MCP configuration is shared by
Codex CLI, the IDE extension, and the desktop app. See the official Codex
[Skills](https://developers.openai.com/codex/skills/) and
[MCP](https://developers.openai.com/codex/mcp/) references for the client-side
contracts.

### Claude Code: skills, MCP, and optional provenance enforcement

```bash
claude plugin marketplace add cfregly/gpu-perf-tune
claude plugin install --scope user \
  profile-and-optimize@profile-and-optimize-plugins

# Install the bundled MCP server inside the current plugin cache entry.
# Add --full when you need the PDF report dependencies.
bash "$(ls -dt ~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/*/server/install.sh | head -1)"

claude mcp get plugin:profile-and-optimize:profile_and_optimize
```

The health check should report `Status: ✔ Connected`. Restart Claude Code.
Invoke a skill such as `/inference-perf-bench`, or describe the task and let
Claude Code select a matching skill. The plugin installs its provenance hook,
but the hook remains inactive until
`PROVENANCE_COMMIT_GATE=ask` or `PROVENANCE_COMMIT_GATE=deny` is present in the
Claude hook environment. Install `jq` on `PATH` before enabling either mode.

### Other clients: best-effort helpers

The repository keeps configuration helpers for Cursor, Gemini CLI, and Google
Antigravity. They are useful starting points, but changes to those clients do
not block a release. The full command list and generic stdio form live in the
[MCP installation reference](plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/INSTALL.md).

```bash
make install-skills CLIENT=cursor
make install-mcp CLIENT=cursor
```

## Upgrading

Read [`docs/UPGRADING.md`](docs/UPGRADING.md) before changing release lines. It
lists the client refresh commands and required installer, AI tuning, MLPerf
rules, and MCP acknowledgement changes.

## Value bar

Every benchmark result, optimization claim, and generated report starts as a
candidate. It must be adversarially-confirmed to add value before it ships. The
workload is named, the baseline is fair, a skeptic has tried to break the
finding, and the receipt maps to lower cost, faster runtime, higher throughput,
better reliability, or a clearer operator action.

## Workload proof contract

[`docs/workload-proof-packet.md`](docs/workload-proof-packet.md) defines the
GPU inference packet shape for neocloud buyers and workflow handoffs.
`make workload-proof-check` validates checked-in packets for completeness and
local handoff metadata.

## Repository layout

| Path | What it is |
| --- | --- |
| `plugins/profile-and-optimize/skills/` | The Agent Skills, one directory per installed skill |
| `plugins/profile-and-optimize/templates/skill/` | Starting point for a new skill |
| `plugins/profile-and-optimize/server/` | MCP server, tool libraries, contract docs, and report renderer |
| `plugins/profile-and-optimize/hooks/` | Runtime-neutral provenance guard plus the Claude Code adapter |
| `configs/sol-ceilings.yaml` | Datasheet-sourced hardware ceilings used by roofline reports |
| `examples/workload-proof-packet/` | Synthetic fixture for packet and handoff validation |
| `schemas/workload-proof-packet-v1.json` | Public JSON Schema for workload proof packets |
| `docs/METHODOLOGY.md` | Measurement and reporting rules shared by the skills |
| `mcp-descriptors/` | Offline MCP tool-schema snapshots used by skill lint |

## Methodology

The skills enforce DRAFT and VERDICT labels, full performance context, asset
validation, and a clear next action. Read
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the shared rules. The
[`performance hints adaptation`](plugins/profile-and-optimize/server/docs/performance-hints.md)
adds estimate-first triage, with a
[synthetic example](examples/performance-hints/README.md).

## Optional integrations

A workflow system can consume `workflow_handoff` blocks when GPU workload
evidence needs to attach to a broader customer record. ProofPlane is one
possible consumer. It does not change this project's local contract,
validation gates, or runtime dependencies.

## Development

- Add a skill from
  [`plugins/profile-and-optimize/templates/skill/SKILL.md`](plugins/profile-and-optimize/templates/skill/SKILL.md).
- Add an MCP verb under
  [`plugins/profile-and-optimize/server/`](plugins/profile-and-optimize/server)
  and update `mcp_surface.py`.
- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.
- Run `make help` for the local command reference.

## Limitations

The project helps agents measure and report. It does not tune a cluster by
itself. Every number depends on the workload and runtime context. Datasheet
speed-of-light ceilings are upper bounds, not promises. External systems such
as Grafana, Prometheus, GitHub, and zymtrace still need operator credentials and
their own client configuration.

## License

Project-authored material is [MIT licensed](LICENSE). Third-party adaptations
retain their own terms and attribution in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
