# claude-gpu-perf-tune

[![ci](https://github.com/cfregly/claude-gpu-perf-tune/actions/workflows/ci.yml/badge.svg)](https://github.com/cfregly/claude-gpu-perf-tune/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

GPU inference profiling and optimization skills for [Claude Code](https://claude.com/claude-code), backed by a bundled MCP server: shipped as the `profile-and-optimize` plugin. 31 task-oriented workflows covering benchmark sweeps, kernel-level profiling (nsys / ncu / DCGM / zymtrace), speed-of-light roofline analysis, quantization and speculative-decode tuning, and a multi-page PDF perf-tune report renderer. Each skill is a `SKILL.md` following the open [Agent Skills standard](https://agentskills.io/).

Born from real GPU-fleet performance engineering work, genericized so any team running GPU inference can use it. This is the cost-of-intelligence work: make inference faster and cheaper, measured not asserted.

- **Problem it solves:** GPU inference cost and latency are set by hardware, precision, parallelism, and engine version, and most teams argue about those instead of measuring them.
- **See the surface in under a minute:** `make demo` prints the tool and skill surface, no GPU needed. A real perf run needs the bundled server and hardware.
- **Production lesson it encodes:** measure against speed-of-light, label every result DRAFT until it is variance-controlled and profiled, and record the hardware, precision, and engine version next to every number.
- **ProofPlane workload evidence:** [`docs/workload-proof-packet.md`](docs/workload-proof-packet.md) defines the GPU/inference packet shape for neocloud buyers and ProofPlane pilots. `make workload-proof-check` validates every checked-in `workload-proof-packet.json` for completeness and ProofPlane handoff metadata.

## Value bar

Every benchmark result, optimization claim, and generated report starts as a candidate. It becomes
shippable only when it is adversarially-confirmed to add value: the workload is named, the baseline is
fair, a skeptic has tried to break the finding, and the receipt maps to lower cost, faster runtime,
higher throughput, better reliability, or a clearer operator action.

## Where this fits

This repo is part of the public Claude proof set, but it sits one layer below the startup journey.
[claude-founder-kit](https://github.com/cfregly/claude-founder-kit) is the main runnable kit for
the founder path from first API call to activation and scale. The platform deep-dives cover memory,
grounding, managed agents, parallel calls, and this GPU-cost layer.

- **Main kit:** [claude-founder-kit](https://github.com/cfregly/claude-founder-kit)
- **Platform deep-dives:** claude-memory, claude-grounding, claude-managed-agents, and claude-parallel
- **GPU cost layer:** **[claude-gpu-perf-tune](https://github.com/cfregly/claude-gpu-perf-tune)**
  turns inference performance work into Claude Code skills and MCP tools

## What this is

1. **Benchmark & sweep**: `inference-perf-bench` load sweeps, `inference-tune-sweep` engine-knob exploration, `inference-model-eval` quality gates, `perf-baseline-record` / `perf-baseline-diff` regression tracking.
2. **Profile**: `inference-workload-profile`, `inference-kernel-profile` (nsys), `inference-kernel-ncu-profile` (per-kernel roofline), `inference-dcgm-correlate`, `analyze-zymtrace-workload`, `inference-graph-diff` (compile-graph diffs), `mirage-graph-coverage`.
3. **Optimize**: `inference-model-optimize` (cross-engine bring-up orchestrator), `inference-quantize-calibrate`, `inference-spec-decode-train` / `-tune` / `-service`, `inference-decode-step-budget`, `inference-capacity-sizing`, `inference-known-good-config`.
4. **Report & track**: `inference-perf-tune-report` (multi-page PDF renderer), `inference-perf-synthesize`, `inference-fleet-leaderboard`, `inference-value-ledger`, `evidence-bundle-init` provenance bundles, `prometheus-anchored-query` / `zymtrace-anchored-query` anchored observability queries.

This is a Claude Code plugin: Claude operates it. The 31 skills and the bundled MCP server (`plugins/profile-and-optimize/server/`) are how Claude drives the cost work, loading a skill when your prompt matches its triggers and calling the MCP tools to run the sweep, profile, and report. The documented bash-tool path is the fallback wherever an external observability server is missing.

## Quickstart

```bash
# 1. Add the marketplace.
claude plugin marketplace add cfregly/claude-gpu-perf-tune

# 2. Install the plugin.
claude plugin install --scope user profile-and-optimize@profile-and-optimize-plugins

# 3. Install the bundled MCP server (one-time; creates a venv in the plugin cache).
#    Add --full for the report-renderer deps (matplotlib / pandas / pyarrow).
bash "$(ls -dt ~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/*/server/install.sh | head -1)"
```

Restart Claude Code, then invoke any skill (e.g. `/inference-perf-bench`) or just describe the task: Claude loads a skill automatically when your prompt matches its triggers.

## Verify it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml
make demo    # prints the skill and MCP tool surface, no GPU needed
make check   # doc, skill-count, tool-count, and version gates
make workload-proof-check
```

## Repository layout

| Path | What it is |
| --- | --- |
| `plugins/profile-and-optimize/skills/` | The 31 skills (one dir per skill, `SKILL.md` + assets) |
| `plugins/profile-and-optimize/server/` | Bundled MCP server: tool libraries, contract docs, report renderer |
| `plugins/profile-and-optimize/hooks/` | Runtime-agnostic safety gates (Claude Code + Cursor wiring) |
| `configs/sol-ceilings.yaml` | Speed-of-light hardware ceilings (datasheet-sourced) used by roofline pages |
| `campaigns/` | Default output root for perf-tune report campaigns |
| `examples/workload-proof-packet/` | Synthetic packet fixture that exercises the neocloud workload proof and ProofPlane handoff gates |
| `schemas/workload-proof-packet-v1.json` | Public JSON shape for buyer-facing workload proof packets |
| `scripts/` | Capture-hygiene helpers (`nsys-validate-capture.sh`, `zymtrace-ingest-wait.sh`) |
| `docs/METHODOLOGY.md` | The measurement-rigor methodology the skills enforce |
| `mcp-descriptors/` | Offline MCP tool-schema snapshots used by skill lint |

## Methodology

The skills share a common rigor discipline: DRAFT-vs-VERDICT result labeling, full-context perf reporting (hardware, precision, parallelism, engine version alongside every number), validation of every generated asset, and explicit next-lever framing. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). For neocloud buyer proof and ProofPlane GPU/inference pilots, use the workload packet contract in [`docs/workload-proof-packet.md`](docs/workload-proof-packet.md).

## Development

- Add a skill: copy [`plugins/profile-and-optimize/skills/_template/SKILL.md`](plugins/profile-and-optimize/skills/_template/SKILL.md).
- Add an MCP verb: add a CLI library under [`plugins/profile-and-optimize/server/`](plugins/profile-and-optimize/server) and update `mcp_surface.py` `LIBRARIES`.
- Common commands: `make help`.

## Limitations

Claude operates the skills to measure and report. They do not tune the cluster
for you. Every number depends on hardware, precision, and engine version, which
the skills record next to the result. The speed-of-light ceilings are datasheet
values, an upper bound rather than a promise. Within the plugin, the bundled MCP
server backs the tool surface and the documented bash-tool path is the fallback
wherever an external observability server is missing.

## License

[MIT](LICENSE)
