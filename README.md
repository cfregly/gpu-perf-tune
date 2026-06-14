# claude-perf-tune

[![ci](https://github.com/cfregly/claude-perf-tune/actions/workflows/ci.yml/badge.svg)](https://github.com/cfregly/claude-perf-tune/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

GPU inference profiling and optimization skills for [Claude Code](https://claude.com/claude-code), backed by a bundled MCP server: shipped as the `profile-and-optimize` plugin. 31 task-oriented workflows covering benchmark sweeps, kernel-level profiling (nsys / ncu / DCGM / zymtrace), speed-of-light roofline analysis, quantization and speculative-decode tuning, and a multi-page PDF perf-tune report renderer. Each skill is a `SKILL.md` following the open [Agent Skills standard](https://agentskills.io/).

Born from real GPU-fleet performance engineering work, genericized so any team running GPU inference can use it. This is the cost-of-intelligence work: make inference faster and cheaper, measured not asserted.

## What this is

1. **Benchmark & sweep**: `inference-perf-bench` load sweeps, `inference-tune-sweep` engine-knob exploration, `inference-model-eval` quality gates, `perf-baseline-record` / `perf-baseline-diff` regression tracking.
2. **Profile**: `inference-workload-profile`, `inference-kernel-profile` (nsys), `inference-kernel-ncu-profile` (per-kernel roofline), `inference-dcgm-correlate`, `analyze-zymtrace-workload`, `inference-graph-diff` (compile-graph diffs), `mirage-graph-coverage`.
3. **Optimize**: `inference-model-optimize` (cross-engine bring-up orchestrator), `inference-quantize-calibrate`, `inference-spec-decode-train` / `-tune` / `-service`, `inference-decode-step-budget`, `inference-capacity-sizing`, `inference-known-good-config`.
4. **Report & track**: `inference-perf-tune-report` (multi-page PDF renderer), `inference-perf-synthesize`, `inference-fleet-leaderboard`, `inference-value-ledger`, `evidence-bundle-init` provenance bundles, `prometheus-anchored-query` / `zymtrace-anchored-query` anchored observability queries.

The bundled MCP server (`plugins/profile-and-optimize/server/`) exposes the tool surface that backs these skills. The documented bash-tool path is the fallback wherever an optional external server is missing.

## Quickstart

```bash
# 1. Add the marketplace.
claude plugin marketplace add cfregly/claude-perf-tune

# 2. Install the plugin.
claude plugin install --scope user profile-and-optimize@profile-and-optimize-plugins

# 3. Install the bundled MCP server (one-time; creates a venv in the plugin cache).
#    Add --full for the report-renderer deps (matplotlib / pandas / pyarrow).
bash "$(ls -dt ~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/*/server/install.sh | head -1)"
```

Restart Claude Code, then invoke any skill (e.g. `/inference-perf-bench`) or just describe the task: Claude loads a skill automatically when your prompt matches its triggers.

## Repository layout

| Path | What it is |
| --- | --- |
| `plugins/profile-and-optimize/skills/` | The 31 skills (one dir per skill, `SKILL.md` + assets) |
| `plugins/profile-and-optimize/server/` | Bundled MCP server: tool libraries, contract docs, report renderer |
| `plugins/profile-and-optimize/hooks/` | Runtime-agnostic safety gates (Claude Code + Cursor wiring) |
| `configs/sol-ceilings.yaml` | Speed-of-light hardware ceilings (datasheet-sourced) used by roofline pages |
| `campaigns/` | Default output root for perf-tune report campaigns |
| `scripts/` | Capture-hygiene helpers (`nsys-validate-capture.sh`, `zymtrace-ingest-wait.sh`) |
| `docs/METHODOLOGY.md` | The measurement-rigor methodology the skills enforce |
| `mcp-descriptors/` | Offline MCP tool-schema snapshots used by skill lint |

## Methodology

The skills share a common rigor discipline: DRAFT-vs-VERDICT result labeling, full-context perf reporting (hardware, precision, parallelism, engine version alongside every number), validation of every generated asset, and explicit next-lever framing. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Development

- Add a skill: copy [`plugins/profile-and-optimize/skills/_template/SKILL.md`](plugins/profile-and-optimize/skills/_template/SKILL.md).
- Add an MCP verb: add a CLI library under [`plugins/profile-and-optimize/server/`](plugins/profile-and-optimize/server) and update `mcp_surface.py` `LIBRARIES`.
- Common commands: `make help`.

## Limitations

The skills measure and report. They do not tune the cluster for you. Every
number depends on hardware, precision, and engine version, which the skills
record next to the result. The speed-of-light ceilings are datasheet values, an
upper bound rather than a promise. The bundled MCP server is optional. The
documented bash-tool path is the fallback.

## License

[MIT](LICENSE)
