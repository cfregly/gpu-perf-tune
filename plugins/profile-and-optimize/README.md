# profile-and-optimize plugin

**Version v0.1.0**

The plugin shipped by this marketplace: 31 GPU inference profiling and
optimization skills plus the bundled `profile_and_optimize` MCP server.

See the [repository README](../../README.md) for the skill families and
quickstart, and [`docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) for the
measurement-rigor canon the skills enforce.

## Skills

One directory per skill under [`skills/`](skills/). Each contains a `SKILL.md`
(frontmatter: name, description, triggers, allowed tools) and optional assets.
Start from [`skills/_template/SKILL.md`](skills/_template/SKILL.md) when adding
a new one.

## Bundled MCP server

[`server/`](server/) hosts the MCP server and its tool libraries. Key entry
points:

- [`server/mcp_surface.py`](server/mcp_surface.py) - the `LIBRARIES` registry
  that defines the exposed tool surface.
- [`server/docs/mcp-tool-io-contract.md`](server/docs/mcp-tool-io-contract.md) -
  the envelope, safety classes, and ack-flag contract every verb follows.
- [`server/docs/mcp-composition.md`](server/docs/mcp-composition.md) - which MCP
  server to reach for in each situation.
- [`server/install.sh`](server/install.sh) - venv install (add `--full` for the
  report-renderer extras).

[`.mcp.json`](.mcp.json) declares the bundled server plus optional `grafana` and
`github` servers. Tokens and URLs come from env vars. Claude Code skips any
server whose env vars are unset, so configure only what you use.

### Operator-side optional MCPs

Two external servers are referenced by skills but deliberately not declared in
`.mcp.json` (Cursor renders undeclared-env servers as connection errors).
Configure them in your own `~/.claude/settings.json` or `~/.cursor/mcp.json`
if you have access:

| Server | Used by | What it provides |
| --- | --- | --- |
| `prometheus_mcp` | `prometheus-anchored-query`, `k8s-troubleshooting`, `inference-dcgm-correlate`, and other observability-anchored skills | Prometheus/Loki queries + observability knowledge base |
| `zymtrace` | `analyze-zymtrace-workload` (and optionally `zymtrace-anchored-query`) | GPU/CPU continuous-profiling flamegraphs and top-functions |

Skills degrade gracefully without them: each SKILL.md documents its bash-tool
fallback where one exists.

## Hooks

[`hooks/`](hooks/) ships runtime-agnostic guard scripts wired for both Claude
Code (`hooks.json`, via `claude-hook-adapter.sh`) and Cursor
(`cursor-hooks.json`): a campaign-teardown confirmation gate and a provenance
commit gate.
