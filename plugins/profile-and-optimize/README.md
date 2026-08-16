# profile-and-optimize skills and Claude Code adapter

**Version v0.3.1**

This package contains 32 GPU inference profiling and optimization Agent Skills,
the bundled `profile_and_optimize` MCP server, and the Claude Code marketplace
adapter. The MCP server is client-neutral. See the root README for the current
client support matrix.

See the [repository README](../../README.md) for the skill families and
quickstart, and [`docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) for the
measurement-rigor canon the skills enforce.

Start performance triage with
[`inference-performance-hints`](skills/inference-performance-hints/SKILL.md).
Its canonical GPU adaptation is available to agents through the bundled MCP as
`perftune://repo/docs/performance-hints.md` and through `search_runbooks`.

## Skills

One directory per skill under [`skills/`](skills/). Each contains a `SKILL.md`
with official Agent Skills frontmatter and optional assets.
Start from [`templates/skill/SKILL.md`](templates/skill/SKILL.md) when adding
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

[`.mcp.json`](.mcp.json) declares only the bundled `profile_and_optimize`
server. Client installers set its absolute server path.

Grafana and GitHub are optional, user-owned MCP entries. Add them to your
client configuration only after setting their required URLs and credentials.
Claude Code treats an unset `${VAR}` without a default as an invalid MCP
configuration. See the [Claude Code MCP configuration
reference](https://code.claude.com/docs/en/mcp#environment-variable-expansion-in-mcp-json).

### Operator-side optional MCPs

External servers are deliberately not declared in the plugin `.mcp.json`.
For Claude Code, add them with `claude mcp add --scope user`. For Cursor, add
them to `~/.cursor/mcp.json`. Configure only the servers you can authenticate:

| Server | Used by | What it provides |
| --- | --- | --- |
| `grafana` | Operator observability workflows | Grafana dashboards, alerts, and data-source queries |
| `github` | Operator repository workflows | GitHub issues, pull requests, and repository data |
| `prometheus_mcp` | `prometheus-anchored-query`, `k8s-troubleshooting`, `inference-dcgm-correlate`, and other observability-anchored skills | Prometheus/Loki queries + observability knowledge base |
| `zymtrace` | `analyze-zymtrace-workload` (and optionally `zymtrace-anchored-query`) | GPU/CPU continuous-profiling flamegraphs and top-functions |

Skills degrade gracefully without them: each SKILL.md documents its bash-tool
fallback where one exists.

## Hooks

[`hooks/`](hooks/) registers the provenance commit gate for Claude Code through
`claude-hook-adapter.sh`. Enforcement is off by default. Set
`PROVENANCE_COMMIT_GATE=ask` or `PROVENANCE_COMMIT_GATE=deny` in the hook
environment to enable it.
