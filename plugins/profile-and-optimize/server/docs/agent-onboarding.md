# Agent Onboarding

Status: Active

This repo ships agent skills and the `profile_and_optimize` MCP server that an
operator or agent needs for MLPerf / performance work. Anyone with a clone
of the repo can install the same surface locally. The package is
`profile_and_optimize_mcp`, the configured MCP server key is `profile_and_optimize`, and
repository docs and runbooks are exposed as MCP resources.

## Install profile_and_optimize

From `plugins/profile-and-optimize/server/`, configure Cursor:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client cursor
```

Or configure every supported local MCP client:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client all
```

The installer creates `~/.local/share/profile-and-optimize-mcp-venv`, installs the
repo package editable, and merges the server block into the chosen
client config. The Cursor block is:

```json
"profile_and_optimize": {
  "command": "~/.local/share/profile-and-optimize-mcp-venv/bin/python",
  "args": ["-m", "profile_and_optimize_mcp", "serve"],
  "env": {
    "PROFILE_AND_OPTIMIZE_REPO_ROOT": "/path/to/claude-perf-tune/plugins/profile-and-optimize/server",
    "PROFILE_AND_OPTIMIZE_LOGIN_HOST": "${USER}@192.0.2.10"
  }
}
```

The package also installs a `profile-and-optimize-mcp serve` console entry point,
but the installer writes the venv-backed `python -m profile_and_optimize_mcp serve`
form so local clients resolve the editable package and its dependencies
from the MCP venv instead of the IDE's ambient Python.

Restart Cursor. The server exposes the repo's docs and runbooks as MCP
resources, plus structured tools that wrap the same CLIs documented in
`tools/README.md`. The canonical agent-facing request/response contract is
[`mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).

MCP tools are derived from [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py), which reads
each library's live CLI parser contract. The FastMCP runtime
is [`server.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/server.py).

Client-specific notes:

- Cursor: `~/.cursor/mcp.json`, then restart Cursor.
- Claude Code: `~/.claude/settings.json`, or use `claude mcp add ...`.
- Codex CLI: `~/.codex/config.toml` under `[mcp_servers.profile_and_optimize]`.
- Gemini CLI: `~/.gemini/settings.json` under `mcpServers.profile_and_optimize`.
- Google Antigravity: raw `mcp_config.json` via Agent window -> Manage
  MCP Servers -> View raw config, then refresh MCP servers.

Full snippets and troubleshooting live in
[`tools/profile_and_optimize_mcp/INSTALL.md`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/INSTALL.md).

## Skills

The plugin's skills live under `plugins/profile-and-optimize/skills/` and
travel with any clone. The plugin README carries the full table. Cursor
users can symlink every skill into `~/.cursor/skills/` with
`make refresh-symlinks` from the repo root.

Use the skills for agent behavior and the `profile_and_optimize` MCP tools for
repeatable tool execution. Mutating MCP tools still require explicit
`i_understand_this_*` arguments per [`../AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md).

## Full profile_and_optimize Tool List

Run `python3 mcp_surface.py list` for the live tool names and
descriptions. Tools are derived from the CLI contracts of these server
libraries:

- `ai_tuning`, `profile`
- `perf_baseline`, `evidence`
- `slurm`, `findings`
- `perf_tune_report`, `known_good_config`

## Mutating Tool Pattern

Every mutating tool mirrors the CLI acknowledgement field. Examples:

- `i_understand_this_submits_jobs=true`
- `i_understand_this_pulls_license_gated_data=true`
- `i_understand_this_stages_artifacts=true`

The MCP wrapper refuses before invoking the underlying command if the
acknowledgement is absent.
