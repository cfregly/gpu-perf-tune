# Install `profile_and_optimize` MCP

The bundled server uses MCP over standard input and output. Claude Code and
Codex are the first-class client paths. Cursor, Gemini CLI, and Google
Antigravity configuration helpers are best effort. Any local MCP client that
can launch a command with environment variables can use the manual stdio form.

## Prerequisites

- Python 3.11 or newer with `venv` support.
- Bash and network access for the Python package install.
- A clone of this repository.
- GPU, cluster, profiler, and credential access only for workflows that need
  them. The local install and MCP handshake do not need a GPU.

The two read-only MCP search tools prefer `rg` when it is available. They fall
back to `grep`, so `rg` is not required.

## Install from the repository root

Run one command from the root of this repository:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client codex
```

The installer creates
`~/.local/share/profile-and-optimize-mcp-venv`, installs the complete bundled
server, and registers one `profile_and_optimize` MCP entry. Select another
first-class client with `--client claude` or `--client codex`. The installer
also accepts the best-effort `cursor`, `gemini`, and `antigravity` helpers.
Pass `--client` more than once or use `--client all`.

Claude and Codex use their official MCP commands for a new registration when
the matching CLI is available. A missing CLI, failed CLI command, or existing
registration uses an atomic config-file update. Cursor, Gemini, and Antigravity
use the atomic updater directly.

Repeat installs replace the existing `profile_and_optimize` entry and preserve
unrelated client settings.

## Preview without changing anything

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client all \
  --dry-run
```

Dry run creates no venv, installs no package, runs no client registration
command, and writes no config. It prints only the proposed server entry or
official CLI command. It never prints an existing client config.

Use `--registration file` to preview the config-file fallback even when Claude
or Codex is installed:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client codex \
  --registration file \
  --dry-run
```

## Claude Code

Install and register through the repository installer:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client claude
claude mcp get profile_and_optimize
```

To register an already installed venv yourself with the official CLI:

```bash
SERVER_ROOT="$(pwd)/plugins/profile-and-optimize/server"
MCP_PY="$HOME/.local/share/profile-and-optimize-mcp-venv/bin/python"

claude mcp add --scope user --transport stdio \
  profile_and_optimize \
  --env PROFILE_AND_OPTIMIZE_REPO_ROOT="$SERVER_ROOT" \
  -- \
  "$MCP_PY" -m profile_and_optimize_mcp serve
```

The file fallback updates `~/.claude.json`.

## Codex

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client codex
codex mcp get profile_and_optimize
```

To register an already installed venv yourself with the official CLI:

```bash
SERVER_ROOT="$(pwd)/plugins/profile-and-optimize/server"
MCP_PY="$HOME/.local/share/profile-and-optimize-mcp-venv/bin/python"

codex mcp add \
  --env PROFILE_AND_OPTIMIZE_REPO_ROOT="$SERVER_ROOT" \
  profile_and_optimize -- \
  "$MCP_PY" -m profile_and_optimize_mcp serve
```

The file fallback updates `~/.codex/config.toml`. It removes every prior
`profile_and_optimize` table before adding one replacement, so repeat installs
do not create duplicate TOML tables. Codex CLI, the IDE extension, and the
desktop app share this configuration.

Install the Agent Skills separately:

```bash
make install-skills CLIENT=codex
```

## Cursor, best effort

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client cursor
```

This updates `~/.cursor/mcp.json` with an absolute venv Python path. Restart
Cursor or reload its MCP servers. The helper is tested for safe config writes,
but Cursor behavior is not a release gate.

## Gemini CLI, best effort

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client gemini
```

The installer updates `~/.gemini/settings.json` with an absolute executable
path. Restart Gemini CLI after the update.

## Google Antigravity, best effort

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client antigravity
```

The default global config path is `~/.gemini/config/mcp_config.json`. A
workspace can instead use `.agents/mcp_config.json`. Override the path when
you want workspace-scoped configuration:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client antigravity \
  --antigravity-config /absolute/path/to/workspace/.agents/mcp_config.json
```

## Other local MCP clients

Install the server and write a JSON entry for another client that uses the
Cursor config shape:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client cursor \
  --cursor-config /absolute/path/to/client-mcp.json
```

The generated entry launches an absolute Python path with these arguments:

```text
-m profile_and_optimize_mcp serve
```

It also sets `PROFILE_AND_OPTIMIZE_REPO_ROOT` to the absolute bundled server
directory.

## Optional install sets

Add `--full` for report-renderer and leaderboard dependencies. Add
`--with-dev` for pytest, Ruff, Pyright, pre-commit, and pytest-xdist.

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client codex \
  --full \
  --with-dev
```

## Troubleshooting

- `bundled server root is incomplete`: run the installer from a clone of this
  repository, or pass the absolute `plugins/profile-and-optimize/server` path
  with `--repo-root`.
- `i_understand_this_* is required`: read the named tool's safety class, then
  retry only when the requested action is intended.
- A client still shows the old command: restart the client or reload its MCP
  servers after installation.
