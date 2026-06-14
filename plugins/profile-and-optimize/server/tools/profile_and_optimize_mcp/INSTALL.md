# Installing profile_and_optimize MCP

From the repo root:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client cursor
```

The installer creates `~/.local/share/profile-and-optimize-mcp-venv`, installs this
package editable, and merges a `profile_and_optimize` server block into the chosen
client config. Use `--client all` to configure every supported local
client.

## Cursor

Config path: `~/.cursor/mcp.json`.

```json
{
  "mcpServers": {
    "profile_and_optimize": {
      "command": "~/.local/share/profile-and-optimize-mcp-venv/bin/python",
      "args": ["-m", "profile_and_optimize_mcp", "serve"],
      "env": {
        "PROFILE_AND_OPTIMIZE_REPO_ROOT": "/path/to/profile-and-optimize",
        "PROFILE_AND_OPTIMIZE_LOGIN_HOST": "${USER}@192.0.2.10"
      }
    }
  }
}
```

Install/update:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client cursor
```

Restart Cursor after saving. `PROFILE_AND_OPTIMIZE_REPO_ROOT` is optional when
Cursor starts inside the repo, but setting it makes the server
independent of the IDE working directory.

The installed package also provides `profile-and-optimize-mcp serve`. The installer
uses the explicit venv Python above so Cursor does not accidentally start
the server with a Python that lacks the editable `profile_and_optimize_mcp` package.

## Claude Code

Preferred CLI install:

```bash
claude mcp add --transport stdio \
  --env PROFILE_AND_OPTIMIZE_REPO_ROOT="$PWD" \
  --env PROFILE_AND_OPTIMIZE_LOGIN_HOST="${PROFILE_AND_OPTIMIZE_LOGIN_HOST:-$USER@192.0.2.10}" \
  profile_and_optimize -- "$HOME/.local/share/profile-and-optimize-mcp-venv/bin/python" -m profile_and_optimize_mcp serve
```

Equivalent JSON lives in `~/.claude/settings.json` under
`mcpServers.profile_and_optimize`. The repo installer can merge that block:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client claude
```

Verify with:

```bash
claude mcp list
```

## Codex CLI

Config path: `~/.codex/config.toml`.

```toml
[mcp_servers.profile_and_optimize]
command = "~/.local/share/profile-and-optimize-mcp-venv/bin/python"
args = ["-m", "profile_and_optimize_mcp", "serve"]
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 300

[mcp_servers.profile_and_optimize.env]
PROFILE_AND_OPTIMIZE_REPO_ROOT = "/path/to/profile-and-optimize"
PROFILE_AND_OPTIMIZE_LOGIN_HOST = "<user>@192.0.2.10"
```

Install/update:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client codex
```

## Gemini CLI

Config path: `~/.gemini/settings.json`.

```json
{
  "mcpServers": {
    "profile_and_optimize": {
      "command": "~/.local/share/profile-and-optimize-mcp-venv/bin/python",
      "args": ["-m", "profile_and_optimize_mcp", "serve"],
      "env": {
        "PROFILE_AND_OPTIMIZE_REPO_ROOT": "/path/to/profile-and-optimize",
        "PROFILE_AND_OPTIMIZE_LOGIN_HOST": "<user>@192.0.2.10"
      }
    }
  }
}
```

Install/update:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client gemini
```

## Google Antigravity

Open Antigravity, use `Agent window -> Manage MCP Servers -> View raw
config`, and confirm the raw config path. By default the installer
writes `~/.config/antigravity/mcp_config.json`. Pass
`--antigravity-config PATH` to `configure_clients.py` if your install
uses a different path.

```json
{
  "mcpServers": {
    "profile_and_optimize": {
      "command": "~/.local/share/profile-and-optimize-mcp-venv/bin/python",
      "args": ["-m", "profile_and_optimize_mcp", "serve"],
      "env": {
        "PROFILE_AND_OPTIMIZE_REPO_ROOT": "/path/to/profile-and-optimize",
        "PROFILE_AND_OPTIMIZE_LOGIN_HOST": "<user>@192.0.2.10"
      }
    }
  }
}
```

Install/update:

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client antigravity
```

Then refresh MCP servers in the Antigravity UI.

## All local clients

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client all
```

Use `--dry-run` to print merged configs without writing them.

## Troubleshooting

- A `cannot locate ... repo root` error: set
  `PROFILE_AND_OPTIMIZE_REPO_ROOT` to the absolute repo path.
- `ssh: Permission denied`: load the operator SSH key or set
  `PROFILE_AND_OPTIMIZE_LOGIN_HOST` to a reachable login host.
- A mutating tool says `i_understand_this_* is required`: retry only
  after reading the relevant runbook and pass the named ack field with
  value `true`.
