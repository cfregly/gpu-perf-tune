# Cursor MCP troubleshooting for `profile-and-optimize`

This document covers the common failure modes for the `profile_and_optimize` MCP server (and optional sibling servers) when used inside Cursor's MCP panel, and the one-liner that resolves each:

1. **`profile_and_optimize`: `spawn .../bin/python ENOENT`** - missing installer venv or stale local config.
2. **Env-var-gated optional servers: `Connection closed`** - expected behavior when the gating env var is unset.
3. **OAuth-backed MCPs (e.g., `github`): "Logout" badge** - OAuth token / session cookie expiry.

If you are arriving here from a screenshot of red badges in the Cursor MCP panel, walk down this list in order.

## 1. `profile_and_optimize`: `spawn ... ENOENT`

### Symptom

The MCP panel shows the `profile_and_optimize` server with a red badge and an error like

```
spawn /Users/<you>/.local/share/profile-and-optimize-mcp-venv/bin/python ENOENT
```

The configured Python path may also point to an old checkout or another venv
that no longer exists.

### Root cause

The Cursor entry was created by an older install, the installer venv was
removed, or the repository moved. Skill links under `~/.cursor/skills` are
separate from the MCP runtime and do not replace its venv.

### Fix

```bash
cd /path/to/profile-and-optimize-checkout
git pull --ff-only origin main
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client cursor
```

The installer creates or updates the persistent venv under
`~/.local/share/profile-and-optimize-mcp-venv` and atomically updates
`~/.cursor/mcp.json`. Inside Cursor, open the MCP panel, find
`profile_and_optimize`, and reload or toggle the server so Cursor reads the new
entry.

If the badge is still red after the toggle, fully restart Cursor (`Cmd-Q` on macOS) - the in-memory cache is cleared on cold start.

### Skills are installed separately

Run `make refresh-symlinks` only when you also need to install or refresh the
32 skills under `~/.cursor/skills`. The Cursor reload is application state and
cannot be driven by the repository installer.

## 2. Env-var-gated optional servers: `Connection closed`

### Symptom

The MCP panel shows an optional server you added to your own `~/.cursor/mcp.json` with a "Connection closed" message.

### Root cause

A common pattern for optional servers is gating the command on an environment variable:

```json
{
  "my_optional_server": {
    "command": "${MY_OPTIONAL_SERVER_COMMAND:-true}",
    "args": ["--stdio"]
  }
}
```

When your environment does not export `MY_OPTIONAL_SERVER_COMMAND`, the shell expands the placeholder to `true`, which is a Unix builtin that exits 0 immediately. Cursor sees the process exit before any MCP handshake and reports the connection as closed.

This is **expected** when you don't have the corresponding tool installed. The plugin runs without optional servers. The gate exists so that operators who DO have access can opt in by exporting the relevant env var before launching Cursor.

### Fix (if you intend to enable the optional server)

Export the env var pointing at the real binary **before** launching Cursor:

```bash
export MY_OPTIONAL_SERVER_COMMAND=/path/to/server-binary
open -a Cursor
```

For persistent setup, put the export in your shell profile (e.g. `~/.zshrc`. Brace variables that are followed by punctuation, like `${var}:port`) and source it before launching Cursor.

### Fix (if you don't intend to enable it. Suppress the red badge)

Two options:

- Ignore the badge. The server is optional and profile-and-optimize does not call into it by default.
- Comment out the corresponding block in your local `~/.cursor/mcp.json`.

## 3. OAuth-backed MCPs show "Logout" badges

### Symptom

Servers that authenticate via OAuth (e.g. `github`) show a "Logout" badge in the MCP panel.

### Root cause

These servers carry OAuth tokens or session cookies that expire on a regular schedule (typically 24 hours to 30 days depending on the provider). Once expired, the server itself runs fine but the per-server credential needs to be refreshed.

### Fix

Inside Cursor's MCP panel, click the "Sign in" or "Re-authenticate" button next to the affected server. Cursor opens the provider's auth flow in a browser. Complete the flow. The badge transitions from "Logout" to "Connected".

If the auth flow fails or hangs, the fallback is to remove and re-add the server in the MCP panel, which forces Cursor to re-run the auth flow from scratch.

### Note on optional servers

Only `profile_and_optimize` ships in the plugin's [`.mcp.json`](../plugins/profile-and-optimize/.mcp.json).
Grafana, GitHub, Prometheus, Zymtrace, and other external servers are optional
entries owned by the user. Add them to your local client configuration when
needed.

## 4. User-owned env placeholders can show red `Connection closed`

### Symptom

An optional server in your local `~/.cursor/mcp.json` shows a red
`Connection closed` badge immediately after Cursor starts.

### Root cause

The local entry references an unset environment variable such as `${VAR}`.
The command then fails before the MCP handshake, and Cursor reports
`Connection closed`.

### Fix (option 1: export the env vars)

Export the missing env var before launching Cursor:

```bash
export GRAFANA_URL=https://grafana.your.tenant
export GRAFANA_SERVICE_ACCOUNT_TOKEN=glsa_...
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
open -a Cursor
```

Persist the variables in your shell profile and source it before launching
Cursor if you want the optional server available in every session.

### Fix (option 2: comment out the server locally)

If you do not intend to use the optional server, remove its block from your
local `~/.cursor/mcp.json`. Do not edit the repository manifest for local
operator state.

### Why external servers are not in the plugin manifest

The plugin owns and tests `profile_and_optimize`. External servers have their
own credentials, release cycles, and local policies, so users configure them
separately.

## Quick troubleshooting matrix

| Badge / error | Failure mode | One-liner fix |
| --- | --- | --- |
| `spawn .../bin/python ENOENT` | Missing installer venv or stale local config | Rerun the installer with `--client cursor`, then reload the server in Cursor |
| `Connection closed` on an env-var-gated optional server | Optional server, gating env var unset | Either ignore (expected), or export the `*_COMMAND` env var and restart Cursor |
| `Connection closed` on a user-owned optional server | Required environment variable is unset | Export the variable and restart, or remove the local entry |
| "Logout" on `github` | OAuth / session expiry | Click "Sign in" in the MCP panel and complete the auth flow |
| Looking for an optional server (`prometheus_mcp`, `zymtrace`, ...) in the plugin section | Optional servers are operator-configured, not shipped | Add the entry to your own `~/.cursor/mcp.json` |

## Related

- [`../plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/INSTALL.md`](../plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/INSTALL.md) - client installation details
- [`../plugins/profile-and-optimize/.mcp.json`](../plugins/profile-and-optimize/.mcp.json) - Claude Code plugin server declaration
