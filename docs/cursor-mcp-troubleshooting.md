# Cursor MCP troubleshooting for `profile-and-optimize`

This document covers the common failure modes for the `profile_and_optimize` MCP server (and optional sibling servers) when used inside Cursor's MCP panel, and the one-liner that resolves each:

1. **`profile_and_optimize` plugin: `spawn .../v1.0.1/server/.venv/bin/python ENOENT`** - stale plugin path after a marketplace version bump.
2. **Env-var-gated optional servers: `Connection closed`** - expected behavior when the gating env var is unset.
3. **OAuth-backed MCPs (e.g., `github`): "Logout" badge** - OAuth token / session cookie expiry.

If you are arriving here from a screenshot of red badges in the Cursor MCP panel, walk down this list in order.

## 1. `profile_and_optimize` plugin: `spawn ... ENOENT` after a version bump

### Symptom

The MCP panel shows the `profile_and_optimize` server with a red badge and an error like

```
spawn /Users/<you>/.../plugins/profile-and-optimize/v1.0.1/server/.venv/bin/python ENOENT
```

where the path includes an older marketplace version (e.g., `v1.0.1`) that no longer exists on disk.

### Root cause

Cursor's MCP server registry caches the absolute path to the previous plugin version's Python venv. When `claude plugin update` (or `make refresh-symlinks`) replaces that venv with the new version's, Cursor still tries to spawn the old one.

### Fix

```bash
cd /path/to/profile-and-optimize-checkout
git pull origin main
claude plugin update profile-and-optimize@profile-and-optimize-plugins
make refresh-symlinks
```

then, **inside Cursor**, open the MCP panel (`Settings -> MCP` or the gear icon next to the chat input), find the `profile_and_optimize` server, and toggle the green slider off and then on. That forces Cursor to re-read the new server descriptor from disk.

If the badge is still red after the toggle, fully restart Cursor (`Cmd-Q` on macOS) - the in-memory cache is cleared on cold start.

### Why this isn't a `make` target

`make refresh-symlinks` already refreshes the on-disk plugin layout. The Cursor-side toggle is workstation state (per-operator, per-Cursor-window) and is not something the marketplace can drive from a shell script. The toggle step is documented as part of the release ritual in [`../CONTRIBUTING.md`](/CONTRIBUTING.md#release-ritual).

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

Only `profile_and_optimize`, `grafana`, and `github` ship in the plugin's [`.mcp.json`](/plugins/profile-and-optimize/.mcp.json). Optional servers (e.g. `prometheus_mcp`, `zymtrace`) do not ship in the plugin install - if you want them, add their entries to your own `~/.cursor/mcp.json` (Cursor) or `~/.claude/settings.json` (Claude Code).

## 4. `.mcp.json` env-var-placeholder entries park as red `Connection closed`

### Symptom

A server declared in [`../plugins/profile-and-optimize/.mcp.json`](/plugins/profile-and-optimize/.mcp.json) (e.g. `grafana`, `github`) shows a red `Connection closed` badge in the MCP panel, with no helpful error message, immediately after Cursor starts.

### Root cause

The manifest entry references an env var via `${VAR}` (no `:-default` fallback). When you haven't exported `VAR` before launching Cursor, Cursor expands the placeholder to the literal string `${VAR}` and passes it to Docker / npx / the binary, which fails before any MCP handshake. Cursor reports this as `Connection closed`. (Claude Code skips servers whose env vars are unset. Cursor does not - hence the asymmetry.)

### Fix (option 1: export the env vars)

Export the missing env var before launching Cursor:

```bash
export GRAFANA_URL=https://grafana.your.tenant
export GRAFANA_SERVICE_ACCOUNT_TOKEN=glsa_...
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
open -a Cursor
```

Persist these in your shell profile and source it before launching Cursor for a permanent fix.

### Fix (option 2: comment out the server locally)

If you don't intend to use `grafana` or `github` via the plugin manifest, comment out the corresponding block in your local `~/.cursor/mcp.json` (NOT the in-repo `.mcp.json`). The repo manifest is the marketplace artifact. Your local file is operator state.

### Why `grafana` and `github` are in `.mcp.json`

The manifest keeps `profile_and_optimize` (bundled, mandatory), `grafana`, and `github`. `grafana` ships as a straightforward Docker image and `github` has a `:-default` fallback on the command (`${GITHUB_MCP_COMMAND:-docker}`). These are the two external servers most operators have credentials for. Servers whose every dimension would be placeholder-gated stay operator-side optional instead.

## Quick troubleshooting matrix

| Badge / error | Failure mode | One-liner fix |
| --- | --- | --- |
| `spawn .../v<old-version>/.../python ENOENT` | Stale path after version bump | `claude plugin update profile-and-optimize@profile-and-optimize-plugins && make refresh-symlinks` + toggle the server in Cursor MCP panel |
| `Connection closed` on an env-var-gated optional server | Optional server, gating env var unset | Either ignore (expected), or export the `*_COMMAND` env var and restart Cursor |
| `Connection closed` on `grafana`, `github` | Unguarded `${VAR}` placeholder, env var unset | Either export the env var and restart, or comment out the block in `~/.cursor/mcp.json` |
| "Logout" on `github` | OAuth / session expiry | Click "Sign in" in the MCP panel and complete the auth flow |
| Looking for an optional server (`prometheus_mcp`, `zymtrace`, ...) in the plugin section | Optional servers are operator-configured, not shipped | Add the entry to your own `~/.cursor/mcp.json` |

## Related

- [`../CONTRIBUTING.md`](/CONTRIBUTING.md#release-ritual) - release ritual including `make refresh-symlinks`
- [`../plugins/profile-and-optimize/.mcp.json`](/plugins/profile-and-optimize/.mcp.json) - server declarations
