# profile-and-optimize beforeShellExecution hooks

Canonical, version-controlled copy of the `beforeShellExecution` guard hooks.
Two gates:

1. **Perf-lake teardown gate** (`perflake-teardown-gate.sh`) - asks for human
   confirmation before an experiment teardown (`scancel`, `helm uninstall`,
   `kubectl delete` with an `experiment=` label) destroys the evidence of an
   experiment whose bundle has not been published yet.
2. **Provenance commit gate** (`provenance-commit-gate.sh`) - surfaces a
   missing/invalid source-attribution block in a staged experiment-bundle
   `SOURCE.md` before `git commit` lands it.

## Files

| File | Role |
| --- | --- |
| `perflake-teardown-gate.sh` | Smart teardown gate. For a matched teardown command it resolves the owning evidence bundle and ALLOWS silently if the bundle is already published (or carries a `perf-lake: intentional-gap` / `perf-lake: published` marker), else returns **`ask`** so a human decides. Resolution is local + fast: it matches the recorded `experiment=<slug>` label (or run-id directory name) in `SOURCE.md`/`summary.md` or the deploy manifest. It never hits the network. Fail-OPEN on a non-teardown / parse error (never wedge normal work). Fail-SAFE (`ask`) on an unresolvable teardown (never silently destroy evidence). |
| `provenance-commit-gate.sh` | On `git commit`, audits staged experiment-bundle `SOURCE.md` files for a valid provenance block - the commit-time analog of the `publish_to_lake --strict` source gate. Phased enforcement via `$PROVENANCE_COMMIT_GATE` (`off` default / `ask` / `deny`). Fail-OPEN on any parse error or non-commit command. |
| `claude-hook-adapter.sh` | Claude Code bridge: wraps a guard so it runs as a `PreToolUse`(Bash) command hook - projects Claude's `tool_input.command`/`cwd` to the guard's Cursor-style `{command,cwd}` stdin and translates the guard's `{permission}` verdict to Claude's `{hookSpecificOutput:{permissionDecision}}`. Fail-closed (a missing/erroring guard denies). Used ONLY by `hooks.json` (Claude). Cursor calls the guards directly, so the guard scripts are unchanged across runtimes. |
| `hooks.json` / `cursor-hooks.json` | `hooks.json` = the Claude Code plugin-hooks manifest (`PreToolUse`(Bash) -> `claude-hook-adapter.sh <guard>`). `cursor-hooks.json` = the ready-to-paste Cursor `beforeShellExecution` manifest for `~/.cursor/hooks.json`. |

Both guard scripts are bash 3.2-compatible (macOS `/bin/bash`): no `mapfile`,
`${var,,}`, or associative arrays.

## Deployment model (IMPORTANT)

This directory is the **canonical, version-controlled source**. Where it is
actually enforced depends on the runtime:

- **Cursor**: profile-and-optimize is consumed via skill symlinks, NOT as a Cursor plugin,
  so this `hooks/hooks.json` is **not** auto-loaded by Cursor. Cursor enforcement
  comes from the user-level `~/.cursor/hooks.json` registering copies of these
  scripts in `~/.cursor/hooks/`. Keep `~/.cursor/hooks/` in sync with this dir
  (copy the guard `*.sh` files, `cursor-hooks.json` is the ready-to-paste
  registration block).
- **Claude Code**: profile-and-optimize is a real plugin, so `hooks/hooks.json` loads as
  plugin hooks. It wires each guard as a `PreToolUse` (matcher `Bash`) command
  hook via `claude-hook-adapter.sh` -- Claude's schema uses `PreToolUse` +
  `${CLAUDE_PLUGIN_ROOT}/...`, NOT Cursor's `beforeShellExecution` (which
  `claude plugin validate` rejects as an invalid key). The adapter bridges
  Claude's `tool_input.command` / `permissionDecision` I/O to the guards' Cursor
  `command` / `permission` contract, so the guard scripts stay unchanged.
  VERIFIED: `claude plugin validate` passes and allow / deny / ask round-trip
  through the adapter.
- **Other agents (generic)**: the `.sh` scripts are runtime-agnostic. Wire them
  into the agent's pre-tool / pre-exec hook (whatever inspects a shell command
  before it runs) using the stdin/stdout contract below.

## Hook contract (runtime-agnostic)

Each guard is a `beforeShellExecution`-style hook with a simple stdin/stdout
contract any runtime can drive:

- **stdin**: JSON with a `.command` field (the shell command about to run), e.g.
  `{"command":"git commit -m 'record results'"}`.
- **stdout**: a JSON verdict:
  - `{"permission":"allow"}` - let it run.
  - `{"permission":"deny","user_message":"...","agent_message":"..."}` - block.
  - `{"permission":"ask","user_message":"...","agent_message":"..."}` - require a
    human to approve in the runtime's UI (the agent cannot self-approve).

Test any guard directly: `echo '{"command":"<cmd>"}' | bash perflake-teardown-gate.sh`.

## Install matrix (quick reference)

| Runtime | Where to register | Notes |
| --- | --- | --- |
| Cursor | `~/.cursor/hooks.json` (user) or `.cursor/hooks.json` (project) -> `./hooks/<script>.sh` | copies of these scripts live in `~/.cursor/hooks/` |
| Claude Code | `hooks/hooks.json` (`PreToolUse`(Bash) -> `claude-hook-adapter.sh <guard>`) | validated: `claude plugin validate` passes. Guards run unchanged via the adapter |
| Other agents | the agent's pre-exec hook -> these `.sh` scripts | uses the stdin/stdout contract above |
