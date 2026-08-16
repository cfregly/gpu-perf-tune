# profile-and-optimize shell guards

The Claude Code plugin registers one guard: `provenance-commit-gate.sh`. It
checks intentionally staged evidence `SOURCE.md` files before a commit. The
guard is installed with the plugin but enforcement is off by default.

Set one of these values in the hook environment to enable it:

```bash
export PROVENANCE_COMMIT_GATE=ask
export PROVENANCE_COMMIT_GATE=deny
```

`ask` requests a human decision when provenance is missing or invalid. `deny`
blocks the commit. Both enabled modes fail closed on malformed input, validator
errors, or missing dependencies. Enabled Claude enforcement requires `jq` on
`PATH`. The validator reads the staged Git index blob, not an unstaged working
tree copy.

Generated evidence bundles are ignored by default. The gate applies when an
operator has scrubbed a bundle for publication and explicitly stages it with
`git add -f`.

## Files

| File | Role |
| --- | --- |
| `provenance-commit-gate.sh` | Detects commit commands, finds staged evidence sources, and applies the selected enforcement mode |
| `provenance-audit.py` | Bundled staged-content validator used from the current checkout |
| `claude-hook-adapter.sh` | Translates Claude Code `PreToolUse` input and guard verdicts. Missing guards and malformed verdicts fail closed |
| `hooks.json` | Claude Code plugin registration for the provenance gate |
| `cursor-hooks.json` | Best-effort manual registration example for Cursor |

## Claude Code behavior

Claude Code loads `hooks.json` with the plugin. The adapter passes the Bash
command and working directory to the provenance guard, then returns `allow`,
`ask`, or `deny` through Claude Code's hook response shape. With
`PROVENANCE_COMMIT_GATE=off`, which is the default, the guard allows the
command without auditing staged content.

The adapter and guard are covered by the default Python test suite. Validate
the package after changing them:

```bash
make pytest
make validate-claude-plugin
```

## Generic guard contract

The guard contract is small enough for another runtime to integrate manually.
Those adapters are not part of the release bar.

Input:

```json
{"command":"git commit -m 'record evidence'","cwd":"/path/to/repo"}
```

Output is one JSON object with a `permission` value of `allow`, `ask`, or
`deny`. An `ask` or `deny` response may also contain `user_message` and
`agent_message` fields.
