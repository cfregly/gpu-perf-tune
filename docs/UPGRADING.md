# Upgrade gpu-perf-tune

The full release history lives in [`CHANGELOG.md`](../CHANGELOG.md).

## Moving from 0.3.x to 0.4.0

Refresh the client package or clone before reinstalling the MCP server.

Codex clone:

```bash
git pull --ff-only
make install-skills CLIENT=codex
```

Claude Code plugin:

```bash
claude plugin update profile-and-optimize@profile-and-optimize-plugins
bash "$(ls -dt ~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/*/server/install.sh | head -1)"
claude mcp get plugin:profile-and-optimize:profile_and_optimize
```

The final command should report `Status: ✔ Connected`.

The direct MCP installer and client configuration helper now require an
explicit client. Use one of the first-class paths from the repository root:

```bash
make install-mcp CLIENT=codex
make install-mcp CLIENT=claude
```

When calling either helper directly, pass `--client codex` or
`--client claude`. An invocation without `--client` exits without changing a
client configuration. This replaces the old implicit Cursor target.

Custom AIPerf command values must not contain `--api-key`. Pass the managed API
key option to the surrounding command, or set its documented environment
variable. The runtime supplies the key through the child process environment
or Kubernetes standard input and keeps it out of receipts and result payloads.

Refresh the development environment before running the new language gates:

```bash
bash plugins/profile-and-optimize/server/install.sh --with-dev
make all
```

## Moving from 0.2.x to 0.3.x

### Codex

Update the clone, refresh the linked skills, reinstall the MCP server, and
check the registered server:

```bash
git pull --ff-only
make install-skills CLIENT=codex
make install-mcp CLIENT=codex
codex mcp get profile_and_optimize
```

The check should report `enabled: true`. Restart Codex before invoking a skill
or MCP tool.

### Claude Code

Update the plugin, reinstall the MCP server inside the current plugin cache
entry, and check the namespaced server:

```bash
claude plugin update profile-and-optimize@profile-and-optimize-plugins
bash "$(ls -dt ~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/*/server/install.sh | head -1)"
claude mcp get plugin:profile-and-optimize:profile_and_optimize
```

The check should report `Status: ✔ Connected`. Restart Claude Code before
invoking a skill or MCP tool.

### Required 0.3 command changes

- AI tuning matrix, proposal, report, validation, and experiment creation
  commands require an explicit `--space` file.
- MLPerf rules validation requires an explicit `--rules` file.
- Do not place acknowledgement flags in raw MCP `args`. Job submission uses
  `i_understand_this_submits_jobs=true`. External publication uses
  `i_understand_this_publishes_externally=true`.

These checks fail closed when a required file or structured acknowledgement is
missing. Review the command before adding either acknowledgement.

## Moving from 0.3.0 to 0.3.1

Version `0.3.1` adds security hardening without a new client configuration or
operator argument migration. Run the relevant client update steps above and
restart the client.
