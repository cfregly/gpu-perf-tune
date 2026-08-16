Status: Active
Audience: new contributors setting up a local checkout.

# First hour

This path verifies the public product without requiring a GPU or cluster.

## 1. Inspect the surface

From the repository root:

```bash
make help
make demo
```

`make demo` should report 32 skills, 8 CLI libraries, and 53 MCP tools.

## 2. Install the development environment

```bash
bash plugins/profile-and-optimize/server/install.sh --with-dev
```

The installer creates an isolated environment under `plugins/profile-and-optimize/server/.venv`.

## 3. Run the local gates

```bash
make check
make smoke-mcp-runtime
make pytest
```

The MCP smoke test starts the stdio server, completes an MCP handshake, lists the tools, and calls the read-only `search_runbooks` tool.

## 4. Choose a path

- To install the MCP server into a client, read [`INSTALL.md`](../tools/profile_and_optimize_mcp/INSTALL.md).
- To add or change a skill, read [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md).
- To review tool behavior, read [`cli-contract.md`](cli-contract.md).
- To run cluster work, choose a runbook under [`runbooks/`](../runbooks).

Keep the exact command and error text when reporting a failure. Do not run live cluster actions during setup. Job submission, Slurm node changes, restricted data pulls, and external writes require an explicit acknowledgement flag.

## Terms

| Term | Meaning |
| --- | --- |
| Skills pack | The 32 installable Agent Skills under `plugins/profile-and-optimize/skills/` |
| MCP server | The client-neutral stdio server exposed as `profile_and_optimize` |
| CLI library | One of the eight parser-backed libraries used by both shell and MCP callers |
| Evidence bundle | A local directory that records inputs, commands, output, and provenance |
| Safety label | The declared effect class for a tool, such as read-only or submits-jobs |
