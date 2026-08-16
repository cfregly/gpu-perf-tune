# Agent and MCP Client Onboarding

Status: Active

`gpu-perf-tune` ships three related surfaces. Agent Skills describe the
workflows. The `profile_and_optimize` MCP server provides repeatable tools and
repository resources. Safety guards inspect risky shell commands before a
supported client runs them.

## Names

| Name | Meaning |
| --- | --- |
| `gpu-perf-tune` | The project and repository |
| `profile-and-optimize` | The skills package and Claude Code plugin adapter |
| `profile_and_optimize` | The configured MCP server key |
| `profile_and_optimize_mcp` | The Python package and module that serves MCP |

## Client coverage

| Client | Agent Skills | MCP server | Safety guards |
| --- | --- | --- | --- |
| Claude Code | First-class marketplace plugin | Plugin or repo installer | Provenance hook installed, enforcement opt-in |
| Codex | First-class repo installer into `~/.agents/skills` | Repo installer | No packaged client hook |
| Cursor, Gemini CLI, and Antigravity | Best-effort helpers | Best-effort repo installer | No release-tested adapter |
| Other stdio MCP clients | Client-owned discovery | Manual stdio configuration | No packaged adapter |

The Skill sources follow the open Agent Skills format. This repository tests
the Claude Code and Codex paths as part of the release bar. The MCP protocol
remains client-neutral. Other client helpers may work, but they do not define
release readiness.

## Install the MCP server

From the `gpu-perf-tune` repository root, choose a client:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client codex
```

First-class client names are `claude` and `codex`. The installer also accepts
the best-effort `cursor`, `gemini`, `antigravity`, and `all` helpers. Add
`--dry-run` to inspect the config changes before writing them.

The installer creates `~/.local/share/profile-and-optimize-mcp-venv`, installs
the MCP package as editable, and merges a `profile_and_optimize` server block
into the selected client config. A typical JSON block is:

```json
"profile_and_optimize": {
  "command": "~/.local/share/profile-and-optimize-mcp-venv/bin/python",
  "args": ["-m", "profile_and_optimize_mcp", "serve"],
  "env": {
    "PROFILE_AND_OPTIMIZE_REPO_ROOT": "/path/to/gpu-perf-tune/plugins/profile-and-optimize/server"
  }
}
```

The package also installs a `profile-and-optimize-mcp serve` console entry
point. The generated config uses the venv Python directly so each client finds
the editable package and its dependencies without relying on ambient Python.

Restart the selected client after installation. Full config snippets and
troubleshooting live in the
[MCP installation reference](../tools/profile_and_optimize_mcp/INSTALL.md).

## Install the Agent Skills

Claude Code installs the Skills through the `profile-and-optimize` marketplace
plugin. Codex users run this command from the repository root:

```bash
make install-skills CLIENT=codex
```

The command links each maintained Skill into `~/.agents/skills/`. Restart
Codex after it finishes. Codex can list the installed skills with `/skills` or
invoke one with `$skill-name`. These paths follow the official Codex
[Skills reference](https://developers.openai.com/codex/skills/).

The repository retains a best-effort Cursor symlink helper. Gemini CLI and
Antigravity do not have maintained skill installers here.

Use Skills for workflow behavior and the MCP tools for repeatable execution.
The MCP server also exposes repository docs and runbooks as resources.

## Install safety guards

Claude Code loads the provenance hook from the plugin. Enforcement remains off
until `PROVENANCE_COMMIT_GATE=ask` or `PROVENANCE_COMMIT_GATE=deny` is present
in the hook environment. Codex and other clients do not have a packaged hook
adapter in this repository. MCP acknowledgement fields still gate registered
external and cluster mutations for every MCP client.

Read the [guard installation reference](../../hooks/README.md)
for the exact contract and client-specific paths.

## MCP contracts and implementation

The canonical request and response contract is
[`mcp-tool-io-contract.md`](mcp-tool-io-contract.md).
[`mcp_surface.py`](../mcp_surface.py) derives
the tool surface from each library's live CLI parser. The FastMCP runtime is
[`server.py`](../tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/server.py).

Run this command from `plugins/profile-and-optimize/server/` for the current
tool names and descriptions:

```bash
python3 mcp_surface.py list
```

The derived libraries are `ai_tuning`, `profile`, `perf_baseline`, `evidence`,
`slurm`, `findings`, `perf_tune_report`, and `known_good_config`.

## Ack-gated tools

The MCP wrapper mirrors an explicit CLI acknowledgement only when the tool
contract declares one. Current fields are:

- `i_understand_this_submits_jobs=true`
- `i_understand_this_substitutes_nodes=true`
- `i_understand_this_publishes_externally=true`

The MCP wrapper refuses the operation before invoking the command when the
required acknowledgement is absent. Local `writes_artifacts` tools use an
explicit output path and do not require a separate acknowledgement.
