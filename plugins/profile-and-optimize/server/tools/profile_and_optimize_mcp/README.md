# profile_and_optimize MCP server

See [`README.md`](/plugins/profile-and-optimize/server/tools/README.md) or the nearest parent index for broader context.

`profile_and_optimize` exposes the cluster-performance operator surface
(which includes the bundled MLPerf Training v6.0 tooling) as local MCP tools.
The package is `profile_and_optimize_mcp` and the configured MCP server key
is `profile_and_optimize`.

The runtime imports [`../../mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py), which
derives one MCP tool per CLI verb across **8 libraries** from the live
parser contracts: `selector`, `contention`, `ai_tuning`, `profile`,
`perf_baseline`, `evidence`, `slurm`, `experiments`, `findings`,
`k8s_launch`, `perf_tune_report`, and `known_good_config`. There is no
hand-maintained
registry of contract-derived tools to keep in sync. The canonical counts
live in [`../../mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s `_TOTAL_*`
constants. Two auxiliary read-only `search_*` tools are registered
directly in [`server.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/server.py). See "Auxiliary
MCP-only tools" below.

## Install

```bash
tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client cursor
```

Then add the `profile_and_optimize` block from [`INSTALL.md`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/INSTALL.md) to
your client config and restart the client. `INSTALL.md` covers Cursor,
Claude Code, Codex, Gemini CLI, and Google Antigravity.

## Safety

Every tool resolves to one explicit safety class, surfaced as the `safety`
field on every MCP response (and `ack_required` for the gate state):

- `read_only` -- never writes to disk. Reads / prints only.
- `writes_artifacts` -- writes local evidence under an operator-selected path.
- `submits_jobs` -- CLI verb can submit Slurm jobs when its ack flag is passed.
- `pulls_data` -- CLI verb can pull license-gated data when its ack flag is passed.

Example mutating call:

```json
{
  "args": ["--bench", "llama31_8b", "--nodes", "8", "--dry-run"],
  "i_understand_this_submits_jobs": true
}
```

The runtime translates `i_understand_this_*` params into the matching CLI ack
flag from the contract. Without that field, the CLI receives no ack flag and
keeps its own dry-run / fail-fast behavior.

## Tool I/O Contract

MCP tools take one optional `params` object.

Accepted `params` fields:

- `args`: CLI arguments forwarded to the underlying verb. Use a list of
  strings. A single string is normalized to a one-item list.
- `allow_nonzero`: when `true`, return a non-zero command result instead
  of raising a runtime error. Default is fail-fast.
- `i_understand_this_*`: explicit acknowledgement gates for mutating
  tools. The response includes the exact `ack_field`.

Every wrapped CLI response uses this envelope:

- `tool`: MCP tool name.
- `library`: one of the 8 libraries listed above.
  Auxiliary `search_*` tools use `library: "mcp_aux"`.
- `verb`: underlying CLI verb.
- `safety`: one of `read_only`, `writes_artifacts`, `submits_jobs`,
  `pulls_data`, or `substitutes_nodes`.
- `ack_required`: whether the verb has a CLI ack flag.
- `ack_field`: MCP param name that forwards the CLI ack flag.
- `args`: exact argv forwarded after the verb.
- `returncode`, `stdout`, `stderr`: raw subprocess result.
- `json`: parsed stdout when the CLI emits JSON, `null` otherwise.

This contract is deliberately thin. Tools stay composable because agents can
pass through CLI-specific flags in `args` while relying on the shared safety,
ack, and output envelope.

## Tool Surface

Run:

```bash
python3 mcp_surface.py counts   # verify canonical-counts constants
python3 mcp_surface.py list     # enumerate every derived tool
```

The expected surface is **51 contract-derived tools plus 2 auxiliary
search tools (53 MCP tools total)** across 8 libraries.
The canonical counts are read from [`../../mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s
`_TOTAL_*` constants. The per-verb roster is intentionally not
duplicated here so it does not drift. Drift is asserted away at commit
time by [`scripts/lint-tool-counts.py`](../../../../../scripts/lint-tool-counts.py)
+ [`scripts/lint-skill-counts.py`](../../../../../scripts/lint-skill-counts.py).

Auxiliary MCP-only tools (registered directly in
[`server.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/server.py). Not derived from any CLI
verb, `library: "mcp_aux"`, `verb: "search"`, `safety: "read_only"`):

- `search_runbooks` -- `rg`-backed search across `runbooks/` and `docs/`.
- `search_evidence` -- `rg`-backed search across `experiments/artifacts/`.

Both auxiliary tools take `query: str` plus an optional `limit: int` and
return the standard envelope. See
[`../../docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md)
"Auxiliary MCP-only tools" for details.

Static MCP resources expose the core runbooks, the CLI contract, the
MCP I/O contract, and the MCP composition router to MCP clients.
