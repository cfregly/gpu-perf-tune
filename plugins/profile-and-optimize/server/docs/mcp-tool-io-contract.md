Status: Active
Audience: maintainers keeping MCP tool inputs and outputs reviewable.
*Last updated: June 2026 | Contact: the repo author*

# MCP Tool I/O Contract

The `profile_and_optimize` MCP server exposes 51 contract-derived MCP tools across
8 libraries (inherited from the original cluster-performance seed:
contention, ai_tuning, profile, 8 profile-and-optimize-native: perf_baseline,
evidence, slurm, experiments, findings, k8s_launch, perf_tune_report,
known_good_config), plus 2 auxiliary MCP-only tools (`search_runbooks` and
`search_evidence`) for **53 MCP tools total**. The canonical counts live in
[`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s `_TOTAL_*` constants
(`_TOTAL_CONTRACT_TOOLS`, `_TOTAL_AUX_TOOLS`, `_TOTAL_MCP_TOOLS`,
`_TOTAL_LIBRARIES`). The source of truth for the 51 contract-derived
tools is [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s `derive_tool_specs()`,
which introspects the live parsers and derives one MCP tool per CLI
verb. The auxiliary tools are registered directly in
[`tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/server.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/server.py).

## Request Shape

Each `profile_and_optimize` tool accepts one optional `params` object.

Supported `params` fields:

- `args`: CLI arguments forwarded to the underlying verb. Pass a list of
  strings. A single string is normalized to a one-item list.
- `allow_nonzero`: when `true`, return a non-zero command result instead of
  raising a runtime error. Default behavior is fail-fast.
- `i_understand_this_*`: explicit acknowledgement fields for mutating tools.
  When present and true, the runtime forwards the matching CLI ack flag.

Example mutating request (`experiments_submit`):

```json
{
  "params": {
    "args": ["--kind", "nccl_tests", "--plan", "plan.json"],
    "i_understand_this_submits_jobs": true
  }
}
```

## Response Shape

Every wrapped CLI response uses this envelope:

- `tool`: MCP tool name.
- `library`: one of `selector`, `contention`, `ai_tuning`, `profile`,
  `perf_baseline`, `evidence`, `slurm`, `experiments`, `findings`,
  `k8s_launch`, `perf_tune_report`, or `known_good_config`.
- `verb`: underlying CLI verb.
- `safety`: one of `read_only`, `writes_artifacts`, `submits_jobs`,
  `pulls_data`, or `substitutes_nodes`. The `substitutes_nodes` class
  covers verbs that mutate Slurm cluster state without submitting new
  jobs. Today that is `slurm drain`, `slurm resume`, and
  `slurm quiet_window`.
- `ack_required`: whether the verb has a CLI ack flag.
- `ack_field`: MCP parameter that forwards the CLI ack flag.
- `args`: exact forwarded argument list after ack / `--json` normalization.
- `returncode`, `stdout`, `stderr`: raw subprocess result.
- `json`: parsed stdout when the CLI emits JSON, `null` otherwise.

Example response skeleton:

```json
{
  "tool": "experiments_describe",
  "library": "experiments",
  "verb": "describe",
  "safety": "read_only",
  "ack_required": false,
  "ack_field": null,
  "args": ["--json"],
  "returncode": 0,
  "stdout": "{...}",
  "stderr": "",
  "json": {}
}
```

## Composition Rules

- Prefer `args` for CLI-specific flags rather than adding one MCP parameter
  per flag.
- Use `safety`, `ack_required`, and `ack_field` to decide whether a workflow
  can proceed automatically or needs explicit operator approval.
- Keep durable outputs under `experiments/artifacts/` unless an operator
  deliberately passes another output path.
- Use `allow_nonzero` only for diagnostic workflows that intentionally inspect
  failing command output.
- For observability-backed workflows, confirm datasource shape before live
  metrics queries and preserve raw payload provenance per
  [`perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md).

[`tools/profile_and_optimize_mcp/tests/test_server_smoke.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/tests/test_server_smoke.py)
ensures the runtime surface equals the live CLI contract.

## Required measurement context (no bare numbers)

Per the methodology rule "every performance number carries its full context (no
bare numbers)" ([`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md)), every MEASURED
atlas row (`AtlasCell` with status `full`/`partial`) MUST carry its full measurement-context
descriptor. The `perf_tune_report` publish/render path enforces this MECHANICALLY, fail-closed:
`methodology_problems()` (in `tools/perf_tune_report/lake_writer.py`) refuses a `--strict`
publish/render when any measured row leaves a descriptor field at its sentinel.

Required descriptor fields on a measured `AtlasCell` (sentinels that FAIL the gate in parens):

| field | meaning | sentinel (fails) |
| --- | --- | --- |
| `model`, `hardware`, `quant`, `tensor_parallel`, `parallel_strategy`, `mtp`, `concurrency` | identity (required at construction) | n/a |
| `max_num_batched_tokens` | shape provenance | `<= 0` |
| `cache_mode` | warm vs cold regime | `"unknown"` |
| `dataset` | workload (random/sharegpt/sonnet/aa/code) | `"unknown"` |
| `cudagraph_mode` | full/piecewise/none/eager (the eager/cudagraph trap) | `"unknown"` |
| `kv_cache_dtype` | fp8_e4m3 / bf16 / nvfp4 / … | `"unknown"` |
| `image` | serving image tag / vllm commit | `"unknown"` |
| `gpu_memory_utilization` | gmu the number was measured at | `None` |
| `data_parallel`, `pipeline_parallel` | DP replicas / PP size | default `1` (always set) |

Populate them via `import_perf_bench` flags (`--dataset`, `--cudagraph-mode` / `--enforce-eager`,
`--kv-cache-dtype`, `--image`, `--gpu-memory-utilization`, `--data-parallel`,
`--pipeline-parallel`) or the bundle's `inference_perfbench_v1.json`. A run that genuinely cannot
supply a field publishes only via the explicit `--no-strict` intentional-gap path (the gap is then
recorded on the `atlas_v1` row), never silently. The descriptor lands as `atlas_v1` columns so
every downstream view (PDF, leaderboard, value ledger) inherits the full context.

## Auxiliary MCP-only tools

Two read-only tools are not derived from a CLI verb. They are MCP-only
helpers for navigating the repo and the evidence tree:

| Tool | Searched paths | Inputs | Safety |
| --- | --- | --- | --- |
| `search_runbooks` | `runbooks/`, `docs/` | `query: str`, `limit: int = 50` | `read_only` |
| `search_evidence` | `experiments/artifacts/` | `query: str`, `limit: int = 50` | `read_only` |

Both wrap `rg --line-number --max-count <limit> <query> <paths>` and
return the same envelope as the contract-derived tools, with two fields
that mark the auxiliary nature:

- `library`: `mcp_aux` (no underlying library / package).
- `verb`: `search` (no underlying CLI verb).

The remaining envelope fields (`tool`, `safety`, `ack_required`,
`ack_field`, `args`, `returncode`, `stdout`, `stderr`, `json`) match the
contract-derived tools exactly. The structured payload appears under
`json` as `{"query": ..., "paths": [...], "matches": [...]}`.

These tools never mutate state and accept no ack flag. Use them as a thin
convenience over `rg` so agents can discover repo content through the same
MCP envelope.
