# Bundled `profile_and_optimize` MCP server

This directory is the source of truth for the project MCP server. It exposes
53 tools, including 51 contract tools and 2 search tools, across 8 libraries.
The tools cover GPU performance baselines, evidence, Slurm operations, findings,
profiling, tuning, known-good configuration, and inference reports.

The server works with any local stdio MCP client. Claude Code and Codex are the
first-class install paths. The repository retains best-effort configuration
helpers for Cursor, Gemini CLI, and Google Antigravity.

Canonical counts live in
[`mcp_surface.py`](mcp_surface.py) and are
checked by `make lint-tool-counts`. See
[`AGENTS.md`](AGENTS.md) for server safety,
evidence, and runtime rules.

## Quick install

From the repository root:

```bash
bash plugins/profile-and-optimize/server/install.sh
```

That creates `server/.venv/`, installs both Python packages in editable mode,
and verifies that the live MCP surface matches its canonical counts.

To install and register the server for a supported client, use the client
installer instead:

```bash
bash plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh \
  --client codex
```

Replace `codex` with `claude`, `cursor`, `gemini`, or `antigravity`. See the
[client installation reference](tools/profile_and_optimize_mcp/INSTALL.md)
for dry runs, file fallbacks, and generic stdio configuration.

### Direct launch

```bash
cd plugins/profile-and-optimize/server
PROFILE_AND_OPTIMIZE_REPO_ROOT="$PWD" \
  .venv/bin/python -m profile_and_optimize_mcp serve
```

Client installers write the absolute server path into the generated
configuration.

## What you get

After install, the venv exposes:

| Asset | Lives at | Purpose |
| --- | --- | --- |
| `profile-and-optimize-mcp` console script | `.venv/bin/profile-and-optimize-mcp` | Same server as `python -m profile_and_optimize_mcp`. |
| `perftunereport` console script | `.venv/bin/perftunereport` | Direct CLI entry into the `perf_tune_report` library (not via MCP). |
| `mcp_surface.py` | `<server>/mcp_surface.py` | `python mcp_surface.py counts` verifies the canonical-counts constants, `python mcp_surface.py list` enumerates every derived tool. |

## 53 tools, 8 libraries

`python mcp_surface.py list` prints the live tool surface, `python mcp_surface.py counts` confirms the canonical-counts constants in [`mcp_surface.py`](mcp_surface.py) agree with the live derivation. Per-library quick reference:

| Library | Verbs | MCP tool prefix |
| --- | --- | --- |
| `ai_tuning` | 8 (`space`, `matrix`, `optimizer`, `report`, `finalize`, `proposal`, `template-patch`, `experiment`) | `ai_tuning_*` |
| `profile` | 2 (`host-overhead`, `profile-diff`) | `profile_*` |
| `perf_baseline` | 2 (`record`, `diff`) | `perf_baseline_*` |
| `evidence` | 1 (`init`) | `evidence_*` |
| `slurm` | 4 (`triage`, `drain`, `resume`, `quiet_window`) | `slurm_*` |
| `findings` | 3 (`record`, `render`, `diff`) | `findings_*` |
| `perf_tune_report` | 29 (campaign lifecycle: `campaign_init`, `campaign_run`, `cell_run`, `atlas_aggregate`, `report_render`, `report_smoke`, `publish_to_lake`. Importers: `import_perf_bench`, `import_nsys`, `import_ncu`, `import_roofline_sweep`, `import_variant_ab`, `import_model_eval`, `import_workloads`. Analysis views: `tpm_summary`, `value_view`, `trend_view`, `portability_view`, `fleet_leaderboard`, `champion_select`, `experiments_index`, `experiment_inventory`, `raw_bench_compare`, `dcgm_correlate`, `graph_diff`. Capture: `kernel_profile`, `kernel_reproducer_scaffold`, `capture_plan`, `materialize_capture_reuse`) | `perf_tune_report_*` |
| `known_good_config` | 2 (`record`, `check`) | `known_good_config_*` |
| `mcp_aux` (auxiliary, not derived) | 2 (`search_runbooks`, `search_evidence`) | `search_*` |

Total: **51 contract-derived verbs across 8 libraries + 2 auxiliary tools = 53 MCP tools**.

The MCP request / response envelope (one optional `params` object, `args` list, `i_understand_this_*` ack fields) is documented in [`docs/mcp-tool-io-contract.md`](docs/mcp-tool-io-contract.md).

## Running the test suite

The bundled server ships [pytest tests](tools) under `tools/` (per-library implementation tests plus the MCP smoke test in `tools/profile_and_optimize_mcp/tests/`). To run them against a fresh venv:

```bash
bash install.sh --with-dev
cd ../../../..
make pytest
make smoke-mcp-runtime
```

`--with-dev` installs the full contributor and CI environment, including the
report renderer and lake test dependencies. The default install stays limited
to runtime dependencies. Use `make pytest-mcp` for a focused rerun of only the
MCP and client-configuration tests.

## Safety classes and ack flags

Every tool's safety class is one of `read_only`, `writes_artifacts`,
`submits_jobs`, `pulls_data`, `substitutes_nodes`, `mutates_cluster`, or
`publishes_external`. A
tool whose contract declares an acknowledgement requires the matching
`i_understand_this_*` field in the request. Current ack-gated tools submit
jobs, change Slurm node state, or publish to external storage. Local artifact
writers use explicit output paths without a separate acknowledgement. The MCP
envelope reports `ack_required` and `ack_field`. See
[`docs/mcp-tool-io-contract.md`](docs/mcp-tool-io-contract.md)
for the full contract.

## Support

Use the public issue templates for bugs and usage questions. Report suspected
vulnerabilities through the private process in [`SECURITY.md`](../../../SECURITY.md).
