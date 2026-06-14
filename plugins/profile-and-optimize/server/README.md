# Bundled `profile_and_optimize` MCP server

This directory is the **source of truth** for the MCP server that the [`profile-and-optimize`](/plugins/profile-and-optimize/README.md) Claude Code plugin ships under `${CLAUDE_PLUGIN_ROOT}/server/`. After `claude plugin install`, the plugin's [`.mcp.json`](/plugins/profile-and-optimize/.mcp.json) launches `${CLAUDE_PLUGIN_ROOT}/server/.venv/bin/python -m profile_and_optimize_mcp serve` to expose **53 MCP tools** (51 contract-derived + 2 auxiliary search tools) across 8 libraries for GPU-cluster performance triage, perf-baseline + experiment workflows, and inference perf-tuning campaigns on GPU clusters. The canonical counts live in [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s `_TOTAL_*` constants and are asserted by `make lint-tool-counts` from the repo root.

See [`AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) for the ownership model and runtime-discovery contract.

## Quick install

From the plugin install directory (Claude Code resolves `${CLAUDE_PLUGIN_ROOT}` to whatever path the plugin cache lives at):

```bash
bash ${CLAUDE_PLUGIN_ROOT}/server/install.sh
```

That creates `${CLAUDE_PLUGIN_ROOT}/server/.venv/`, `pip install -e`s both the server `pyproject.toml` (the 8 stub libraries + `tools/` namespace) and the bundled `profile_and_optimize_mcp` package (the FastMCP server), and runs `mcp_surface.py counts` to verify the canonical-counts constants (`_TOTAL_LIBRARIES`, `_TOTAL_CONTRACT_TOOLS`, `_TOTAL_AUX_TOOLS`, `_TOTAL_MCP_TOOLS`) match the live derivation.

### Manual / local-dev install

If you're working on the bundled server source directly (not through the Claude Code plugin install flow):

```bash
cd <plugin-checkout>/plugins/profile-and-optimize/server
./install.sh --venv .venv --login-host "${USER}@192.0.2.10"
```

The launch envelope after install:

```bash
.venv/bin/python -m profile_and_optimize_mcp serve \
  # with these env vars:
  #   PROFILE_AND_OPTIMIZE_REPO_ROOT=<absolute path to this server/ directory>
  #   PROFILE_AND_OPTIMIZE_LOGIN_HOST=<your-user>@192.0.2.10
```

The plugin's [`.mcp.json`](/plugins/profile-and-optimize/.mcp.json) already encodes both env vars. You only need to set them by hand if you launch the server outside the plugin runtime.

## What you get

After install, the venv exposes:

| Asset | Lives at | Purpose |
| --- | --- | --- |
| `profile-and-optimize-mcp` console script | `.venv/bin/profile-and-optimize-mcp` | Same as `python -m profile_and_optimize_mcp`, but slightly faster to launch. |
| `perftunereport` console script | `.venv/bin/perftunereport` | Direct CLI entry into the `perf_tune_report` library (not via MCP). |
| `mcp_surface.py` | `<server>/mcp_surface.py` | `python mcp_surface.py counts` verifies the canonical-counts constants, `python mcp_surface.py list` enumerates every derived tool. |

## 53 tools, 8 libraries

`python mcp_surface.py list` prints the live tool surface, `python mcp_surface.py counts` confirms the canonical-counts constants in [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py) agree with the live derivation. Per-library quick reference:

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

The MCP request / response envelope (one optional `params` object, `args` list, `i_understand_this_*` ack fields) is documented in [`docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).

## Running the test suite

The bundled server ships [pytest tests](/plugins/profile-and-optimize/server/tools) under `tools/` (per-library implementation tests plus the MCP smoke test in `tools/profile_and_optimize_mcp/tests/`). To run them against a fresh venv:

```bash
bash install.sh --with-dev          # installs the `dev` extras (pytest, pyright, ruff, pre-commit, pytest-xdist)
.venv/bin/python -m pytest tools/
```

`--with-dev` is opt-in so the default install footprint stays minimal (runtime deps only). Without it, the server runs but the test suite needs a side-channel `pip install pytest` first.

## Safety classes and ack flags

Every tool's safety class is one of `read_only`, `writes_artifacts`, `submits_jobs`, `pulls_data`, or `substitutes_nodes`. Mutating tools require the matching `i_understand_this_*` field in the request. The MCP envelope returns the `ack_field` name in every response. See [`docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md) for the full contract.

## Contact

the repo author.
