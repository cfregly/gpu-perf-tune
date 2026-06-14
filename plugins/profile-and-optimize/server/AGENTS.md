# Bundled `profile_and_optimize` MCP server

This `server/` directory is the **source of truth** for the MCP server that the [`profile-and-optimize`](/plugins/profile-and-optimize/README.md) Claude Code plugin ships. It is self-contained: a fresh clone of this repository plus `claude plugin install` provides the skills, the manifests, and the actual MCP server in one shot, with no dependency on any other repository at runtime.

Ownership of this code lives in this repository. Changes land here directly. There is no external upstream this directory tracks or syncs from.

## Why this `AGENTS.md` exists

The `profile_and_optimize_mcp` runtime ([`tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/repo.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/repo.py)) walks up the filesystem at startup looking for the first directory that contains both `AGENTS.md` and `tools/`. That is how it discovers the "repo root" it needs to file-load each library's `cli.py` for `mcp_surface.py`. This `AGENTS.md` exists to satisfy that discovery check. The operational guidance for the plugin lives in the sibling [`README.md`](/plugins/profile-and-optimize/server/README.md) and in the marketplace-level docs ([CONTRIBUTING.md](/CONTRIBUTING.md), [REVIEWERS.md](/REVIEWERS.md)).

**zymtrace query hygiene (empty != gap):** zymtrace flushes to ClickHouse asynchronously, so a query that comes back empty right after a bench is usually **ingest lag, not absence** - wait and requery before concluding. The capture-side poll lives in [`scripts/zymtrace-ingest-wait.sh`](/scripts/zymtrace-ingest-wait.sh). The recipe and the importer's fail-fast rationale are in [`docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md), cited by the zymtrace skills.

## What lives here

| Path | Role |
| --- | --- |
| `mcp_surface.py` | Derives the 51 contract-driven MCP tools (+ 2 auxiliary registered separately = 53 MCP tools total) by introspecting each library's `CONTRACT` dict. Holds the canonical-counts constants `_TOTAL_LIBRARIES=8`, `_TOTAL_CONTRACT_TOOLS=51`, `_TOTAL_AUX_TOOLS=2`, `_TOTAL_MCP_TOOLS=53`. The `counts` subcommand (`python mcp_surface.py counts`) verifies the constants against the live derivation, `scripts/lint-tool-counts.py` reads them and fails any doc that names a different number. |
| `ai_tuning/` / `profile/` | The 2 cluster-performance libraries inherited from the original seed (10 verbs: AI-assisted tuning, host/nsys profiling). Each has a `cli.py` with a `CONTRACT` dict that `mcp_surface.py` introspects and delegates the verb implementations to `tools/.../`. |
| `perf_baseline/` | Workload-agnostic perf-baseline registry. Verbs: `record`, `diff`. Backs the [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md) and [`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md) skills. |
| `evidence/` | Reproducibility-grade evidence scaffolder. Verb: `init`. Backs the [`evidence-bundle-init`](/plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md) skill. |
| `slurm/` | Workload-agnostic Slurm operations. Verbs: `triage` (read_only. Parses sacct + `slurm-*.out` against the failure-signature catalog), plus `drain` / `resume` / `quiet_window` (ack-gated `substitutes_nodes`. Slurm-on-K8s `scontrol` drain/resume and the drain-run-resume try/finally orchestrator). |
| `findings/` | Structured-findings capture for evidence bundles. Verbs: `record`, `render`, `diff`. Schema: [`docs/findings-schema.md`](/docs/findings-schema.md). |
| `perf_tune_report/` | Inference perf-tuning campaign engine and report renderer (29 verbs: campaign lifecycle, bundle importers, analysis views, capture helpers). Backs the [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) skill and ships the `perftunereport` console script. Ingests bench output via the canonical `AtlasCell` JSONL schema with first-class `failed` / `partial` / `evicted` cell handling. |
| `known_good_config/` | Per-model required-flag registry + drift check. Verbs: `record`, `check`. Backs the [`inference-known-good-config`](/plugins/profile-and-optimize/skills/inference-known-good-config/SKILL.md) skill. |
| `tools/` | Implementations the stub libraries import from. Includes `tools/profile_and_optimize_mcp/` (the FastMCP server itself, registered as the `profile-and-optimize-mcp` console script) and per-library implementation trees (`tools/perf_baseline/`, `tools/evidence/`, `tools/slurm/`, `tools/findings/`, `tools/perf_tune_report/`, `tools/known_good_config/`, `tools/ai_tuning/`, `tools/pipeline/`). |
| `runbooks/` | Searchable operator runbooks indexed by the `search_runbooks` aux tool (see `runbooks/README.md`). |
| `tuning/` | Best-known knob ledger (`tuning/best-known/`). Ships empty, populated by `ai_tuning` promotions. |
| `experiments/artifacts/learnings/` | Cited evidence bundles. The wider `experiments/artifacts/` tree is intentionally not in the repo (it's per-operator mutable evidence). |
| `pyproject.toml` | Editable-installable package containing the 8 stub libraries + the `tools/` namespace. |
| `install.sh` | Bootstrap: create `${CLAUDE_PLUGIN_ROOT}/server/.venv/`, `pip install -e` both this directory and `tools/profile_and_optimize_mcp/`, then run `mcp_surface.py counts` to confirm the canonical-counts constants match the live derivation. |
| `AGENTS.md` | This file. Satisfies the `repo.py` discovery contract. |

## What does NOT live here

- `experiments/artifacts/<other-families>/` - mutable run-record evidence. The bundled `mcp__profile_and_optimize__search_evidence` aux tool reads from whatever `experiments/artifacts/` lives under `${PROFILE_AND_OPTIMIZE_REPO_ROOT}` at runtime (which is this directory after plugin install).
- `.git/`, `.venv/`, `__pycache__/`, `*.pyc` - excluded from the install footprint.

## Discovery contract for `repo.py`

[`tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/repo.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/src/profile_and_optimize_mcp/repo.py) `find_repo_root()` either honors `${PROFILE_AND_OPTIMIZE_REPO_ROOT}` directly or walks up from `cwd` looking for an ancestor that contains both `AGENTS.md` and `tools/`. The plugin's [`.mcp.json`](/plugins/profile-and-optimize/.mcp.json) sets `PROFILE_AND_OPTIMIZE_REPO_ROOT="${CLAUDE_PLUGIN_ROOT}/server"`, which points at this directory. The combination of this `AGENTS.md` + the `tools/` directory satisfies the discovery check.

## Contact

the repo author.
