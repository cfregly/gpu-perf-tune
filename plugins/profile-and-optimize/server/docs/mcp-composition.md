Status: Active
Audience: agents and operators composing repo-local MCP tools with knowledge and observability tools.
*Last updated: June 2026 | Contact: the repo author*

# MCP Composition

This repo's MCP surface is intentionally thin: `profile_and_optimize` wraps the
checked-in CLIs and exposes repo docs as `perftune://repo/...` resources.
Use it for repeatable cluster-performance actions, then compose with read-only
knowledge and observability MCPs when the question needs context outside
the repo.

## Identity

- Package: `profile_and_optimize_mcp`
- MCP server key / FastMCP name: `profile_and_optimize`
- Repo resource URI prefix: `perftune://repo/...`
- Tool I/O contract resource: `perftune://repo/docs/mcp-tool-io-contract.md`

## Default Routing

Performance-first launch discipline applies to MCP workflows too: ground any
job submission or live-cluster action in measured evidence first, then use
external observability only to strengthen or explain that decision. Vendor
first-party guidance remains the default baseline unless the repo contains
measured evidence for a deviation.

Metrics-backed evidence discipline also applies: use your Grafana / Prometheus
and ClickHouse-backed performance data whenever those signals materially
improve a gate or diagnosis. If required telemetry is unreachable or
incomplete, fail closed or record an explicit degraded mode. Do not treat
missing metrics as all-clear. Lake-backed evidence must follow
[`perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md) so future
ClickHouse, Parquet, Iceberg, Spark, Trino, StarRocks, or other perf-lake
backends can replace the query engine without changing agent workflows.

| Need | First surface | Follow-up surface |
| --- | --- | --- |
| Runbook or repo-grounded operation | `profile_and_optimize` resource or `search_runbooks` | Local docs / tests when changing behavior |
| Repo + evidence-tree discovery | `search_runbooks` / `search_evidence` aux tools | Direct file reads once the path is known |
| AI-assisted tuning proposals / ledger / finalize | `ai_tuning_*` family (subverbs via `args`, `ai_tuning_experiment` `submit` requires its own ack flag in `args`) | Operator runs the resulting Slurm submissions through documented launch wrappers |
| Host-overhead and nsys profile-diff analysis | `profile_host_overhead` (top / record / dump subverbs) and `profile_profile_diff` for nsys-rep deltas | `perf_tune_report_kernel_profile` for live-pod kernel capture |
| Workload-agnostic perf-baseline record + diff | `perf_baseline_record` (capture any measurement with full provenance) and `perf_baseline_diff` (per-dimension delta + GREEN/YELLOW/RED verdict) | Diff verdicts feed `findings_record` rows in the evidence bundle |
| Reproducibility-grade evidence bundle scaffolding | `evidence_init` (writes SOURCE.md / summary.md / commands/) | The [`evidence-bundle-init`](/plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md) skill orchestrates the full capture |
| Workload-agnostic Slurm-job failure triage | `slurm_triage` (sacct + slurm-<jobid>.out against the failure-signature catalog) | Per-node deep dive via your cluster's node-diagnosis tooling |
| Node drain / resume / quiet window | `slurm_drain` / `slurm_resume` / `slurm_quiet_window` (ack-gated `substitutes_nodes`. Quiet_window always resumes via try/finally) | Evidence bundle captured automatically per invocation |
| Structured findings capture across evidence bundles | `findings_record` (append a finding row) / `findings_render` (findings.yaml -> findings.md) / `findings_diff` (drift report between two runs) | Schema: [`findings-schema.md`](/docs/findings-schema.md) |
| Inference perf-tuning campaign lifecycle | `perf_tune_report_campaign_init` / `cell_run` / `campaign_run` (ack-gated) / `atlas_aggregate` / `report_render` / `publish_to_lake` | The [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) skill orchestrates the full flow |
| Importing existing bench / profile bundles into a campaign | `perf_tune_report_import_perf_bench` / `import_nsys` / `import_ncu` / `import_roofline_sweep` / `import_variant_ab` / `import_model_eval` / `import_workloads` / `dcgm_correlate` | `capture_plan` + `materialize_capture_reuse` to reuse exact-variant captures across campaigns |
| Fleet-level analysis views | `perf_tune_report_fleet_leaderboard` / `value_view` / `portability_view` / `trend_view` / `champion_select` / `tpm_summary` | The [`inference-fleet-leaderboard`](/plugins/profile-and-optimize/skills/inference-fleet-leaderboard/SKILL.md) and [`inference-value-ledger`](/plugins/profile-and-optimize/skills/inference-value-ledger/SKILL.md) skills |
| Known-good serving-config drift check | `known_good_config_record` / `known_good_config_check` (fail-closed on missing boot-blockers) | The [`inference-known-good-config`](/plugins/profile-and-optimize/skills/inference-known-good-config/SKILL.md) skill |
| Anchored metrics / profiling queries | The [`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md) and [`zymtrace-anchored-query`](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md) skills against your metrics MCPs | Normalize lake-backed results per [`perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md) |
| GitHub PRs, issues, and repo metadata | GitHub MCP or `gh` CLI per task rules | Code search tooling for cross-repo examples |


## Safety Rules

- Mutating `profile_and_optimize` tools require the matching `i_understand_this_*`
  field and mirror the underlying CLI guardrail.
- `profile_and_optimize` does not replace the runbooks. It invokes the same
  canonical CLI entrypoints.
- Prefer durable repo docs over prompt-only memory. If an operational lesson
  changes a gate, status, or workflow, update the workstream source of
  truth in the same change.

## Composable MCP Server Classes

Treat repo docs and `profile_and_optimize` as the durable contract. Local MCP
descriptor names can change across workstations. The classes that compose
well with this surface:

| Class | Example servers | Role |
| --- | --- | --- |
| Repo-local operator surface | `profile_and_optimize` (this server) | Calls checked-in performance tools and exposes `perftune://repo/...` resources. |
| Knowledge and discovery | GitHub MCP, code-search MCPs | Finds code examples, PR/issue metadata, and source references. |
| Observability and perf data | Prometheus / Grafana MCPs, ClickHouse MCPs | Live metrics, logs, and profiling-backend queries. Normalize results through [`perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md). |
