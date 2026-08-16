---
name: inference-model-optimize
license: MIT
compatibility: Requires the profile_and_optimize MCP server, a user-supplied deployment or campaign configuration, and cluster access for live runs.
metadata:
  last-validated: "2026-08-16"
description: >-
  Coordinate an evidence-backed inference model optimization run from a
  user-supplied deployment or campaign configuration. Establishes readiness
  and a production-shaped baseline, profiles measured bottlenecks, runs a
  bounded tune sweep, checks quality, selects a champion, and renders a report.
  Use for a full model optimization pass. It does not scaffold an external
  deployment repository or depend on private helper scripts.
allowed-tools: "mcp__profile_and_optimize__evidence_init mcp__profile_and_optimize__perf_tune_report_campaign_init mcp__profile_and_optimize__perf_tune_report_campaign_run mcp__profile_and_optimize__perf_tune_report_cell_run mcp__profile_and_optimize__perf_tune_report_atlas_aggregate mcp__profile_and_optimize__perf_tune_report_dcgm_correlate mcp__profile_and_optimize__perf_tune_report_import_model_eval mcp__profile_and_optimize__perf_tune_report_import_roofline_sweep mcp__profile_and_optimize__perf_tune_report_champion_select mcp__profile_and_optimize__perf_tune_report_report_render mcp__profile_and_optimize__perf_tune_report_publish_to_lake mcp__profile_and_optimize__perf_baseline_record mcp__profile_and_optimize__perf_baseline_diff mcp__profile_and_optimize__known_good_config_record mcp__profile_and_optimize__search_runbooks mcp__profile_and_optimize__search_evidence Bash(kubectl:*) Bash(helm:*) Bash(sinfo:*) Bash(squeue:*) Bash(perftunereport:*) Read Write"
---

# Inference model optimize

Coordinate the repository's existing evidence, profiling, tuning, quality, and
report tools. Do not invent a deployment scaffold. The operator must provide a
real deployment or campaign config that identifies what will run.

Every result starts as DRAFT. Promote it only when the source, delivery method,
workload, baseline, repeated trials, quality gate, and measurement receipts
support the claim.

## Required inputs

Stop if any required input is missing:

1. Model identifier and engine.
2. Exact source revision or immutable image digest.
3. Hardware and scheduler target.
4. A deployment config or campaign YAML supplied by the operator.
5. Benchmark shapes, load levels, warmup, and trial count.
6. Focus metric and accuracy floor.
7. A production-shaped baseline or enough information to capture one.

For a multi-engine comparison, the supplied campaign must define both engines,
their deployment settings, and one shared benchmark protocol. This skill does
not provide a hidden cross-engine harness.

## 1. Create one run identity

Call `evidence_init` with a family, intent, and run ID. Use the same run ID as
the experiment ID, campaign ID, and `experiment=<run-id>` label for cluster
objects.

Call `perf_tune_report_campaign_init` with that experiment ID, evidence
bundle, and the operator-supplied matrix:

```json
{
  "args": [
    "--config", "<matrix.yaml>",
    "--experiment-id", "<run-id>",
    "--family", "<family>",
    "--evidence-bundle", "<bundle-path>"
  ]
}
```

Record the returned campaign path in `SOURCE.md`.

## 2. Estimate the likely bottleneck

Use
[`inference-performance-hints`](../inference-performance-hints/SKILL.md) to
write a rough ledger for model-loading bytes, active weight bytes, prefill
FLOPs, KV bytes, collective bytes, and likely host or launch cost. State which
profile can refute each estimate. Estimates remain DRAFT.

## 3. Pass readiness before tuning

Use the operator's Kubernetes, Helm, or Slurm configuration. Confirm:

- The intended image digest or source revision is running.
- The model is Ready and a smoke request succeeds.
- The namespace, node class, GPU count, and scheduler allocation are correct.
- The benchmark client can reach the serving endpoint.
- The run ID appears on every experiment-owned object.

Capture the read-only readiness queries with
`tools/shared/capture_cmd.sh`. Stop on a load failure, crash loop, unhealthy
endpoint, or ambiguous code identity.

## 4. Capture and record the baseline

Run the production-shaped workload before changing configuration. Record exact
request shapes, concurrency, cache state, graph mode, GPU count, source,
delivery method, and trial count.

Use `perf_baseline_record` for the focus metric. Pass
`--direction higher-is-better` for throughput or
`--direction lower-is-better` for latency. Keep the baseline immutable.
Use a new run ID if the workload, source, delivery, or measurement method
changes.

## 5. Profile the measured bottleneck

Choose the smallest profile that can test the ledger:

- [`inference-workload-profile`](../inference-workload-profile/SKILL.md) for
  request-level latency and throughput shape.
- [`inference-kernel-profile`](../inference-kernel-profile/SKILL.md) for the
  kernel mix and time share.
- [`inference-kernel-ncu-profile`](../inference-kernel-ncu-profile/SKILL.md)
  for selected kernel counters.
- [`inference-dcgm-correlate`](../inference-dcgm-correlate/SKILL.md) for
  timestamp-aligned GPU telemetry.

Use `perf_tune_report_dcgm_correlate` for checked-in campaign data. Import a
roofline sweep only when the operator provides a real compatible bundle. Do not
claim that this repository supplies an external capture producer.

Rank candidate changes by measured contributor share and maximum possible
impact. Stop instruction-level tuning when setup, data movement, repeated work,
or synchronization dominates.

## 6. Run a bounded tune sweep

Follow
[`inference-tune-sweep`](../inference-tune-sweep/SKILL.md). Dry-run the campaign
first. The live `perf_tune_report_campaign_run` call requires the structured
field `i_understand_this_mutates_cluster: true`. It covers node cordon, Helm
release changes, and benchmark execution. Do not put acknowledgement flags
in raw `args`.

Rerun the production-shaped baseline after each material change. Keep one
mechanism per controlled comparison when the result will support a causal
claim.

## 7. Gate quality

Run the operator's declared quality evaluation on the baseline and each kept
candidate. Import its checked result with
`perf_tune_report_import_model_eval`. Reject a candidate that misses the
accuracy floor, changes the evaluation dataset, or lacks a reproducible
receipt.

Quantization and speculative decoding are separate workstreams. Use
[`inference-quantize-calibrate`](../inference-quantize-calibrate/SKILL.md) or
the relevant speculative decoding skill only when the operator asks for that
scope and supplies its required inputs.

## 8. Select and preserve the champion

Run `perf_tune_report_atlas_aggregate`, then `perf_baseline_diff` with a
predeclared tolerance. Call `perf_tune_report_champion_select` only after the
candidate passes workload coverage, accuracy, comparability, and repeated-trial
checks.

If the candidate wins, call `known_good_config_record` with the immutable
source or image identity and the validated required flags. If the result is
within noise or misses a gate, keep the previous configuration and report no
improvement.

## 9. Render and publish

Run `perf_tune_report_report_render` for the local artifact. Publishing is
strict by default. `--no-strict` records an intentional evidence gap and cannot
promote a result beyond DRAFT.

A dry run is local-only. A real external publish requires
`i_understand_this_publishes_externally: true` as a structured field on that
MCP call.

## Gates

| Gate | Pass condition | Failure action |
|---|---|---|
| Identity | Immutable source or image and delivery method captured | Stop |
| Readiness | Intended deployment is Ready and serves a smoke request | Stop |
| Baseline | Production-shaped workload has repeatable receipts | Stop |
| Mechanism | Profile supports the proposed change | Keep DRAFT or skip change |
| Performance | Improvement exceeds declared tolerance | Keep baseline |
| Quality | Accuracy meets the declared floor on the same evaluation | Reject candidate |
| Coverage | Required workloads and load points are present | Keep DRAFT |
| Publication | Strict validation passes | Do not publish |

## Outputs

Return the evidence bundle, campaign directory, exact source and deployment
identity, baseline, profiles, cost ledger, tested matrix, quality receipts,
baseline diff, selected config or no-improvement result, and rendered report.
State DRAFT or VERDICT explicitly.
