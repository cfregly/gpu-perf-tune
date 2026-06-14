Status: Active
Audience: contributors implementing or reviewing any of the 12 contract-bearing libraries. The MCP server that auto-derives its tool surface from these CLIs.

# CLI Contract for the 12 contract-bearing libraries

This document is the single source of truth for the operator-facing CLI verbs the active surface exposes. The CLIs under [`selector/`](/plugins/profile-and-optimize/server/selector), [`contention/`](/plugins/profile-and-optimize/server/contention), [`ai_tuning/`](/plugins/profile-and-optimize/server/ai_tuning), [`profile/`](/plugins/profile-and-optimize/server/profile), [`perf_baseline/`](/plugins/profile-and-optimize/server/perf_baseline), [`evidence/`](/plugins/profile-and-optimize/server/evidence), [`slurm/`](/plugins/profile-and-optimize/server/slurm), [`experiments/`](/plugins/profile-and-optimize/server/experiments), [`findings/`](/plugins/profile-and-optimize/server/findings), [`k8s_launch/`](/plugins/profile-and-optimize/server/k8s_launch), [`perf_tune_report/`](/plugins/profile-and-optimize/server/perf_tune_report), and [`known_good_config/`](/plugins/profile-and-optimize/server/known_good_config) implement every verb listed below. The MCP server in [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py) introspects the same parsers and registers one MCP tool per verb (76 in total).

The 4 cluster-performance libraries inherited from the original seed (`selector`, `contention`, `ai_tuning`, `profile`) account for 27 verbs. The 8 profile-and-optimize-native libraries (`perf_baseline`, `evidence`, `slurm`, `experiments`, `findings`, `k8s_launch`, `perf_tune_report`, `known_good_config`) account for the remaining 49 verbs, with `perf_tune_report` - the inference perf-tuning campaign engine - carrying 29 of them.

Three of the seed libraries are shim packages: [`contention/`](/plugins/profile-and-optimize/server/contention), [`ai_tuning/`](/plugins/profile-and-optimize/server/ai_tuning), and [`profile/`](/plugins/profile-and-optimize/server/profile) re-export the canonical implementations from `tools/pipeline/gb300/contention/contention_cli.py`, `tools/ai_tuning/ai_tuning.py`, and `tools/pipeline/submission/profile/profile_cli.py` respectively. The profile-and-optimize-native libraries follow the same pattern: each ships a thin `<library>/cli.py` that re-exports from `tools/<library>/<library>_cli.py`. Only `selector` carries its implementation in the package itself. The shims exist so the MCP surface can resolve `<repo_root>/<library>/cli.py` the same way for every library.

The contract is small on purpose. Every verb fits the same schema, every safety class derives from a single flag, and every verb exposes a JSON output mode. New operator-grade workflows add a verb here. Lower-level helpers stay as importable Python and do not get a CLI surface unless an operator workflow needs them.

## Conventions

- **Library names**: `selector`, `contention`, `ai_tuning`, `profile`, `perf_baseline`, `evidence`, `slurm`, `experiments`, `findings`, `k8s_launch`, `perf_tune_report`, `known_good_config`. Each is invokable as `python -m <library>`. The `selector` package additionally ships the `mlperf-selector` console script and `perf_tune_report` ships the `perftunereport` console script. The remaining ten packages are invoked through `python -m <library>` only.
- **MCP tool names**: `<library>_<verb>` with hyphenated verbs converted to underscores (`gate-256n` -> `selector_gate_256n`).
- **Safety classes** derive from the verb's flag set:
  - `read_only` - no `--i-understand-*` flag and no `--out-dir`-by-default writes.
  - `writes_artifacts` - takes `--out-dir` and writes there (no ack flag), or carries `--i-understand-this-stages-artifacts`.
  - `submits_jobs` - carries `--i-understand-this-submits-jobs`.
  - `pulls_data` - carries `--i-understand-this-pulls-license-gated-data`. Defined for license-gated dataset pulls. No active verb currently uses it.
  - `substitutes_nodes` - carries `--i-understand-this-substitutes-nodes`. Mutates Slurm cluster state (node drain/resume) without submitting new jobs. Used by `slurm drain`, `slurm resume`, and `slurm quiet_window`.
- **Output mode**: every verb supports `--json` for machine-consumable output. Verbs that only print human text MUST still expose `--json` (return a single-key payload at minimum).
- **Help line**: `--help` returns at least one paragraph. The first line is a single-sentence description that fits in the verb table below.
- **Dry-run pattern**: every verb that can mutate state defaults to dry-run when no `--i-understand-*` flag is set. The `--dry-run` flag is also accepted explicitly for symmetry.

## Verb matrix

Every verb in this table has the columns shown, plus a "Required flags" and "Optional flags" subsection in the verb spec below.

| Library | Verb | Safety | One-line description |
| --- | --- | --- | --- |
| `ai_tuning` | `space` | `read_only` | Print a tuning space manifest (parameters, ranges, defaults). |
| `ai_tuning` | `matrix` | `writes_artifacts` | Generate a bounded cartesian proposal matrix from a tuning space. |
| `ai_tuning` | `optimizer` | `writes_artifacts` | Optimizer-backed proposal helpers (subverb: propose / status / history / compare / import-hyp). |
| `ai_tuning` | `report` | `read_only` | Build a bounded JSON tuning report from raw results, ledger, and fabric evidence. |
| `ai_tuning` | `finalize` | `writes_artifacts` | Assemble or dry-run a benchmark result bundle from a tuned run. |
| `ai_tuning` | `proposal` | `read_only` | Proposal helpers (subverb: validate / diff). |
| `ai_tuning` | `template-patch` | `writes_artifacts` | Template patch helpers (subverb: validate, --apply mutates files). |
| `ai_tuning` | `experiment` | `writes_artifacts` | Experiment ledger helpers (subverb: create / update / summary / submit / poll / collect). The `submit` subverb takes its own `--i-understand-this-submits-jobs` ack flag. |
| `profile` | `host-overhead` | `writes_artifacts` | py-spy CPU sampler for the rank-0 process (subverb: top / record / dump). |
| `profile` | `profile-diff` | `writes_artifacts` | Diff two nsys-rep files (or pre-extracted nsys-stats CSV dirs) and emit per-area NVTX / kernel / CUDA-API / NCCL delta tables. |
| `perf_baseline` | `record` | `writes_artifacts` | Register a new perf-baseline entry under `experiments/artifacts/perf-baselines/`. |
| `perf_baseline` | `diff` | `writes_artifacts` | Diff a current measurement against a registered baseline. Emit per-dimension delta + verdict. |
| `evidence` | `init` | `writes_artifacts` | Scaffold a new immutable evidence bundle directory (SOURCE.md, summary.md, commands/). |
| `slurm` | `triage` | `read_only` | Triage a failed Slurm job by parsing sacct + slurm-<jobid>.out against the failure-signature catalog. |
| `slurm` | `drain` | `substitutes_nodes` | Drain a comma-list of Slurm node names (`scontrol update State=DRAIN`) on a Slurm-on-K8s cluster, with full evidence-bundle capture. |
| `slurm` | `resume` | `substitutes_nodes` | Resume a comma-list of Slurm node names (`scontrol update State=RESUME`) on a Slurm-on-K8s cluster, with full evidence-bundle capture. |
| `slurm` | `quiet_window` | `substitutes_nodes` | Drain Slurm nodes, run an operator-supplied inner command, and ALWAYS resume via try/finally even on failure or KeyboardInterrupt. |
| `findings` | `record` | `writes_artifacts` | Append a structured finding (one row in findings.yaml) to a bundle. |
| `findings` | `render` | `read_only` | Convert findings.yaml to a presentable findings.md table grouped by severity. |
| `findings` | `diff` | `read_only` | Compare two findings.yaml files and emit a markdown drift report (new / resolved / status-changed). |
| `perf_tune_report` | `campaign_init` | `writes_artifacts` | Scaffold a new campaign directory from a YAML config (`--experiment-id` makes campaign_id == the evidence-bundle run-id). |
| `perf_tune_report` | `cell_run` | `submits_jobs` | Run one cell via vllm-sweep, aiperf, or aa backend (ack-gated). |
| `perf_tune_report` | `campaign_run` | `submits_jobs` | Campaign-level orchestrator: loops a matrix YAML, running the full drain -> deploy -> bench -> aggregate -> render pipeline per cell (ack-gated). |
| `perf_tune_report` | `atlas_aggregate` | `writes_artifacts` | Aggregate per-cell normalized.json into atlas.jsonl + coverage summary. |
| `perf_tune_report` | `report_render` | `writes_artifacts` | Render the perf-report PDF from a campaign's atlas.jsonl. Omitted SoL/kernel/DCGM pages surfaced loudly. |
| `perf_tune_report` | `report_smoke` | `read_only` | Render the PDF from the bundled synthetic fixture (no cluster needed). |
| `perf_tune_report` | `publish_to_lake` | `writes_artifacts` | Publish a campaign's atlas + provenance as Parquet to the perf-lake object store. |
| `perf_tune_report` | `import_perf_bench` | `writes_artifacts` | Import an existing inference-perf-bench bundle into a perf-report campaign as cells/<cell-id>/normalized.json. |
| `perf_tune_report` | `import_nsys` | `writes_artifacts` | Import an nsys per-kernel bundle into a campaign cell as cells/<cell-id>/kernels.json. |
| `perf_tune_report` | `import_ncu` | `writes_artifacts` | Import an ncu per-kernel bundle into a campaign cell as cells/<cell-id>/ncu_kernels.json (roofline scatter). |
| `perf_tune_report` | `import_roofline_sweep` | `writes_artifacts` | Import a prefill+decode roofline sweep bundle into per-phase campaign cells + roofline render data. |
| `perf_tune_report` | `import_variant_ab` | `writes_artifacts` | Import a cross-engine variant A/B bundle as one engine-tagged, trial-averaged cell per arm. |
| `perf_tune_report` | `import_model_eval` | `writes_artifacts` | Import an lm-eval-harness results.json into a quality cell so serving quality lands in quality_v1 on publish. |
| `perf_tune_report` | `import_workloads` | `writes_artifacts` | Import a bench-all-workloads output dir into dataset-tagged per-workload campaign cells. |
| `perf_tune_report` | `dcgm_correlate` | `writes_artifacts` | Fold a frozen DCGM measurement YAML into a campaign cell's dcgm_correlation.json (byte-grounded workload-level SoL). |
| `perf_tune_report` | `graph_diff` | `writes_artifacts` | Diff two torch.compile dynamo+inductor log dumps and emit a structured graph_diff.json + per-graph unified diffs. |
| `perf_tune_report` | `kernel_profile` | `submits_jobs` | Capture per-kernel CUDA profile from a live vLLM pod via the nsys-sidecar (ack-gated. Adds an ephemeral container). |
| `perf_tune_report` | `kernel_reproducer_scaffold` | `writes_artifacts` | Scaffold a standalone CUDA/CUTLASS kernel reproducer (.cu + build script) for white-box kernel debugging. |
| `perf_tune_report` | `capture_plan` | `writes_artifacts` | Build an exact-variant capture-reuse plan for a target campaign from local atlas rows + capture artifacts. |
| `perf_tune_report` | `materialize_capture_reuse` | `writes_artifacts` | Copy exact-match capture artifacts from a capture_plan JSON into target cells with provenance. |
| `perf_tune_report` | `raw_bench_compare` | `writes_artifacts` | Render a multi-bundle vllm-bench-serve linear-comparison PDF from a raw_bench_compare_v1 YAML manifest. |
| `perf_tune_report` | `experiments_index` | `writes_artifacts` | Enumerate all local campaigns into a cross-experiment index keyed by experiment_id. |
| `perf_tune_report` | `experiment_inventory` | `writes_artifacts` | Unify local campaigns + run-id-stamped evidence bundles into one deduped experiment count and breakdown. |
| `perf_tune_report` | `tpm_summary` | `writes_artifacts` | Roll a campaign's atlas.jsonl into a per-hardware tokens-per-minute (TPM) capacity summary for pricing. |
| `perf_tune_report` | `trend_view` | `writes_artifacts` | Longitudinal (model, variant) perf/quality trend across campaigns with regression flags. |
| `perf_tune_report` | `value_view` | `writes_artifacts` | Render the value-prop ledger: curated value-findings.yaml joined with live campaigns into a grouped status table. |
| `perf_tune_report` | `portability_view` | `writes_artifacts` | Render the lever-by-model portability matrix (validated / candidate / refuted / untested) from value-findings.yaml. |
| `perf_tune_report` | `fleet_leaderboard` | `writes_artifacts` | Render cross-model fleet leaderboards (latency tier, peak throughput, perf Pareto frontier) from local campaigns. |
| `perf_tune_report` | `champion_select` | `writes_artifacts` | Rank cross-engine variant arms under a focus metric + TPOT SLO and emit a tiered production recommendation (CHAMPION.md). |
| `known_good_config` | `record` | `writes_artifacts` | Append a new model entry to the known-good config registry (comment-preserving). |
| `known_good_config` | `check` | `read_only` | Check a deploy's serve args contain every required flag for a model (fail-closed on a missing boot-blocker). |

## Verb specs

Each block below documents one verb with its required flags, optional flags, output mode, and ack contract.


















### `ai_tuning space`

- Description: Print a tuning space manifest (parameters, ranges, defaults).
- Safety: `read_only`
- Required flags: (none)
- Optional flags: `--space`, `--names-only`, `--output`
- Output mode: `--json`
- Ack flag: none

### `ai_tuning matrix`

- Description: Generate a bounded cartesian proposal matrix from a tuning space.
- Safety: `writes_artifacts`
- Required flags: `--parameter`
- Optional flags: `--space`, `--limit`, `--output`
- Output mode: `--json`
- Ack flag: none

### `ai_tuning optimizer`

- Description: Optimizer-backed proposal helpers. Takes a subverb (`propose` | `status` | `history` | `compare` | `import-hyp`) plus subverb-specific flags. The subverb is a nested argparse subparser. All flags are on the subverb parsers, not on the optimizer parent.
- Safety: `writes_artifacts`
- Required flags: subverb (`propose` | `status` | `history` | `compare` | `import-hyp`)
- Optional flags: subverb-specific (see [`tools/ai_tuning/ai_tuning.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/ai_tuning.py) for the per-subverb flag set. Every subverb accepts an output path).
- Output mode: `--json`
- Ack flag: none

### `ai_tuning report`

- Description: Build a bounded JSON tuning report from raw results, ledger, and fabric evidence.
- Safety: `read_only`
- Required flags: (none)
- Optional flags: `--space`, `--raw-results-dir`, `--raw-benchmark`, `--min-runs`, `--objective`, `--gb300-fabric-dir`, `--gb300-fabric-require-clean`, `--gb300-node-selection-dir`, `--gb300-fabric-localization-dir`, `--ledger`, `--remaining-limit`, `--template-hint-file`, `--template-hint-limit`, `--error-limit`, `--output`
- Output mode: `--json`
- Ack flag: none

### `ai_tuning finalize`

- Description: Assemble or dry-run a benchmark result bundle from a tuned run.
- Safety: `writes_artifacts`
- Required flags: `--log-dir`, `--workdir`, `--results-dir`
- Optional flags: `--benchmark`, `--run-id`, `--required-runs`, `--launcher-file`, `--dry-run`, `--output`
- Output mode: `--json`
- Ack flag: none

### `ai_tuning proposal`

- Description: Proposal helpers. Takes a subverb (`validate` | `diff`) plus subverb-specific flags. The subverb is a nested argparse subparser. All `--*` flags are on the subverb parsers, not on the `proposal` parent.
- Safety: `read_only`
- Required flags: subverb (`validate` | `diff`)
- Optional flags: subverb-specific (see [`tools/ai_tuning/ai_tuning.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/ai_tuning.py) for the per-subverb flag set).
- Output mode: `--json`
- Ack flag: none

### `ai_tuning template-patch`

- Description: Template patch helpers. Takes a subverb (`validate`) plus subverb-specific flags. The validate subverb has an apply mode that mutates files, so the umbrella safety class is `writes_artifacts`.
- Safety: `writes_artifacts`
- Required flags: subverb (`validate`)
- Optional flags: subverb-specific (see [`tools/ai_tuning/ai_tuning.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/ai_tuning.py) for the per-subverb flag set).
- Output mode: `--json`
- Ack flag: none

### `ai_tuning experiment`

- Description: Experiment ledger helpers. Takes a subverb (`create` | `update` | `summary` | `submit` | `poll` | `collect`) plus subverb-specific flags. The submit subverb carries its own ack flag at the subparser level. Pass it through `args` when actually submitting. The umbrella stays `writes_artifacts` so the MCP runtime does not auto-append the ack flag to read-only subverbs (summary, poll).
- Safety: `writes_artifacts`
- Required flags: subverb (`create` | `update` | `summary` | `submit` | `poll` | `collect`)
- Optional flags: subverb-specific (see [`tools/ai_tuning/ai_tuning.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/ai_tuning.py) for the per-subverb flag set. The submit subverb requires its own ack flag for actual submission).
- Output mode: `--json`
- Ack flag: none (subverb-level only. The submit subverb owns its own ack flag).

### `profile host-overhead`

- Description: py-spy CPU sampler for the rank-0 process. Takes a subverb (`top` | `record` | `dump`) plus subverb-specific flags.
- Safety: `writes_artifacts`
- Required flags: subverb (`top` | `record` | `dump`)
- Optional flags: subverb-specific (see [`tools/pipeline/submission/profile/host_overhead.py`](/plugins/profile-and-optimize/server/tools/pipeline/submission/profile/host_overhead.py). Every subverb supports `--json`).
- Output mode: `--json`
- Ack flag: none

### `profile profile-diff`

- Description: Diff two nsys-rep files (or pre-extracted nsys-stats CSV dirs) and emit per-area NVTX / kernel / CUDA-API / NCCL delta tables. The legacy `--json PATH` flag has been renamed to `--json-out PATH`. The bare `--json` is now a no-op flag accepted so the MCP runtime's auto-appended `--json` does not error.
- Safety: `writes_artifacts`
- Required flags: (none, `--baseline` / `--baseline-csv-dir` mutually exclusive, same for candidate)
- Optional flags: `--baseline`, `--baseline-csv-dir`, `--candidate`, `--candidate-csv-dir`, `--baseline-label`, `--candidate-label`, `--out`, `--json-out`, `--limit`, `--scratch`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_baseline record`

- Description: Register a new perf-baseline entry under `experiments/artifacts/perf-baselines/` (workload-agnostic. Family + measurement name are operator-supplied).
- Safety: `writes_artifacts`
- Required flags: `--family`, `--measurement`, `--source`
- Optional flags: `--value`, `--unit`, `--schema`, `--notes`, `--repo-root`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_baseline diff`

- Description: Diff a current measurement against a registered baseline. Emit per-dimension delta + GREEN/YELLOW/RED verdict.
- Safety: `writes_artifacts`
- Required flags: `--baseline`, `--current`
- Optional flags: `--tolerance-percent`, `--tolerance-absolute`, `--repo-root`, `--json`
- Output mode: `--json`
- Ack flag: none

### `evidence init`

- Description: Scaffold a new immutable evidence bundle directory ready for the reproducibility-grade-evidence rule (SOURCE.md, summary.md, commands/).
- Safety: `writes_artifacts`
- Required flags: `--family`, `--intent`
- Optional flags: `--run-id`, `--repo-root`, `--json`
- Output mode: `--json`
- Ack flag: none

### `slurm triage`

- Description: Triage a failed Slurm job by parsing sacct + slurm-<jobid>.out against the failure-signature catalog (OOM, NCCL hang, NCCL ECONNREFUSED, missing dataset path, missing image, node failure, signal-9, walltime exceeded, fabric error). Workload-agnostic.
- Safety: `read_only`
- Required flags: `--jobid`
- Optional flags: `--logdir`, `--repo-root`, `--json`
- Output mode: `--json`
- Ack flag: none

### `slurm drain`

- Description: Drain a comma-list of Slurm node names (`scontrol update State=DRAIN`) on a Slurm-on-K8s cluster, with full evidence-bundle capture.
- Safety: `substitutes_nodes`
- Required flags: `--nodes`
- Optional flags: `--reason`, `--ns`, `--ctl`, `--ctl-container`, `--bundle`, `--json`
- Output mode: `--json`
- Ack flag: `--i-understand-this-substitutes-nodes`

### `slurm resume`

- Description: Resume a comma-list of Slurm node names (`scontrol update State=RESUME`) on a Slurm-on-K8s cluster, with full evidence-bundle capture.
- Safety: `substitutes_nodes`
- Required flags: `--nodes`
- Optional flags: `--reason`, `--ns`, `--ctl`, `--ctl-container`, `--bundle`, `--json`
- Output mode: `--json`
- Ack flag: `--i-understand-this-substitutes-nodes`

### `slurm quiet_window`

- Description: Drain Slurm nodes, run an operator-supplied inner command, and ALWAYS resume via try/finally even on failure or KeyboardInterrupt.
- Safety: `substitutes_nodes`
- Required flags: `--nodes`, `--cmd`
- Optional flags: `--reason`, `--ns`, `--ctl`, `--ctl-container`, `--bundle`, `--json`
- Output mode: `--json`
- Ack flag: `--i-understand-this-substitutes-nodes`







### `findings record`

- Description: Append a structured finding (one row in findings.yaml) to a bundle. Used by bundle-producing workflows to capture per-finding severity, source skill, source query, recommended action, and evidence path.
- Safety: `writes_artifacts`
- Required flags: `--findings-yaml`, `--id`, `--severity`, `--source-skill`, `--source-query`, `--headline`, `--recommended-action`
- Optional flags: `--status`, `--evidence-path`, `--affected-entity`, `--notes`, `--json`
- Output mode: `--json`
- Ack flag: none

### `findings render`

- Description: Convert a findings.yaml file to a presentable findings.md table grouped by severity (RED / YELLOW / GREEN).
- Safety: `read_only`
- Required flags: `--findings-yaml`
- Optional flags: `--out`, `--json`
- Output mode: `--json`
- Ack flag: none

### `findings diff`

- Description: Compare two findings.yaml files and emit a markdown drift report (new / resolved / status-changed) for periodic re-runs of the same workflow.
- Safety: `read_only`
- Required flags: `--baseline`, `--current`
- Optional flags: `--out`, `--json`
- Output mode: `--json`
- Ack flag: none



### `perf_tune_report campaign_init`

- Description: Scaffold a new campaign directory from a YAML config. Pass `--experiment-id` to make campaign_id == the evidence-bundle run-id (the single join key across bundle / cluster label / perf-lake).
- Safety: `writes_artifacts`
- Required flags: `--config`
- Optional flags: `--slug`, `--experiment-id`, `--family`, `--evidence-bundle`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report cell_run`

- Description: Run one cell via vllm-sweep, aiperf, or aa backend (ack-gated).
- Safety: `submits_jobs`
- Required flags: `--campaign`, `--cell`, `--backend`
- Optional flags: `--serve-cmd`, `--bench-cmd`, `--namespace`, `--bench-pod`, `--kube-context`, `--endpoint-url`, `--served-model`, `--dataset-split`, `--conversation-count`, `--aa-shape`, `--aa-mode`, `--request-count`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: `--i-understand-this-submits-jobs`

### `perf_tune_report campaign_run`

- Description: Campaign-level orchestrator: loops over a matrix YAML and for each cell runs the full pipeline (drain -> deploy -> warmup -> bench -> profile-ingest -> import -> aggregate -> render -> baseline-record -> baseline-diff). Always-resume on Ctrl-C / exception via try/finally. Fail-fast on RED verdict unless `--continue-on-red` is passed.
- Safety: `submits_jobs`
- Required flags: `--config`, `--campaign`
- Optional flags: `--continue-on-red`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: `--i-understand-this-submits-jobs`

### `perf_tune_report atlas_aggregate`

- Description: Aggregate per-cell normalized.json into atlas.jsonl + coverage summary.
- Safety: `writes_artifacts`
- Required flags: `--campaign`
- Optional flags: `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report report_render`

- Description: Render the perf-report PDF from a campaign's atlas.jsonl. Omitted SoL/kernel/DCGM pages and empty charts are surfaced loudly (why + how-to-fix) on a completeness page + report_status.json, `--strict` exits non-zero when SoL is incomplete or 0 plot-ready points.
- Safety: `writes_artifacts`
- Required flags: `--campaign`
- Optional flags: `--out`, `--title`, `--variants-line`, `--data-source-line`, `--strict`, `--allow-ungrounded`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report report_smoke`

- Description: Render the PDF from the bundled synthetic fixture (no cluster needed).
- Safety: `read_only`
- Required flags: (none)
- Optional flags: `--out`, `--title`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report publish_to_lake`

- Description: Publish a campaign's atlas + provenance as Parquet to the perf-lake object store, ready for lakehouse registration. Under `--strict` the methodology gate enforces each measured row's full descriptor + its own ISL/OSL shape (see [`docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md)).
- Safety: `writes_artifacts`
- Required flags: `--campaign`
- Optional flags: `--s3-endpoint`, `--s3-bucket`, `--s3-access-key-file`, `--s3-secret-key-file`, `--if-exists`, `--strict`, `--no-strict`, `--allow-incomplete`, `--allow-ungrounded`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_perf_bench`

- Description: Import an existing inference-perf-bench bundle into a perf-report campaign as cells/<cell-id>/normalized.json. Auto-detects the bundle pattern (vLLM bench-serve text format or drive_load.py JSONL). Metadata sourced from the bundle's inference_perfbench_v1.json with `--model` / `--hardware` / ... overrides for any missing field. Records each cell's own ISL/OSL shape (per-number exact shape, no smoothing).
- Safety: `writes_artifacts`
- Required flags: `--campaign`, `--bundle`
- Optional flags: `--cell-id`, `--model`, `--hardware`, `--quant`, `--tensor-parallel`, `--parallel-strategy`, `--mtp`, `--max-num-batched-tokens`, `--max-num-seqs`, `--patched-vllm-enabled`, `--notes`, `--concurrency`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_nsys`

- Description: Import an nsys per-kernel bundle (with capture_sources.json declaring 'nsys' + nsys/cuda_gpu_kern_sum.txt from `nsys stats --report cuda_gpu_kern_sum`) into a campaign cell as cells/<cell-id>/kernels.json (zymtrace-compatible schema), which the renderer's kernel-breakdown and cross-attribution pages consume. Use when a zymtrace GPU flamegraph is unavailable.
- Safety: `writes_artifacts`
- Required flags: `--campaign`, `--cell-id`, `--bundle`
- Optional flags: `--kern-sum-name`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_ncu`

- Description: Import an ncu per-kernel bundle (with capture_sources.json declaring 'ncu' + ncu-profiles/*-sol.csv + *-raw.csv pairs) into a perf-report campaign as cells/<cell-id>/ncu_kernels.json, which the renderer's Speed-of-Light roofline scatter consumes. Handles both ncu wide and long/melted CSV shapes, `--set=basic` kernels import with measured %SoL but null arithmetic intensity. `--hw-key` selects the sol-ceilings.yaml hardware row (default b200_sm100).
- Safety: `writes_artifacts`
- Required flags: `--campaign`, `--cell-id`, `--bundle`
- Optional flags: `--hw-key`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_roofline_sweep`

- Description: Import an always-on prefill+decode roofline sweep bundle (decode_sweep.jsonl + prefill_sweep.jsonl) into a campaign. Emits cells/<id>-decode + <id>-prefill normalized.json (AtlasCell rows carrying per-(phase, concurrency/ISL) DCGM PROF utilization plus analytical roofline coords) and a roofline_sweep.json the prefill/decode roofline renderer page consumes. Metadata defaults from roofline_sweep_manifest.json. Override hardware/TP/quant/kv-dtype via flags.
- Safety: `writes_artifacts`
- Required flags: `--campaign`, `--bundle`
- Optional flags: `--cell-id`, `--model`, `--hardware`, `--quant`, `--kv-dtype`, `--kv-cache-dtype`, `--model-config`, `--tensor-parallel`, `--parallel-strategy`, `--mtp`, `--max-num-batched-tokens`, `--cache-mode`, `--dataset`, `--cudagraph-mode`, `--enforce-eager`, `--gpu-memory-utilization`, `--image`, `--delivery`, `--overlay-mode`, `--patch-files`, `--data-parallel`, `--pipeline-parallel`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_variant_ab`

- Description: Import a cross-engine variant A/B bundle (one `<arm>/c<C>-t<T>.txt` per arm) into a campaign as one cells/<arm>/normalized.json per arm, trial-averaged and engine-tagged (vllm-sweep | sglang-sweep) so both engines' arms are first-class and cross-engine-comparable. Per-arm zymtrace SoL is auto-ingested when the bundle declares capture sources. This is the first-class form of the variant-A/B path that `import_perf_bench` auto-dispatches. It feeds `champion_select`. `--require-plot-ready` hard-fails if any arm lacks the throughput-scatter fields a strict publish needs.
- Safety: `writes_artifacts`
- Required flags: `--campaign`, `--bundle`, `--model`
- Optional flags: `--hardware`, `--quant`, `--tensor-parallel`, `--parallel-strategy`, `--mtp`, `--max-num-batched-tokens`, `--cache-mode`, `--notes`, `--dataset`, `--cudagraph-mode`, `--enforce-eager`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--image`, `--delivery`, `--overlay-mode`, `--patch-files`, `--data-parallel`, `--pipeline-parallel`, `--require-plot-ready`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_model_eval`

- Description: Import an lm-eval-harness results.json into a perf-report quality cell (extra.metric_kind=eval_acc + quality_metrics) so eval-suite serving quality lands in quality_v1 on publish. The cell carries no throughput.
- Safety: `writes_artifacts`
- Required flags: `--results`, `--campaign`, `--model`, `--hardware`, `--quant`
- Optional flags: `--tensor-parallel`, `--cell-id`, `--parallel-strategy`, `--kv-cache-dtype`, `--image`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report import_workloads`

- Description: Import a bench-all-workloads output dir (one `<tag>-c<c>.txt` per workload x concurrency + bench-workloads.json) into per-workload campaign cells, each row tagged with its dataset + typed ISL/OSL - closing dataset=unknown at the source so the full multi-workload suite lands on aggregate and publish.
- Safety: `writes_artifacts`
- Required flags: `--bench-dir`, `--campaign`, `--model`, `--hardware`, `--tensor-parallel`
- Optional flags: `--quant`, `--parallel-strategy`, `--max-num-batched-tokens`, `--kv-cache-dtype`, `--image`, `--cudagraph-mode`, `--gpu-memory-utilization`, `--bench-backend`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report dcgm_correlate`

- Description: Fold a frozen DCGM measurement YAML (dcgm_frozen_v1) into a campaign cell's cells/<cell-id>/dcgm_correlation.json - the byte/FLOP workload-level Speed-of-Light grounding the renderer's DCGM page consumes. When the cell has a kernels.json, the per-category cross-attribution page is also populated. This is the byte-grounding step that flips a campaign from sol_complete-only to dcgm_grounded=true.
- Safety: `writes_artifacts`
- Required flags: `--campaign`, `--cell-id`, `--frozen-yaml`
- Optional flags: `--kernels-json`, `--ceilings`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report graph_diff`

- Description: Diff two torch.compile dynamo+inductor log dumps and emit a structured graph_diff.json + per-graph unified diffs. The operator pre-collects each side's log via `TORCH_LOGS=+dynamo,+inductor,+graph_breaks`. This verb is read-only on the cluster (parses local log files only).
- Safety: `writes_artifacts`
- Required flags: `--side-a-log`, `--side-b-log`, `--output-dir`
- Optional flags: `--side-a-label`, `--side-b-label`, `--notes`, `--dry-run`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report kernel_profile`

- Description: Capture per-kernel CUDA profile from a live vLLM inference pod via the nsys-sidecar. Uses `kubectl debug --share-processes` to attach an ephemeral container that runs nsys profile against the engine PID, then extracts .nsys-rep + summary CSVs into `--output-dir`. ALWAYS ack-gated: this mutates the cluster (adds an ephemeral container). Use `--dry-run` to print the step commands without executing.
- Safety: `submits_jobs`
- Required flags: `--namespace`, `--pod`, `--target-container`, `--output-dir`
- Optional flags: `--sidecar-image`, `--duration-seconds`, `--sample`, `--trace`, `--sampling-frequency`, `--vllm-pid-pattern`, `--bundle`, `--dry-run`, `--json`
- Output mode: `--json`
- Ack flag: `--i-understand-this-submits-jobs`

### `perf_tune_report kernel_reproducer_scaffold`

- Description: Scaffold a standalone CUDA/CUTLASS kernel reproducer (.cu + build script) for white-box kernel debugging - Track B of the [`inference-kernel-whitebox-debug`](/plugins/profile-and-optimize/skills/inference-kernel-whitebox-debug/SKILL.md) skill. Emits a self-contained harness parameterized by the GEMM dims + mirage tree + GPU arch: it instantiates the kernel template, feeds controlled inputs, and diffs against a host GEMM. The operator transcribes the exact template params from the codegen site into the marked block. Read-only on the cluster (writes local artifacts only).
- Safety: `writes_artifacts`
- Required flags: `--kernel-name`, `--header`, `--output-dir`
- Optional flags: `--mma-m`, `--mma-n`, `--batch`, `--out-dim`, `--k`, `--mirage-tree`, `--arch`, `--dry-run`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report capture_plan`

- Description: Build an exact-variant capture-reuse plan for one target campaign. Scans local atlas rows + cells/* capture artifacts, computes a conservative serving-variant signature, groups missing captures by signature, and lists exact-match reuse candidates from source campaigns. Writes the plan JSON when `--out` is supplied.
- Safety: `writes_artifacts`
- Required flags: `--campaign`
- Optional flags: `--source-campaign`, `--out`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report materialize_capture_reuse`

- Description: Copy exact-match capture artifacts from a capture_plan JSON into target cells and write capture_reuse.json provenance. Copies only candidates whose source artifact exists and whose target artifact is still absent.
- Safety: `writes_artifacts`
- Required flags: `--plan`
- Optional flags: `--dry-run`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report raw_bench_compare`

- Description: Render a multi-bundle vllm-bench-serve linear-comparison PDF from a raw_bench_compare_v1 YAML manifest. Sibling to report_render: overlays N bundles' per-concurrency curves onto a single chart per metric (throughput / TTFT / TPOT) + a peak-bars chart with %gain-vs-baseline + a summary table. Targeted at the multi-variant champion comparison where faceting hides the linear story.
- Safety: `writes_artifacts`
- Required flags: `--manifest`, `--out`
- Optional flags: `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report experiments_index`

- Description: Enumerate all local campaigns into a cross-experiment index (experiments-index.jsonl + EXPERIMENTS-INDEX.md), keyed by experiment_id, so an analyst can compare effectiveness across experiments (filter by `--family`).
- Safety: `writes_artifacts`
- Required flags: (none)
- Optional flags: `--family`, `--out`, `--include-s3`, `--s3-endpoint`, `--s3-bucket`, `--s3-access-key-file`, `--s3-secret-key-file`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report experiment_inventory`

- Description: Canonical experiment count: unify local perf-report campaigns with run-id-stamped evidence bundles (`--bundle-root`, repeatable) into one headline count + per-family/model breakdown (EXPERIMENT-INVENTORY.md + experiment-inventory.json), deduped by the run-id join key. Answers "how many experiments have we run" without the campaigns-vs-bundles ambiguity.
- Safety: `writes_artifacts`
- Required flags: (none)
- Optional flags: `--bundle-root`, `--out`, `--include-s3`, `--s3-endpoint`, `--s3-bucket`, `--s3-access-key-file`, `--s3-secret-key-file`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report tpm_summary`

- Description: Roll a campaign's atlas.jsonl into a per-hardware tokens-per-minute (TPM) capacity summary for pricing / capacity discussions. For each (model, hardware, quant, TP, strategy, MTP) group it reports a peak-capacity point and - when `--ttft-sla-ms` / `--tpot-sla-ms` are supplied - a latency-SLA-bounded point, each at per-GPU, per-replica, and per-node bases, for both output-only and total TPM. Pure post-processing of already-measured atlas data (no cluster runs).
- Safety: `writes_artifacts`
- Required flags: `--campaign`
- Optional flags: `--ttft-sla-ms`, `--tpot-sla-ms`, `--gpus-per-node`, `--context`, `--out-dir`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report trend_view`

- Description: Longitudinal (model, variant_key) perf/quality trend across campaigns: group atlas rows by the stable capture-signature variant key + concurrency, order by captured_at, flag regressions, and show the serving image (engine-version axis). Local-first. Same row shape as the lake's atlas_v1.variant_key for a published-lake pull.
- Safety: `writes_artifacts`
- Required flags: (none)
- Optional flags: `--metric`, `--concurrency`, `--regression-pct`, `--hardware`, `--out`, `--campaigns-dir`, `--lake-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report value_view`

- Description: Render the value-prop ledger: join the curated value-findings.yaml registry with live perf-lake campaigns (sol_rigor + verdict tier) into a grouped DONE / IN-PROGRESS / NOT-DONE / CLOSED-NEGATIVE table. Read-only on the lake. Flags any finding whose backing campaign is missing locally or ungrounded. Writes markdown to `--out` (or stdout).
- Safety: `writes_artifacts`
- Required flags: (none)
- Optional flags: `--registry`, `--out`, `--format`, `--title`, `--gpu-hr`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report portability_view`

- Description: Render the lever-by-model portability matrix from value-findings.yaml: rows = perf levers, columns = fleet models, cells = validated / candidate / refuted / untested, plus a per-model "try-next" candidate list. Answers "which of our proven levers should I try on model X" in one lookup. Writes markdown to `--out` (or stdout).
- Safety: `writes_artifacts`
- Required flags: (none)
- Optional flags: `--registry`, `--out`, `--title`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report fleet_leaderboard`

- Description: Render the cross-model fleet leaderboards from local campaigns: a latency-tier leaderboard, a peak tok/s/GPU throughput leaderboard, and a perf Pareto frontier for model selection. Auto-discovers every model's AA + roofline cells. Re-run after new campaigns publish.
- Safety: `writes_artifacts`
- Required flags: (none)
- Optional flags: `--hardware`, `--gpu-hr`, `--out`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `perf_tune_report champion_select`

- Description: Select the production champion: from a campaign's atlas + per-cell SoL artifacts, rank the cross-engine (vLLM + SGLang) variant arms under the focus metric (tok/s/GPU or median TPOT) + a TPOT SLO, pick the baseline + top-X, summarize each across the SoL ladder, and emit a tiered (DRAFT/VERDICT) production recommendation. A VERDICT requires the variance, multi-workload, and accuracy gates AND byte-grounding of the champion. Anything short is a DRAFT. Writes CHAMPION.md + champion_select.json. Pure post-processing - no cluster runs.
- Safety: `writes_artifacts`
- Required flags: `--campaign`
- Optional flags: `--focus`, `--focus-c`, `--top`, `--baseline`, `--metric`, `--slo-rel`, `--slo-abs-ms`, `--trials`, `--same-node`, `--require-workloads`, `--workloads-present`, `--accuracy-gate`, `--accuracy-floor`, `--out`, `--title`, `--dry-run`, `--campaigns-dir`, `--json`
- Output mode: `--json`
- Ack flag: none

### `known_good_config record`

- Description: Append a new model entry to the known-good config registry (comment-preserving): the per-model required serving flags, champion-config reference, fallback, and grind frontier.
- Safety: `writes_artifacts`
- Required flags: `--model`
- Optional flags: `--registry`, `--slug`, `--arch`, `--hardware`, `--engine`, `--required-flag`, `--champion-config-ref`, `--champion-verdict`, `--champion-campaign`, `--fallback`, `--grind-frontier`, `--notes`, `--json`
- Output mode: `--json`
- Ack flag: none

### `known_good_config check`

- Description: Check a deploy's serve args contain every required flag for a model. Fail-closed on a missing boot-blocker. Backs the [`inference-known-good-config`](/plugins/profile-and-optimize/skills/inference-known-good-config/SKILL.md) skill.
- Safety: `read_only`
- Required flags: `--model`
- Optional flags: `--registry`, `--serve-args`, `--deploy-file`, `--require-registered`, `--json`
- Output mode: `--json`
- Ack flag: none

## MCP-surface derivation

The MCP server walks all 8 CLI parsers and registers one MCP tool per verb listed above (51 contract-derived tools total. Plus 2 auxiliary search tools registered separately by the FastMCP runtime = 53 MCP tools). The canonical counts live in [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s `_TOTAL_*` constants. Tool naming and safety derivation:

- Tool name: `<library>_<verb_with_underscores>` (e.g. `selector_gate_256n`, `perf_tune_report_report_render`).
- Safety: copied from this contract's "Safety" column verbatim.
- Ack required: `True` whenever the verb has a non-empty "Ack flag" entry above.
- JSON parsing: `True` whenever the verb's "Output mode" is `--json`.

Every safety class in this contract is one of the five allowed values (`read_only`, `writes_artifacts`, `submits_jobs`, `pulls_data`, `substitutes_nodes`), [`tools/profile_and_optimize_mcp/tests/test_server_smoke.py`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/tests/test_server_smoke.py) asserts the live MCP-derived surface matches the canonical counts and safety classes.

## Out of scope

- Lower-level Python helpers (e.g. `selector.scoring`) keep their import-only API and do not get CLI verbs.
- Subverbs (e.g. `o11y print-queries` vs `o11y ingest`) are documented inside the parent verb spec. They do not appear as separate rows in the matrix.
