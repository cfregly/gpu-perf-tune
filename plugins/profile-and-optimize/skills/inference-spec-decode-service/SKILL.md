---
name: inference-spec-decode-service
license: MIT
metadata:
  last-validated: "2026-08-16"
description: >-
  Coordinate a measured speculative decoding workflow for a served model.
  Profile representative traffic, build a matched corpus, run an
  operator-provided draft trainer, compare the draft with the standing config,
  and promote only a measured win. Use for profile-matched speculative decoding
  work that needs one evidence trail and fail-closed gates.
allowed-tools: "mcp__profile_and_optimize__evidence_init mcp__profile_and_optimize__perf_tune_report_campaign_init mcp__profile_and_optimize__perf_tune_report_campaign_run mcp__profile_and_optimize__perf_tune_report_atlas_aggregate mcp__profile_and_optimize__perf_tune_report_report_render mcp__profile_and_optimize__perf_tune_report_publish_to_lake mcp__profile_and_optimize__perf_baseline_record mcp__profile_and_optimize__perf_baseline_diff mcp__profile_and_optimize__search_runbooks Bash(python3:*) Bash(kubectl:*) Bash(sinfo:*) Bash(squeue:*) Bash(sbatch:*) Read Write"
---

# Speculative decoding service workflow

## Scope

This skill coordinates checked-in profiling and evidence tools with training,
deployment, and evaluation commands supplied by the operator. The repository
does not bundle a speculative decoding controller, cluster launcher, canary
manifest, or draft training implementation.

Use this skill when a served model needs a draft matched to representative
traffic and promotion must depend on measured serving results. Use
[`inference-spec-decode-train`](../inference-spec-decode-train/SKILL.md) for a
single draft training job. Use
[`inference-spec-decode-tune`](../inference-spec-decode-tune/SKILL.md) when the
training recipe is already stable and several training configurations need a
bounded comparison.

## Required inputs

Stop before cluster work unless the operator provides all of these:

- A redacted traffic JSONL or a named workload shape set.
- The target checkpoint and current standing speculative config.
- Exact operator-owned commands or manifests for draft training, serving, and
  evaluation.
- A campaign matrix that names model identity, hardware, workload shape,
  concurrency, source revision, and trial count.
- An idle-node check appropriate for the operator's scheduler.
- An accuracy or acceptance floor and a latency objective.

## Workflow

### 1. Create one identity

Call `evidence_init`. Use the returned run ID as the experiment ID, campaign
ID, scheduler label, and join key. Store the operator-provided commands and
source revisions in the evidence bundle before execution.

### 2. Profile representative traffic

From the repository root, run the checked-in profiler:

```bash
python3 plugins/profile-and-optimize/skills/inference-workload-profile/tools/workload-profile.py \
  --in <redacted-traffic.jsonl> \
  --out <bundle>/workload-profile.json
```

For the bundled synthetic AA shapes, replace `--in` with `--aa-shapes`. Review
the input and output length distributions, concurrency, content mix, and
recommended method. A profile from a different traffic regime is not a valid
training input.

### 3. Build the matched corpus

Use the checked-in corpus converter:

```bash
python3 plugins/profile-and-optimize/skills/inference-workload-profile/tools/profile-to-corpus.py \
  --profile <bundle>/workload-profile.json \
  --traffic <redacted-traffic.jsonl> \
  --out <bundle>/corpus.jsonl
```

Verify that the target tokenizer and chat template render a sample correctly.
Record the corpus hash. Do not retain unredacted production prompts in a public
bundle.

### 4. Verify capacity, then train

Use scheduler-native read-only checks. For example, use `sinfo` and `squeue`
for Slurm, or `kubectl get nodes` and `kubectl get pods -A` for Kubernetes.
Capture the selected node and any existing workload before submitting. Never
infer idleness from a node label alone.

Hand the corpus and target checkpoint to
[`inference-spec-decode-train`](../inference-spec-decode-train/SKILL.md). The
operator-provided trainer remains the source of truth. Capture its exact argv,
stdout, stderr, exit code, and output hashes. A failed capture-correctness probe
or non-finite loss stops the workflow.

### 5. Run a controlled serving comparison

Prepare one canonical matrix YAML with `cell_id` entries for the standing and
candidate configs. Each vLLM cell must include its top-level identity fields
and a `vllm_sweep` block with operator-provided `serve_cmd` and `bench_cmd`.

Initialize the campaign with the required config:

```json
{
  "args": [
    "--config", "<matrix.yaml>",
    "--experiment-id", "<run-id>",
    "--family", "<family>",
    "--evidence-bundle", "<bundle>"
  ]
}
```

Dry-run `perf_tune_report_campaign_run` with `--config`, `--campaign`, and
`--dry-run` in `args`. Inspect placement, deployment changes, workload shape,
warmup, and cleanup. The MCP dry-run does not require a mutation
acknowledgement.

For the live call, pass the structured field
`i_understand_this_mutates_cluster: true`. Never place an acknowledgement flag
inside raw `args`.

### 6. Gate the result

Measure in-engine accepted tokens and a user-facing latency metric such as
TPOT or ITL. Run at least three matched trials on the same node and under the
same graph mode. Record separate directional baselines:

- Acceptance and throughput use `--direction higher-is-better`.
- TPOT, ITL, and TTFT use `--direction lower-is-better`.

Promote only when the candidate passes the acceptance floor and improves the
declared latency objective outside the noise bound. Training accuracy is a
proxy and cannot authorize promotion.

### 7. Publish only reviewed evidence

Aggregate and render the campaign. Keep incomplete results marked DRAFT. Use a
dry-run of `perf_tune_report_publish_to_lake` first. A live external publish
requires `i_understand_this_publishes_externally: true` for that call.

## Safety rules

- Never preempt a standing serving workload to create training capacity.
- Give every experiment object a unique name and `experiment=<run-id>` label.
- Delete only objects selected by that label. Preserve persistent volumes
  until outputs and hashes are captured.
- Keep the standing speculative config until the controlled comparison wins.
- Treat controller deployment and automatic promotion as out of scope. They
  require a separate reviewed implementation.

## Reporting

Every number must include model, hardware, quantization, parallelism, serving
config, workload shape, concurrency, warm or cold regime, source revision,
trial count, and named baseline. End with the next measured lever and the gate
that can prove or reject it.
