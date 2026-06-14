---
name: inference-model-optimize
last_validated: 2026-06-07
description: >-
  End-to-end orchestrator that takes a NEW model from a bare HuggingFace id to a
  perf-lake-published, validated CROSS-ENGINE (vLLM + SGLang) champion on
  B200/GB300. Scaffolds a per-model harness + evidence bundle (run-id ==
  experiment-id), then drives gate-driven phases: deploy baseline -> 4-layer
  profile (zymtrace L1 / DCGM L3 / ncu L4) -> tune CROSS-ENGINE via the variant
  A/B (vLLM AND SGLang arms) -> quantize -> validate -> spec-decode
  train+validate -> bench the multi-workload suite -> champion_select (baseline
  vs top-X, the obvious production pick) -> publish_to_lake. Pauses at every red
  gate. Every number defaults to DRAFT (a champion VERDICT needs the
  multi-workload + accuracy gates + L3 byte-grounding). Triggers on "optimize a
  new model", "bring up a model end-to-end", "model bring-up pipeline", "find the
  best perf for <model>", "run the full optimization pipeline", or any
  combination of "optimize / bring-up / end-to-end / pipeline / champion" with
  "new model / inference / vllm / sglang".
allowed-tools:
  - mcp__profile_and_optimize__perf_tune_report_campaign_init
  - mcp__profile_and_optimize__perf_tune_report_campaign_run
  - mcp__profile_and_optimize__perf_tune_report_cell_run
  - mcp__profile_and_optimize__perf_tune_report_import_variant_ab
  - mcp__profile_and_optimize__perf_tune_report_atlas_aggregate
  - mcp__profile_and_optimize__perf_tune_report_dcgm_correlate
  - mcp__profile_and_optimize__perf_tune_report_import_roofline_sweep
  - mcp__profile_and_optimize__perf_tune_report_champion_select
  - mcp__profile_and_optimize__perf_tune_report_report_render
  - mcp__profile_and_optimize__perf_tune_report_publish_to_lake
  - mcp__profile_and_optimize__perf_baseline_record
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__evidence_init
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Bash(scaffold-model-bringup.sh:*)
  - Bash(kubectl:*)
  - Bash(sinfo:*)
  - Bash(squeue:*)
  - Bash(helm:*)
  - Bash(perftunereport:*)
  - Bash(run-variant-ab.sh:*)
  - Read
  - Write
---

# inference-model-optimize

> **Fast model loading (always-applied):** when standing up a vLLM deploy, never load 100s-of-GB single-stream via s3fs FUSE (~50 min for a large model). Prefer, in order: a fast-model-loading endpoint when reachable -> parallel multipart to local NVMe (the template's `stage-model-parallel.py`) -> `runai_model_streamer` (`--load-format runai_streamer`) -> tensorizer. Flag a slow load loudly: if effective rate < ~500 MB/s on a large model, STOP and switch. Details: [`server/docs/inference-fast-model-loading.md`](/plugins/profile-and-optimize/server/docs/inference-fast-model-loading.md).

## Purpose

One orchestrator for the question "given a new model, how do we find its best
inference performance on a B200 / GB300 fleet?" It does **not** re-implement
any phase -- it sequences the mature per-phase skills behind a single,
gate-driven workflow and one evidence bundle, so an operator drives the whole
profile -> tune -> calibrate -> quantize -> validate -> spec-decode -> bench ->
publish loop without hand-stitching seven skills together.

The orchestrator owns three things the individual skills do not:

1. **Phase sequencing + gates** -- each phase has an explicit go/no-go. A red
   pauses the pipeline and reports, never auto-advances.
2. **One experiment-id end to end** -- the run-id is the join key across the
   harness bundle, every cluster object's `experiment=<id-slug>` label, and the
   perf-lake `campaign=<id>` rows.
3. **Discipline enforcement** -- every number defaults to **DRAFT**. The run
   always publishes to the perf-lake with a recorded `focus` + `sol_rigor`.

## When to use

- A model the team has never deployed lands (new HF id) and you want its best
  serving config + a published roofline, not just "does it boot".
- You want to re-baseline an existing model through the full pipeline after a
  vLLM bump or a new GPU generation.
- You want a single reproducible record (one evidence bundle) that ties the
  deploy config, profiles, quant recipe, draft head, and final numbers together.

Do **not** use this skill for:

- Promoting a model to **staging / prod** -- that is a separate
  onboarding track (deploy CLI + QA suite). This harness stops at a lab champion.
- A single one-off benchmark of an already-deployed model -- use
  [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) directly.
- Any individual phase in isolation -- call that phase's skill directly
  (they are all listed under "Phases" below).

## Example prompts

- "Optimize MiniMax-M2.7 for inference end-to-end on B200."
- "Bring up `zai-org/GLM-5.2` through the full pipeline and publish the roofline."
- "Find the best perf for this new model: profile, tune, quantize, train a draft."
- `/inference-model-optimize --model minimax-m27 --hf-id MiniMaxAI/MiniMax-M2.7 --tp 8 --gpu b200`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. The profile-and-optimize bundled MCP server is installed (the `perftunereport` console
   script + `perf_baseline_*` / `evidence_*` verbs are reachable).
2. The model-bringup template workspace is checked out alongside this repo (the
   `scaffold-model-bringup.sh` scaffolder and the per-model bring-up template
   live there).
3. Live cluster access + the right **cluster profile** (`scaffold-model-bringup.sh
   --cluster-profile`): **`k8s`** = plain Kubernetes (GB300:
   `default-scheduler`, k8s `nvidia.com/gpu` authoritative, NO storageclass
   -> emptyDir + hf-pull, artifacts -> S3), **`slurm`** = Slurm-on-K8s (legacy
   B200: the Slurm scheduler pod, VAST PVC). Detect: a `slurm-controller`
   + `slurm.example.com/lock` taint => `slurm`, else `k8s`.
4. Free GPUs confirmed BEFORE any GPU phase: on **k8s** via `kubectl get nodes -l
   node.kubernetes.io/instance-type=<gpu>,cluster.example.com/state=dev` + k8s GPU
   requests. On **slurm** via `sinfo`/`squeue`. GB300 node = 4 GPU (TP8 spans 2 nodes
   via the NVLink domain). B200 = 8 GPU.
5. For the quant + spec-decode phases: a non-NVFP4 base checkpoint (BF16 or FP8)
   is reachable on HF/S3 (those phases consume the base, not the serving NVFP4).

## Interaction style

Iterative and gate-driven: run one phase,
report the gate verdict + the headline number (DRAFT unless promoted), then ask
before advancing. Never auto-advance past a red gate. The operator may start at
any phase (`--start-phase`) or skip phases that do not apply (e.g. skip Phase 6/7
for a model with no draft-head plan).

## Workflow

### Phase 0: resolve intent + scaffold

Resolve the operator's request to concrete parameters and state them back:

- `model-slug` (short, no quant suffix, e.g. `minimax-m27`), `hf-id` (base,
  e.g. `MiniMaxAI/MiniMax-M2.7`), optional `hf-nvfp4-id` (a pre-quantized
  variant if one already exists), `tp`, `gpu-type` (`b200` | `gb300`),
  `quant-target` (`nvfp4` | `fp8` | `none`), `spec-method`
  (`eagle3` | `deepseek_mtp` | `dflash` | `ngram` | `none`).
- Read the model card / `config.json` for arch family, attention type
  (MLA needs `block-size: 64`), `num_hidden_layers`, `max_position_embeddings`,
  parsers -- the same step-1 discovery a production onboarding does.

Then scaffold the per-model harness bundle + evidence bundle (one shell call,
no GPU spend):

```text
scaffold-model-bringup.sh \
  --model-slug minimax-m27 \
  --hf-id MiniMaxAI/MiniMax-M2.7 --hf-serve-id nvidia/MiniMax-M2.7-NVFP4 \
  --tp 4 --gpu-type gb300 \
  --cluster-profile k8s --ns <namespace> \
  --quant-target none --spec-method deepseek_mtp
```

(Legacy Slurm-on-K8s/B200 form: `--tp 8 --gpu-type b200 --cluster-profile slurm`.)

This renders the per-model harness bundle (03a PVC, 03b puller,
03c deploy, `my-values-minimax-m27.yaml`, `profiling/`, `quantize/`,
`spec-decode/`) and an evidence bundle at
`experiments/artifacts/model-optimize/<run-id>/` with `SOURCE.md` +
`summary.md`. **The run-id printed here is the experiment-id** -- record it. Every
later phase uses it as the `experiment=<id-slug>` label and the
`campaign=<id>` key. Report the scaffolded paths. Ask before deploying.

### Phase 1: deploy baseline (cluster-profile aware)

**First, resolve the loader (the single source of truth, not a guess).** Run the loader advisor
to pick the weight loader for this tier (decision tree + measured basis in
[`server/docs/inference-fast-model-loading.md`](/plugins/profile-and-optimize/server/docs/inference-fast-model-loading.md)
"Loader selection for a serving tier"):
`python3 tools/loader_advisor.py --serve-args "<the tier's vllm serve args>"
--hf-egress yes|no --image-has-runai yes|no --object-store yes|no
--emit experiments/artifacts/model-optimize/<run-id>/`. It writes `loader_advisor.json` + `LOADER.md`
(recommended loader + tier + per-gate pass/fail + rationale). Record the recommended loader +
rationale in the bundle `SOURCE.md`, then drive the scaffolder with it
(`scaffold-model-bringup.sh --loader auto|hf-pull|runai`) so the rendered baseline matches.
Rule of thumb the advisor encodes: **non-MTP tier -> RunAI streamer** (streams weights from
object storage straight to GPU, fastest, no HF dependency / emptyDir / FUSE), **MTP tier ->
hf-pull** (MTP-native. RunAI needs the drafter patch + double-streams).

**Then resolve the TP (right-size to active params, do not over-shard).** Run the TP right-size
advisor to size tensor-parallel to the model's ACTIVE params, not its total -- a small-active MoE
served at a TP higher than memory requires spreads ~3B active over N near-idle GPUs (the
over-provisioning trap: measured **~2.7x worse tok/s/GPU** plus flat-power OPEX waste
on the 30B-A3B class):
`python3 tools/tp_rightsize_advisor.py --total-params-b <T> --active-params-b <A>
--dtype nvfp4|fp8|bf16 --current-tp <TP> --emit experiments/artifacts/model-optimize/<run-id>/`. It
writes `tp_rightsize.json` + `TP-RIGHTSIZE.md` (recommended TP + over-provisioned flag +
~$/replica/month at matched throughput + `measured|extrapolated` confidence). Rule it encodes:
**recommend the lowest TP that fits weights in memory while keeping active >= ~3B/GPU**. It flags
OVER-PROVISIONED only when active/current_TP < 3B (so large-active tiers -- e.g. a 13B-active
model at TP4 -- are NOT touched). `extrapolated` (total > ~40B) = verify with a same-node TP A/B
before shipping. Record the recommended TP + rationale in `SOURCE.md` and deploy at that TP.

Preflight a free GPU node, then deploy the baseline (BF16 or vendor FP8/NVFP4 if
one exists) via the scaffolded bundle's `03a -> 03b -> 03c`.

- **plain-K8s GB300 (the DEFAULT):**
  `schedulerName: default-scheduler` + `resources.requests.nvidia.com/gpu`
  (k8s device-plugin is authoritative). Preflight `kubectl get nodes -l
  node.kubernetes.io/instance-type=<gpu>,cluster.example.com/state=dev` + k8s GPU
  requests==0. NO storageclass -> emptyDir + hf-pull. GB300 node = 4 GPU (TP8
  spans 2 nodes via the NVLink domain).
- **legacy Slurm-on-K8s B200 (only if you detect a `slurm-controller` +
  `slurm.example.com/lock` taint):** `schedulerName:
  <slurm-scheduler-pod>` + the lock toleration. Preflight `sinfo -N -o
  '%N %t %G'` / `squeue`. NEVER `default-scheduler` + a hard `nodeSelector` GPU
  grab on Slurm-on-K8s (it double-books GPUs Slurm has already allocated).

The deploy carries the zymtrace CUDA injection by default (copied from the
bring-up template's `profiling/experiment-deploy-template.yaml`).
Delivery: for a modified-vLLM arm, default to the runtime overlay ladder --
a subPath ConfigMap, or the initContainer patch-set
(the template's `overlay-patchset.sh`) for a whole engine patch set
on a pullable base image. Bake an image only for prod / compiled changes.
Gate: pod Ready + a curl smoke + `verify-experiment-labels.sh <id-slug> --post`
returns PASS. Record the baseline as a DRAFT perf-baseline.

### Phase 2: profile the baseline

Drive the profiling skills against the live pod (all on by default in the
template). Sequence:

- [`inference-decode-step-budget`](/plugins/profile-and-optimize/skills/inference-decode-step-budget/SKILL.md) at
  c=1..8 -- is the hot path kernel-bound / host-bound / comm-bound?
- [`analyze-zymtrace-workload`](/plugins/profile-and-optimize/skills/analyze-zymtrace-workload/SKILL.md) -- GPU+CPU
  cross-view. Which kernel family dominates.
- [`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md) (nsys) and,
  when a per-kernel which-kernel verdict is needed,
  [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md) (ncu).
- [`inference-dcgm-correlate`](/plugins/profile-and-optimize/skills/inference-dcgm-correlate/SKILL.md) for
  byte-grounded workload %SoL (raises sol_rigor to L3).
- [`inference-graph-diff`](/plugins/profile-and-optimize/skills/inference-graph-diff/SKILL.md) only when comparing
  two compile configs.

Gate: a profile actually captured (the nsys capture-validation gate must be
clear -- an empty rep is a capture bug, not a verdict). Capture all four where a
where/why verdict matters. The next phase fuses them. **Capture orchestration**
(one pod can't run all four. Use a submit-and-queue self-driving Job with a
variance-controlled bench >=3 trials/concurrency. Pass `--tokenizer <local>` in an
offline pod) is documented in
[`inference-perf-synthesize`](/plugins/profile-and-optimize/skills/inference-perf-synthesize/SKILL.md) "Capture
orchestration" -- reuse the `experiments/` Job
templates the scaffolder renders.

### Phase 2.5: synthesize the profiles into ranked recommendations

Hand off to [`inference-perf-synthesize`](/plugins/profile-and-optimize/skills/inference-perf-synthesize/SKILL.md):
fuse the four profiler artifacts (nsys + ncu + zymtrace + DCGM) into ONE ranked,
artifact-cited recommendation ledger (config knobs AND concrete vLLM-source
overlays). Every row cites its backing artifact + `sol_rigor` (the explainability
contract). Every row is DRAFT (predicted) until an A/B proves it. The top-ranked
levers seed Phase 3, and any `vllm-src` lever becomes an overlay A/B there.

### Phase 3: tune CROSS-ENGINE (vLLM + SGLang) via the variant A/B

The tuning spine is the **engine-agnostic variant A/B** -- we no longer pick an
engine by reputation. Every candidate config (vLLM AND SGLang) is an arm proven
on ONE shared bench client so the numbers are comparable by construction. Hand
off to [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md), which fills
the rendered bundle's `variants/arms.tsv` with both
engines' levers (one arm per lever. First row per engine = that engine's
baseline) and runs the same-node, >=3-trial sweep:

```text
run-variant-ab.sh readback arms-crossengine.tsv <out>   # vLLM + SGLang arms, c-sweep, SoL/arm
```

- **vLLM levers:** `max_num_batched_tokens`, `kv-cache-dtype`, `cudagraph_mode`,
  `--enable-expert-parallel`, MoE backend. Plus any Phase-2.5 `vllm-src` overlay.
- **SGLang levers:** `--moe-runner-backend`, `--attention-backend`, cuda-graph,
  `--mem-fraction-static`, `--disable-radix-cache` (match caching across engines
  BEFORE crowning a throughput champion -- a radix-cache replay can manufacture a
  phantom win).

Then `perftunereport import_variant_ab --bundle <out> --model <id>` (engine-tagged
`vllm-sweep` / `sglang-sweep` cells) -> `atlas_aggregate`. For any decode-latency
ship/no-ship claim, confirm with the same-node controlled A/B
(`run-controlled-ab.sh`. SGLang via `BENCH_TARGET=deploy/<client> BACKEND=openai`).
Gate: a champion that beats the Phase-1 baseline on the run's `focus` metric
(throughput tok/s/GPU OR median TPOT/ITL -- never output tok/s at small
`num_prompts` for a latency claim). Promote DRAFT -> VERDICT only same-node + >=3
trials. The cross-engine ranking + the production pick happen in Phase 8.4
(`champion_select`). Record the per-engine top-3 here.

### Phase 4: calibrate + quantize

Only if `--quant-target` != `none`. Hand off to
[`inference-quantize-calibrate`](/plugins/profile-and-optimize/skills/inference-quantize-calibrate/SKILL.md): a
calibration-dataset Job + a quantize Job (NVIDIA ModelOpt -> NVFP4 by default,
llm-compressor -> FP8 alternate) that writes weights + per-layer KV/activation
scales to an experiment-prefixed PVC. Gate: the quantize Job completes and the
output `config.json` declares the expected quant method. This phase is
**ack-gated** (it submits GPU jobs). Fails closed without the ack flag.

The generated `quantize/quantize.py` is driven by a **model-family registry**
(`gemma4_text` native-plugin / `qwen3_moe` + `nemotron_h` generic) that
auto-resolves the recipe + exclude scope from `config.json` `model_type` -- match
the `nvidia/*-NVFP4` sibling's `hf_quant_config` excludes for an apples-to-apples
A/B. Key gemma4/qwen3 VERDICT lessons (full detail in `inference-quantize-calibrate`
Phase 3d/3e): fused-3D experts are the #1 hazard (a naive quantize silently skips
them -- the registry + coverage guard handle it. Modelopt natively quantizes many
fused MoEs e.g. Qwen3 `_QuantSparseMoe`). Calibrate with
`create_forward_loop(cnn_dailymail)` + `HF_HUB_OFFLINE=0`. Serve **marlin-first**
(our exports can serve degenerate on cutlass). Pad the MoE intermediate only if a
backend rejects the alignment.

### Phase 5: validate the quantized weights

Deploy the quantized weights (re-use Phase 1 with the new PVC) and run
[`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md) (GPQA + MMLU-Pro) plus
a perf A/B vs the BF16/FP8 baseline. Gate: accuracy delta within the
operator-set tolerance (default: no metric drops > 1.0 absolute point) AND a perf
win. A quant that wins perf but fails accuracy is a **no-ship** -- report and
stop. On pass, the quantized weights become the new serving champion.

**Gate hygiene (the gate must measure the model, not the prompt format).** For
an instruction-tuned (`-it`) model, the quick known-answer sanity gate MUST use
`/v1/chat/completions` (the chat template), NOT raw `/v1/completions` - a bare
prompt on an `-it` model yields degenerate repetition by design and will falsely
fail a correct checkpoint. When any gate fails, first run a known-good control
(the BF16 base, or TP=1) on the SAME gate to attribute model-vs-gate before
concluding the quant is broken. (Worked failure: gemma4 NVFP4 falsely read
"garbage 0/4" on a raw-completions gate. The chat gate returned 4/4.) For
NVFP4-MoE serving blockers (activation-capability + intermediate-alignment),
see `inference-quantize-calibrate` Phase 3c.

### Phase 6: train a speculative-decoding draft

Only if `--spec-method` is a trained method (`eagle3` | `dflash`). Hand off to
[`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md): capture
target hidden states, train the draft head on the offline dataset, convert to a
vLLM-loadable draft. (For `deepseek_mtp` / `ngram` there is nothing to train --
the head is built-in or prompt-driven -- skip straight to Phase 7 with the
config lever.) This phase runs a multi-hour Slurm job. Submit-and-queue, do not
babysit. Gate: training `acc_0` / loss is non-degenerate.

### Phase 7: validate the draft (acceptance A/B)

Serve the trained draft via `--speculative-config` and measure in-engine
acceptance (the bundle's `spec-decode/dflash_vllm_eval.py` + vLLM
`spec_decode_*` Prometheus counters). Gate: a measured acceptance-length **win**
vs the standing config (e.g. MTP K=3) on a same-node A/B. A draft head that
loses (trained drafts can measure ~0% acceptance) is **do-not-ship** -- keep the standing config.

### Phase 8: bench the FULL multi-workload suite, byte-ground, roofline

Run the baseline + the top-X variants through the **full workload suite**
(`bench-all-workloads.sh`: AA + Sonnet + ShareGPT + random + code at the tier's
concurrency -- a single-workload number is a DRAFT, not a verdict. A config can
win one workload and lose the suite), aggregate into the campaign atlas, byte-ground
each cell (`dcgm_correlate`), and **always generate the prefill/decode roofline
(page 7)** via `roofline-sweep.sh` + `perf_tune_report_import_roofline_sweep` (the
"what C maxes the TFLOPs / is decode >=75% HBM / which sharding degree" answers --
see [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) Phase D3. Sweep
the baseline AND every top-X variant so the champion roofline overlay has them
all). Record which workloads ran (`<out>/workloads.txt`) for the Phase-8.4 gate.

### Phase 8.4: champion selection (the obvious production pick)

Run [`perftunereport champion_select`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) to fuse the
campaign into the single "what do we ship" deliverable:

```text
perftunereport champion_select --campaign <run-id> --top 3 \
  --focus <throughput|latency> --workloads-present <aa,sonnet,sharegpt,random,code> \
  --accuracy-gate <pass|fail|unknown> --same-node --trials 3
```

It ranks the baseline + top-X variants CROSS-ENGINE (vLLM + SGLang) under the
focus metric + TPOT SLO, summarizes each across the 4-layer SoL ladder, overlays
their rooflines, and emits `CHAMPION.md` + `champion_select.json` + the PDF
champion page with a **RECOMMENDED-FOR-PRODUCTION** banner. The recommendation is
tiered: a **VERDICT** requires variance (same-node + >=3 trials), the
multi-workload suite (`--workloads-present` covers the canonical set), the
accuracy gate (`--accuracy-gate pass`), AND L3 byte-grounding of the champion,
anything short is a **DRAFT** recommendation (the gate that failed is named in
`CHAMPION.md`). Render the PDF + `publish_to_lake` with `focus` + `sol_rigor`:
the champion lands in the perf-lake (`campaign_v1.recommended_cell` +
`champion_v1` per-variant rows) so the prod pick + its proof are queryable. The
campaign-id == the run-id == the experiment-id. Record the campaign +
`s3://perf-lake/...` paths + the recommendation back into `SOURCE.md`/`summary.md`.
Always-publish: a latency-bound / proxy / `dcgm_grounded=false` run still lands
(`--no-strict`), with the gap recorded.

### Phase 8.5: post-champion synthesis + RATCHET (a champion is the next baseline)

Re-run [`inference-perf-synthesize`](/plugins/profile-and-optimize/skills/inference-perf-synthesize/SKILL.md) against
the champion's profiles to produce the final, data-backed verdict ledger + the
ranked **future-work levers** (the recommendations not yet exhausted, each still
artifact-cited). This is the durable hand-off: the next operator (or the next
model) inherits a profiled, explained starting point instead of a bare number.
Publish the ledger into the same `campaign=<run-id>` so it lands in the perf-lake.

**This bring-up is ratcheted, NOT done** (`docs/METHODOLOGY.md` "Always be
grinding"). Close the loop:

- **The champion is the next baseline.** The won config replaces the Phase-1 baseline,
  the next grind measures against it, not the original.
- **Record the next lever in the ledger.** Add/refresh the model's entry in the operator-side
  `configs/value-findings.yaml`
  with a `next_lever:` (+ `next_value`) = the top still-open lever from the synthesis
  (the highest value-x-cost recommendation not yet exhausted). A bring-up that ships a
  champion with no queued `next_lever` is incomplete -- `perftunereport value_view` flags it.
- **Verify the frontier moved.** `perftunereport value_view` should now show this model's
  next lever on the ranked GRIND FRONTIER. Only `frontier-exhausted: <evidence>` (SoL
  at-ceiling or a K/R/H/P/A H4/P4 argument) is an allowed terminal -- everything else
  queues the next grind.
- **Escalate to the NOVEL kernel-level frontier before ANY "exhausted" claim.** When the config /
  quant / spec-decode / parallelism levers are spent you have reached the kernel-level frontier, NOT
  "frontier-exhausted" (see `docs/METHODOLOGY.md`). A byte-grounded
  config-bound conclusion (occupancy / host-gap bound, tensor-pipe far from its FLOP roofline) is the
  START of the kernel hunt: explicitly enumerate + K/R/H/P/A-classify the megakernel / persistent-kernel
  decode, the per-step op-chain FUSION (attention + MoE-route + expert-FFN + norm), and the candidate
  vLLM/SGLang kernel patches that attack the MEASURED bound resource (megakernel + fusion target exactly
  the occupancy / host-gap regime a faster GEMM cannot touch). Pursue the top, or bank each with a
  K/R/H/P/A structural-cap / infra-wall reason. `frontier-exhausted` is valid ONLY once this kernel
  frontier is assessed, never on config levers alone.

### Teardown

When done, tear down by label in PV-last order
(`kubectl delete deploy,pod,secret -l experiment=<id-slug>` -> wait on
VolumeAttachment -> `pvc` -> the experiment PV last), then
`verify-experiment-labels.sh <id-slug> --post` should return zero objects and
standing/migration objects must be untouched + Ready.

## Phase -> skill map

| Phase | Skill / tool | Gate |
| --- | --- | --- |
| 0 scaffold | `scaffold-model-bringup.sh` + `evidence_init` | bundle + run-id created |
| 1 deploy | bundle `03a/03b/03c` + `verify-experiment-labels.sh` | pod Ready + label PASS |
| 2 profile | `inference-decode-step-budget`, `analyze-zymtrace-workload`, `inference-kernel-profile`, `inference-kernel-ncu-profile`, `inference-dcgm-correlate`, `inference-graph-diff` | a real profile captured |
| 2.5 synthesize | [`inference-perf-synthesize`](/plugins/profile-and-optimize/skills/inference-perf-synthesize/SKILL.md) | ranked, artifact-cited recommendation ledger |
| 3 tune (cross-engine) | [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md) + `run-variant-ab.sh` (vLLM + SGLang arms) -> `import_variant_ab` | per-engine top-3 beat their baselines. Same-node >=3-trial |
| 4 quantize | [`inference-quantize-calibrate`](/plugins/profile-and-optimize/skills/inference-quantize-calibrate/SKILL.md) | quantize Job complete (ack-gated) |
| 5 validate quant | [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md) + perf A/B | accuracy within tolerance + perf win |
| 6 spec-decode train | [`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md) | non-degenerate training |
| 7 validate draft | bundle `spec-decode/dflash_vllm_eval.py` + spec_decode metrics | acceptance win vs standing |
| 8 bench (multi-workload) | `bench-all-workloads.sh` (AA/Sonnet/ShareGPT/random/code) -> `dcgm_correlate` + `import_roofline_sweep` | baseline + top-X benched across the full suite. Rooflines captured |
| 8.4 champion select | `perftunereport champion_select` -> `report_render` -> `publish_to_lake` | baseline vs top-X champion picked + tiered (VERDICT needs multi-workload + accuracy + L3) + published |
| 8.5 synthesize + ratchet | [`inference-perf-synthesize`](/plugins/profile-and-optimize/skills/inference-perf-synthesize/SKILL.md) + `value-findings.yaml` `next_lever` | final verdict ledger + future-work levers published. Champion = next baseline. Model's `next_lever` recorded (ratcheted, not done) |

## Explainable verdicts (every claim -> artifact)

Per `docs/METHODOLOGY.md`, all attribution claims must be matched with collected
profile data: every where/why verdict and every recommendation this orchestrator
surfaces carries a citation to its backing artifact (the `.nsys-rep` / `.ncu-rep`
/ zymtrace TSV / `dcgm_correlation.json` / eval json), the `sol_rigor` tier of
that evidence, and a DRAFT/VERDICT tier. A claim with no linked artifact is not
emitted. The ledger lives in the evidence bundle (`findings/`, via
`findings_record`) and is published into the `campaign=<run-id>` rows, so any
verdict in the perf-lake is traceable back to the exact capture that backs it.
[`inference-perf-synthesize`](/plugins/profile-and-optimize/skills/inference-perf-synthesize/SKILL.md) is the
enforcement point.

## Verdict rigor (DRAFT vs VERDICT)

Per `docs/METHODOLOGY.md` "Verdict rigor: DRAFT vs VERDICT", every number this
orchestrator surfaces defaults to **DRAFT** and is labeled provisional. Promote a
phase result to **VERDICT** only when it is variance-controlled (same-node, >=3
trials, mean +/- std), metric-isolated (median TPOT/ITL for decode-latency
claims -- NOT output tok/s at small `num_prompts`), compared to a
production-representative baseline, and (for which-kernel claims) backed by
nsys/ncu per-kernel data. The Phase-8 campaign is published with the honest
`verdict_tier`. An unsupported `verdict` auto-downgrades to `draft` and still
lands (the honest tier is recorded). Supersede a DRAFT everywhere it propagated
once a VERDICT overturns it.

## Safety

- **Scheduling is cluster-profile aware** -- on a **plain-K8s GB300**
  cluster (the default): `schedulerName: default-scheduler` +
  `resources.requests.nvidia.com/gpu` (k8s device-plugin authoritative). Preflight
  k8s GPU requests==0 on a dev node. On a **legacy Slurm-on-K8s** cluster (detect
  a `slurm-controller` + `slurm.example.com/lock` taint): `schedulerName:
  <slurm-scheduler-pod>` + the lock toleration, `sinfo`/`squeue` preflight,
  and NEVER `default-scheduler` + a hard `nodeSelector` to grab GPUs (it
  double-books GPUs Slurm has already allocated).
- **GPU-telemetry gate before any bench (hard precondition)** --
  `capture-run-env.sh` must return `gpu_frames_gate=pass` (the zymtrace implant is
  intercepting the pod) before a measurement bench is trusted. The gated wrappers
  (`bench-with-sol.sh` / `run-controlled-ab.sh` / `run-variant-ab.sh`) fail closed
  on it. Split the injection tax: headline perf injection-OFF, L1 SoL in a
  separate injection-ON window.
- **GB300 profiling default = the NGC CUDA-13 templates** (`nsys-ngc.yaml` /
  `nsys-sglang.yaml` / `ncu-ngc.yaml`). The in-image/apt nsys/ncu records 0
  kernels under the CUPTI image-vs-driver skew (Gate 0).
- **Strict publish only** -- land results via the gated path
  (`champion_select` -> `report_render` -> `publish_to_lake`, strict-by-default),
  never an ad-hoc driver. A VERDICT publish requires a clean pinned source commit
  (provenance gate).
- **Experiment isolation is mandatory** -- experiment-prefixed names +
  `experiment=<id-slug>` labels on every object. Never touch
  standing/platform/migration names (standing deployments, shared caches,
  image-pull secrets, ...). Teardown by label, PV last, per
  `docs/METHODOLOGY.md`.
- **Ack-gated phases** -- Phase 4 (quantize) and Phase 6 (train) submit GPU
  jobs. They fail closed without the explicit ack flag, per
  [`server/docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).
- **No silent fallbacks** -- a failed/empty profile is a capture bug to debug,
  not a verdict (the nsys capture-validation gate) -- EXCEPT the GB300
  CUDA image-vs-driver CUPTI skew (12.x toolkit vs 13.x driver ->
  `CUPTI_ERROR_INVALID_DEVICE` -> 0 kernels for all CUPTI clients. Needs a
  CUDA-13 image or zymtrace, not a re-capture). A red gate pauses the
  pipeline. It is never skipped. (zymtrace empty-now right after a bench is
  usually ClickHouse INGEST LAG, not a gap -- `capture-sol-window.sh` polls +
  requeries for the freshest data. See
  [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md).)
- **No external posting** -- this skill never pushes to prod (`infr-cli`) or
  posts outside the workspace without explicit per-turn operator approval.

## Source-of-truth references

- **The model-bringup template** -- the SHARED scaffold the scaffolder
  renders for every model: `variants/` (the cross-engine
  vLLM+SGLang A/B: `run-variant-ab.sh` + `variant-deploy-{template,sglang}.yaml`
  + `rank-variants.py` + `compare-engines.py`), the GB300 NGC CUDA-13 nsys/ncu
  templates (incl. `nsys-sglang.yaml`), and
  `profiling/` (the engine-agnostic `bench-all-workloads.sh` +
  `run-controlled-ab.sh` + `experiment-deploy-template.yaml`). This is the
  source-of-truth for the rendered bundle's scaffolding (NOT a specific operator
  deploy bundle).
- The template's cross-engine docs -- the cross-engine A/B mechanism +
  per-model campaign runbook.
- The template's deep-profiling runbook -- the controlled-A/B +
  submit-and-queue gated-capture mechanics the GPU phases reuse.
- [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) -- the canonical
  results -> rooflines -> champion_select -> perf-lake pipeline Phases 8/8.4 drive.
- `docs/METHODOLOGY.md` -- experiment isolation, scheduling profiles,
  benchmark methodology hygiene, multi-workload coverage, DRAFT-vs-VERDICT,
  speed-of-light framing.

## Full-context reporting (no bare numbers)

Per the canon "Every performance number carries its full context (no bare numbers)"
(`docs/METHODOLOGY.md` "Full-context reporting"): every number this
skill emits MUST carry its full measurement-context descriptor, and every comparison MUST be
matched on it. A bare `tok/s` / TPOT / BW / %SoL / speedup is a defect - it cannot set a
default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named.
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

If this skill produces a measurement (tok/s, latency, %SoL, speedup), follow the
rigor discipline: capture L1 zymtrace + L3 DCGM (L4 ncu where feasible)
Speed-of-Light and publish `--strict`. Canonical map: `docs/METHODOLOGY.md`. Skills that
do not produce measurements are exempt (`docs/METHODOLOGY.md` "Speed-of-light framing").

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per `docs/METHODOLOGY.md` "Always be grinding". A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.

## Champion close: capture the known-good config (CONFIG half)

A model bring-up is ratcheted, not done -- and a champion is not closed until its **known-good
config** is captured (rigor principle k, `docs/METHODOLOGY.md` "the performance ratchet ->
CONFIG half"). At Phase 8.4 (`champion_select`), also **`known_good_config record`** the model's
REQUIRED serve flags (boot-blocker / crash-at-high-c / deploy-correctness workarounds it needs)
into `configs/known-good-configs.yaml`
via the [`inference-known-good-config`](/plugins/profile-and-optimize/skills/inference-known-good-config/SKILL.md) skill. The
champion-close gate (`verify-grind-closure.sh <bundle> --model <m>`)
fails closed unless BOTH the next lever (grind-ledger) AND a registered known-good config are present,
the `~/.cursor/hooks/known-good-config-gate.sh` hook then ASKS before any deploy that drops a required flag.
