---
name: inference-spec-decode-service
last_validated: 2026-06-01
description: >-
  Closed-loop speculative-decoding-as-a-service orchestrator -- a self-hosted analog
  of Fireworks FireOptimizer's adaptive speculative execution. Composes the per-phase
  skills into one profile-matched loop: profile live traffic (inference-workload-profile)
  -> hit-rate-matched corpus -> train a draft (inference-spec-decode-train) -> in-engine
  acceptance + same-node TPOT A/B vs the standing config -> promote ONLY on a measured win
  (standing config stays otherwise) -> publish. It is Phase 6 of inference-model-optimize
  run as a profile-matched closed loop instead of one-shot. Defines a service-state.json
  contract + a --mode oneshot|controller flag so it can later graduate to a deployed
  in-cluster controller without a rewrite. Triggers on "spec-dec as a service",
  "adaptive speculative decoding service", "fireoptimizer equivalent", "auto-train a draft",
  "close the spec-decode loop", "stand up the spec-dec service", or any combination of
  "service / loop / adaptive" with "spec-decode / speculative / draft".
allowed-tools:
  - mcp__profile_and_optimize__evidence_init
  - mcp__profile_and_optimize__perf_tune_report_campaign_init
  - mcp__profile_and_optimize__perf_tune_report_cell_run
  - mcp__profile_and_optimize__perf_tune_report_atlas_aggregate
  - mcp__profile_and_optimize__perf_tune_report_dcgm_correlate
  - mcp__profile_and_optimize__perf_tune_report_report_render
  - mcp__profile_and_optimize__perf_tune_report_publish_to_lake
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__search_runbooks
  - Bash(python3:*)
  - Bash(kubectl:*)
  - Bash(sinfo:*)
  - Bash(squeue:*)
  - Bash(sbatch:*)
  - Read
  - Write
---

# inference-spec-decode-service

## Purpose

One orchestrator for "give a served model a draft tailored to its OWN traffic, and
keep the standing config unless the tailored draft measurably wins." It does **not**
re-implement any phase -- it sequences the mature per-phase skills behind a single,
gate-driven, profile-matched loop and one evidence bundle, so an operator drives the
whole `profile -> corpus -> train -> A/B -> promote -> publish` loop without
hand-stitching skills together.

It is a self-hosted analog of Fireworks FireOptimizer's "adaptive speculative
execution": the differentiator is operational, not algorithmic -- the draft is trained
on a corpus **matched to the measured workload profile** (the documented source of the
higher draft hit-rate), and it is promoted only on a measured acceptance + TPOT win.

This is the closed-loop framing of
[`inference-model-optimize`](/plugins/profile-and-optimize/skills/inference-model-optimize/SKILL.md) Phase 6/7: that
orchestrator trains a draft on a generic corpus one-shot. This one matches the corpus to
traffic and is structured to run repeatedly.

**Implementation (v1 `oneshot`).** The loop is driven by `specdec-loop.sh`, which
EXECUTES C3-C5 behind `--i-understand-this-uses-the-cluster` (DRY by default), with
companions `specdec-decide.py` (the variance-controlled GO/NO-GO gate),
`canary-arm.yaml` (the parameterized experiment-prefixed A/B arm), and `specdec-presets.sh`. The
Workflow phases below map 1:1 to its `C1..C5 + publish`. `--mode controller` (drift-triggered
re-adaptation) remains the v2 seam - see "Mode: controller" below.

## When to use

- A served model has a latency ceiling you want to beat with a draft tailored to its
  actual traffic (not a generic UltraChat+ShareGPT draft).
- You want the "spec-dec as a service" capability: profile -> matched draft ->
  measured promote, as one workflow.
- You are standing up the loop now and want it architected to later become a deployed
  controller (continuous re-adaptation to traffic drift).

Do **not** use this skill for:

- A one-shot draft on a generic corpus with no workload profiling -- use
  [`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md) directly.
- The full new-model bring-up (deploy/profile/tune/quantize/...) -- that is
  [`inference-model-optimize`](/plugins/profile-and-optimize/skills/inference-model-optimize/SKILL.md). This skill is its
  spec-dec sub-loop, made adaptive + repeatable.
- `deepseek_mtp` / `ngram` / Predicted-Outputs methods -- there is nothing to train
  (built-in head / prompt-driven / per-request reference text). Set the config and run
  only the acceptance A/B (Phase 4 below).

## Example prompts

- "Stand up the spec-dec service for GLM-5.1 matched to our AA-shape traffic."
- "Auto-train a draft for this model's workload and promote it only if it wins."
- "Close the speculative-decoding loop: profile, match, train, A/B, publish."
- `/inference-spec-decode-service --target /models/base --served <deploy> --profile access.jsonl --mode oneshot`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. A representative traffic sample (OpenAI-style access JSONL) OR a named bench-shape
   set (e.g. the `inference-aa-workload` shapes) to use as the profile.
2. The **target** base checkpoint (BF16/FP8 -- NOT the NVFP4 serving copy) staged on a
   shared FS/PVC the Slurm trainer can read (the `inference-spec-decode-train` prereq).
3. The standing served deploy + its current `--speculative-config` (the A/B baseline).
4. Serving-A/B cluster access. The canary runs on EITHER plain K8s
   (`schedulerName: default-scheduler` + `nvidia.com/gpu`) OR Slurm-on-K8s (the Slurm
   `schedulerName` + the `slurm.example.com/lock` toleration). The loop
   AUTO-DETECTS which (Slurm-on-K8s = a Slurm control plane + the lock taint) and `--scheduler
   default|slurm` overrides. For a TRAINED method (`eagle3`/`dflash`), also `sbatch` access to the
   SpecForge Slurm trainer (C3 is Slurm-native in BOTH cases). An idle GPU node confirmed BEFORE
   any GPU phase (Slurm-on-K8s: `pin-node.sh`-verified). Training never preempts serving. (The no-train
   `deepseek_mtp`/`ngram` path needs only the serving cluster.)
5. For a `--mode controller` run only: a writable `service-state.json` location.

## Interaction style

Iterative + gate-driven, modeled on
[`inference-model-optimize`](/plugins/profile-and-optimize/skills/inference-model-optimize/SKILL.md): run one phase, report
the gate verdict + the headline number (DRAFT unless promoted), then ask before
advancing. Never auto-advance past a red gate. The operator may start at any phase
(`--start-phase`) or skip the train phases for a non-trained method.

## Workflow

### Phase 0: resolve intent + scaffold + service-state

State back the parameters: `target-model-path`, `served-deploy`, `standing-config`,
`profile-source` (traffic JSONL | AA shapes | provided log), `spec-method`
(`eagle3` | `dflash` | `deepseek_mtp` | `ngram` | `predicted_outputs`), `mode`
(`oneshot` | `controller`). Scaffold one evidence bundle (run-id == experiment-id ==
campaign) and initialize `service-state.json`:

```json
{"schema":"specdec_service_v1","experiment_id":"<run-id>","target":"<path>",
 "served_deploy":"<name>","standing_config":"<json>","champion_draft":null,
 "last_profile_sha256":null,"last_ab_verdict":null,"history":[]}
```

`service-state.json` is the **graduation seam to a deployed controller**: it records the
last profile hash, current champion draft, and last A/B verdict so a future controller
Deployment can decide "has the workload drifted enough to retrain?" without re-deriving
state. Report the scaffolded paths. Ask before any GPU phase.

### Phase 1: profile the workload (WS-C1)

Hand off to [`inference-workload-profile`](/plugins/profile-and-optimize/skills/inference-workload-profile/SKILL.md):
emit `workload-profile.json` (length distributions, content-class mix, ISL/OSL bench
shapes, `recommended_spec_decode.method`). For an AA-shape profile source, use the
AA-shape adapter (`workload-profile.py --aa-shapes ...`) so the three AA shapes
(1k/10k/100k) become the profile. Record `last_profile_sha256` in service-state.
Gate: a profile with a method recommendation. If the recommendation is
`predicted_outputs` (rewrite-heavy traffic), skip to Phase 4 with the per-request
prediction path -- there is no draft to train.

### Phase 2: build the hit-rate-matched corpus (WS-C2)

Run `profile-to-corpus.py` (in the `inference-workload-profile` skill dir): mode (b)
converts a redacted traffic JSONL directly to the SpecForge `conversations` schema,
mode (a) emits a weighted `prepare_data.py` plan matched to the content-class mix +
length distribution. The corpus is the `DATA_PATH` for the trainer. Gate: a corpus that
renders cleanly through the target chat template (a bad loss-mask shows up as
training loss stuck at zero).

### Phase 3: train the matched draft (WS-C3)

Hand off to [`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md)
with the Phase-2 corpus as `DATA_PATH`. **The capture-correctness probe
(`--probe-logits`) MUST pass before any full re-capture/retrain** -- a broken
hidden-state capture trains a head that serves at ~0% acceptance and wastes days
of GPU time. Submit-and-queue on idle Slurm nodes. Never babysit, never preempt serving.
Gate: non-degenerate training `acc_0` / loss.

### Phase 4: acceptance + TPOT A/B (WS-C4, the gate that matters)

Serve the trained draft via `--speculative-config` on an experiment-prefixed canary and
measure **in-engine** acceptance (`dflash_vllm_eval.py` + vLLM `spec_decode_*`
counters) on a **same-node** A/B vs the standing config, **under cudagraph**, **>=3
trials** (DRAFT->VERDICT). Gate: an acceptance-length win AND a median-TPOT win. A draft
that loses is **do-not-ship**. Record `last_ab_verdict`.

### Phase 5: promote on a win (WS-C5) + publish

On a measured win, flip the experiment-prefixed canary's `--speculative-config`
(`specdec-presets.sh` renders the line). Set `champion_draft` in service-state. On a
loss, keep the standing config and report why. Either way, publish the campaign:
`perf_tune_report_campaign_init --experiment-id <run-id> --focus latency` ->
`cell_run` (at the profile's `bench_shapes`) -> `atlas_aggregate` ->
`dcgm_correlate` (raises `sol_rigor` to L3) -> `report_render` -> `publish_to_lake`.
Record `campaign=<id>` + `s3://perf-lake/...` in SOURCE.md/summary.md.

### Mode: controller (graduation seam -- NOT v1)

`--mode controller` is a documented stub for the deployed-service evolution: a
long-running loop that periodically re-profiles live traffic, compares
`last_profile_sha256` to detect drift, retrains + re-A/Bs on drift, and auto-promotes on
a win -- plus the dynamic runtime fallback (disable spec-dec when the live
`spec_decode_*` acceptance drops). v1 implements only `oneshot`, `controller` reuses
Phases 1-5 verbatim driven by a controller Deployment reading/writing
`service-state.json`. Implementing the Deployment + fallback engine policy is the
separate "(b) deployed controller" track.

## Verdict rigor (DRAFT vs VERDICT)

Training `acc_0` is a training-time signal, not a serve verdict. The promote/no-promote
claim is the Phase-4 serve acceptance + TPOT A/B: DRAFT unless same-node + >=3 trials +
metric-isolated (median TPOT/ITL) + matching `cudagraph_mode` + a
production-representative baseline (the standing config, e.g. MTP K=3 -- never an eager
strawman). The Phase-5 campaign publishes with the honest `verdict_tier`.

## Safety

- **Scheduling (plain K8s OR Slurm-on-K8s)** -- the serving canary is rendered for whichever the loop
  detects (override with `--scheduler default|slurm`): plain K8s uses `default-scheduler` +
  `nvidia.com/gpu`. Slurm-on-K8s uses the Slurm `schedulerName` + the `slurm.example.com/lock`
  toleration, same-node-pinned to a `pin-node.sh`-verified-idle node (a bare `nodeSelector` to a
  non-idle node loops on NodeAffinity under the Slurm scheduler). TRAINED-method Slurm jobs
  (eagle3/dflash) submit-and-queue and never preempt serving. Preflight an idle node first.
- **Standing config stays until a measured win** -- the served `--speculative-config`
  is not changed until the Phase-4 A/B shows a real acceptance + TPOT win under cudagraph.
- **Capture-correctness gate** -- never launch a full re-capture/retrain until the
  `--probe-logits` probe passes. A broken capture costs days of wasted training.
- **Experiment isolation** -- canary + draft-staging objects experiment-prefixed +
  `experiment=<id-slug>`. Teardown by label, PV last. Never touch standing deploy names.
- **Local/fork only** -- no upstream PR / external posting without explicit per-turn
  operator approval.

## Source-of-truth references

- **v1 loop components:** `specdec-loop.sh` (the loop), `specdec-decide.py` (GO/NO-GO gate),
  `canary-arm.yaml` (A/B arm), `specdec-presets.sh` (config rendering).
- [`inference-workload-profile`](/plugins/profile-and-optimize/skills/inference-workload-profile/SKILL.md) -- Phase 1/2 (profile + corpus).
- [`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md) -- Phase 3 (SpecForge train). Its capture-correctness + serve-wiring gates are the ones this loop encodes.
- [`inference-model-optimize`](/plugins/profile-and-optimize/skills/inference-model-optimize/SKILL.md) -- the parent orchestrator whose Phase 6/7 this loop makes adaptive + repeatable.
- [`inference-aa-workload`](/plugins/profile-and-optimize/skills/inference-aa-workload/SKILL.md) -- the AA-shape profile source.

## Contact

Open an issue in this repository.

## Full-context reporting (no bare numbers)

Per the methodology canon "Every performance number carries its full context (no bare
numbers)" (`docs/METHODOLOGY.md`, "Full-context reporting"): every number this
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
per the Grind Mandate (`docs/METHODOLOGY.md`, "Always be grinding"). A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.
