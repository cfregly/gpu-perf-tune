---
name: inference-spec-decode-tune
last_validated: 2026-06-04
description: >-
  Tune a speculative-decoding DRAFT head's TRAINING hyperparameters (global
  batch, learning rate, accumulation, warmup) for a target LLM, optimizing the
  TRUE serving in-engine acceptance length (vLLM spec_decode counters) with a
  cheap training-acc proxy for triage. Reuses the search ALGORITHMS
  (hyperband/grid/random natively. Bayesian-TPE via optuna if installed) wired
  to a GB300/managed K8s pod launcher + an offline EAGLE3/DFlash trainer (distinct from
  the MLPerf ai_tuning contract, which does not map to draft training). The
  draft-training analog of inference-tune-sweep (which tunes vLLM SERVING
  config). Triggers on "tune the draft head", "tune eagle3 / dflash training",
  "bayesian/hyperband tune the draft", "search global batch and LR for the
  speculator", "spec-decode hyperparameter sweep", "draft-training tuner", or
  any combination of "tune / sweep / optimize / search / bayesian / hyperband"
  with "eagle3 / dflash / draft / speculator / spec-decode".
allowed-tools:
  - mcp__profile_and_optimize__ai_tuning_optimizer
  - mcp__profile_and_optimize__ai_tuning_space
  - mcp__profile_and_optimize__perf_baseline_record
  - mcp__profile_and_optimize__perf_baseline_diff
  - Bash(kubectl:*)
  - Bash(python3:*)
  - Bash(bash:*)
  - Read
  - Write
---

# inference-spec-decode-tune

## Purpose

Find the draft head's best **training** hyperparameters by searching
{global_batch, learning_rate[, accum, warmup]} instead of hand-picking one
config, optimizing the **measured serving acceptance length** (the metric that
actually determines speculative-decode speedup) with a cheap training-acc proxy
for early triage.

It is the draft-**training** counterpart to
[`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md) (which tunes vLLM
**serving** config) and the tuning loop around
[`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md) (which
trains a single head). It is the realization of the `--optimizer bayesian|hyperband`
hook that `inference-tune-sweep` documents as a stub.

## Why a separate tuner (not the MLPerf ai_tuning MCP)

The `profile_and_optimize` `ai_tuning_*` family ships the engines we want
(`random` / `bayesian` tpe,gp / `multifidelity` hyperband,bohb) but is coupled
to MLPerf **training**: `config_patches` validated against an extracted shell
config (`config_DGXB200_*.sh`), a `~/.hypertune` ledger, and `submit` via a
Slurm launcher. Draft training has no shell-config template and runs as a
GB300/managed K8s **pod**, so this skill reuses the search *algorithms* (natively for
hyperband/grid/random. Optuna for TPE) wired to the draft launcher, and keeps
the `ai_tuning_*` MCP as the cousin for the training side.

## When to use

- A draft head trains stably (via `inference-spec-decode-train`) and you want
  its acceptance-optimal training config, not just one hand-picked point.
- A coarse batch-size / LR sweep (e.g. global-batch 8/16/32/128 arms) showed the
  metric is still moving and you want an adaptive search to find the knee.

Do **not** use for: vLLM serving config (use `inference-tune-sweep`). The first
stable-training pass on a brand-new draft (use `inference-spec-decode-train`).

## Inputs / artifacts (EAGLE3 reference layout)

The reference implementation (a GLM-5.1 EAGLE3 tuner) lays these artifacts down
in the deploy bundle (e.g. under `deploy/gb300/`):

- `tuning-space.eagle3-draft.json` - the search space + objective (serving
  acceptance) + proxy (training acc) + hyperband rungs (in samples-consumed).
- `tune-trial.sh` - one trial: maps `global_batch -> (BS, accum)` on the 4-GPU
  node (BS<=4. BS=8 OOM'd), launches a training arm (`e3bs-arm.yaml`), reads
  proxy-acc at a matched-sample cap (`read-acc0.sh`), `promote`s to serving eval.
- `tune-driver.py` - the loop: `--strategy hyperband` (default) | `grid` |
  `random` | `tpe` (optuna), `--seed-from` prior arms, durable `tune-ledger.json`,
  `--dry-run` (zero cluster spend).
- serving objective: `eagle3_vllm_eval.py` +
  `deploy/gb300/eagle3-accept-eval.yaml` (vLLM `method=eagle3`,
  `vllm:spec_decode_num_accepted_tokens / num_drafts` -> mean accept length).

## Workflow

### Phase 0 - state objective + space + strategy

- Objective of record = **measured serving acceptance length** (NOT the proxy).
- Proxy = training `acc=` at matched samples (drives the cheap hyperband rungs).
- Strategy (all selectable via `--strategy`. Pick by search-space size):
  - `hyperband` (default) - successive-halving bandit. Best GPU-efficiency (early-kills weak arms). Use it unless you have a reason not to.
  - `grid` - exhaustive over the categorical grid, `random` - uniform-draw baseline. Both dependency-free.
  - `tpe` - Bayesian, model-based search. Sample-efficient on LARGER / continuous spaces. **Opt-in**: needs `pip install optuna` (the other three are dependency-free. Tpe exits with a clear message if optuna is absent).

### Phase 1 - dry-run the plan (no spend)

```text
tune-driver.py --strategy hyperband --seed-from <prior-arms.json> --dry-run
```

Prints the bracket, the per-config `(BS, accum, lr)` mapping, and the rung
read-caps. Confirm the bracket before any GPU spend.

### Phase 2 - run the proxy search (cheap rungs)

```text
tune-driver.py --strategy hyperband --seed-from <prior-arms.json>
```

Launches the bracket as experiment-isolated training pods, reads proxy-acc at
each rung from the SAME run's log, early-kills the bottom `(1 - 1/eta)` to save
GPU, lets survivors train on. Every rung read is appended to `tune-ledger.json`
(durable across churn).

### Phase 3 - promote survivors to measured serving acceptance (the objective)

For the top-K survivors, run `eagle3_vllm_eval.py` against the GB300
serving deploy and read `vllm:spec_decode_*` -> mean acceptance length. The
**champion is decided here**, on measured serving acceptance - the proxy only
ranked candidates for promotion.

### Phase 4 - gate + record

Gate the champion vs the standing config with `perf_baseline_diff`. On a real
win, record the new champion baseline + publish to the perf-lake (serving
acceptance/throughput IS perf-lake-eligible). On a within-noise tie, report "no
improvement" - never round a proxy win up to a serving win.

## Verdict rigor (DRAFT vs VERDICT)

- proxy-acc rung results are **DRAFT** (a surrogate, single-trial).
- The champion is a **VERDICT** only on **measured serving acceptance**, with a
  same-config repeat for noise, against the production-representative baseline.

## Full-context reporting (no bare numbers)

Per the methodology canon "Every performance number carries its full context (no bare
numbers)" (`docs/METHODOLOGY.md`, "Full-context reporting"): every number this
skill emits (throughput, latency, TPOT/ITL, BW, %SoL, speedup, efficiency, goodput, acceptance
rate, scaling efficiency, thermal/failure rate - whatever it reports) MUST carry its full
measurement-context descriptor, and every comparison MUST be matched on it. A bare number is a
defect - it cannot set a default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named. (If the metric is not roofline-bound - e.g. accuracy/acceptance - omit `%SoL` but keep the rest of the descriptor.)
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

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

## Safety

- Experiment isolation: every trial pod is experiment-prefixed (e.g.
  `<slug>-e3bs-<run-id>`) + `experiment=<run-id>` label. Teardown by label. Never
  reuse standing names. Never touch a parallel session's pods (`workstream=`
  label check first).
- Durability: stream each trial's head + log to object storage before pod delete
  (emptyDir is ephemeral).
- Ack-gated: any submit/serving-deploy step fails closed without its ack. The
  `--dry-run` is the safe preview.
- Cost honesty: serving eval is expensive (train -> convert -> deploy -> eval),
  so hyperband triages on the proxy and only promoted survivors pay for serving.

## Source-of-truth references

- [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md) - the serving-config
  sibling + the `--optimizer` hook this skill realizes.
- [`inference-spec-decode-train`](/plugins/profile-and-optimize/skills/inference-spec-decode-train/SKILL.md) - the
  single-head trainer this loops around.
- The `ai_tuning_*` MCP family - the MLPerf-training cousin (same engines,
  different contract).

## Contact

Open an issue in this repository.
