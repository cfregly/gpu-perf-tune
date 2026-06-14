---
name: inference-model-eval
last_validated: 2026-05-23
description: >-
  Drive lm-eval-harness quality evals (GPQA, MMLU-Pro) inside model pods
  plus optional ExternalEval (externally operated). Use to validate model
  quality before promoting to staging or prod, after vLLM / quantization /
  KV-cache changes, or to compare against published HuggingFace / paper
  baselines. Pair with inference-perf-bench (the perf-side counterpart) for
  full pre-promotion validation. Triggers on "lm-eval-harness", "GPQA",
  "MMLU-Pro", "ExternalEval", "model quality eval", "/run-model-eval",
  "run evals", "run gpqa", "run mmlu", "run external-eval", "run evals on the
  model", or any combination of "eval / quality / accuracy" with
  "inference / model / vllm".
allowed-tools:
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Bash(kubectl:*)
  - Bash(curl:*)
  - Bash(jq:*)
  - Bash(date:*)
  - Read
  - Write
---

# inference-model-eval

## Purpose

Run quality and accuracy evaluations against a live inference
endpoint to validate that a model deployment is fit for promotion.
Three benchmark families:

- **GPQA** - graduate-level Q&A; runs inside the model pod via
  [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- **MMLU-Pro** - broad multi-task understanding. Runs inside the model
  pod via lm-eval-harness.
- **ExternalEval** - externally operated. Cockpit surfaces contact info
  and endpoint details, operator coordinates with the ExternalEval
  operator out-of-band.

The eval workflow itself (task selection, in-pod lm-eval-harness
invocation, monitoring, results download, ExternalEval handoff) is
summarized under "Workflow" below. This skill's main job is the
cockpit-specific glue - the evidence bundle and the perf-baseline
tie-in.

## When to use

- Before promoting a new model to staging or prod - validate quality
  alongside the perf check
  ([`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)).
- After vLLM version bumps, quantization changes (NVFP4 vs FP8 vs
  BF16), KV-cache-dtype changes - ensure no quality regression.
- Regression check against published baselines (HuggingFace model
  card, paper numbers).
- Pairing perf-vs-quality A/B for proposed config changes.

Do **not** use this skill for:

- Inference performance measurement - that is
  [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md).
- Terminal-Bench 2.0 / SWE-Bench Verified at scale - those are larger
  evaluation-harness runs driven by a dedicated eval pipeline. This
  skill covers the in-pod lm-eval-harness path only.

## Example prompts

- "Run GPQA + MMLU-Pro on the kimi-k25 dev pods."
- "Run model-eval on the new minimax-m2.7 deployment, batch size 64."
- "Quality regression check on glm-5-fp8 after the vllm 0.20 bump."
- `/run-model-eval --model kimi-k25 --tasks gpqa,mmlu_pro`
- `/inference-model-eval --pods c2-kimi-k25-fp4-* --tasks gpqa`

## Prerequisites

1. **`kubectl` context for a dev cluster**.
2. **Namespace** containing the target pods.
3. **HF_TOKEN** if the model card / dataset requires it.
4. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for the result bundle.

## Interaction style

Iterative. The workflow pauses naturally at task selection, where the
operator chooses which evals to run.

## Workflow

### Phase A: scaffold an evidence bundle (cockpit-side)

```text
/evidence-bundle-init --family inference-model-eval \
  --intent "model-eval on <model> tasks=<gpqa,mmlu_pro,external-eval>"
```

### Phase B: run the evals

Select the tasks (GPQA / MMLU-Pro), invoke lm-eval-harness inside the
target model pod against the served endpoint, monitor the run, and
download the results into the evidence bundle. For ExternalEval, hand off
to the ExternalEval operator out-of-band and record the returned scores.

### Phase C: tie evals to a perf-baseline registry entry (cockpit-side)

When a model passes both `inference-perf-bench` and
`inference-model-eval`, register the perf baseline with a `notes`
field that names the eval scores:

```text
/inference-perf-baseline-bridge record \
  --model <model> \
  --source experiments/artifacts/inference-perf-bench/<run-id>/ \
  --notes "GPQA=<score>; MMLU-Pro=<score>; ExternalEval=<score>"
```

This lets a future
[`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md)
diff confirm that a perf regression isn't masked by a quality gain
(or vice versa).

## Safety

- **Read-only on the cluster.** lm-eval-harness runs inside the
  existing model pod. The workflow does not create or delete pods.
- **No customer-data leakage.** GPQA / MMLU-Pro datasets are public.
  Any in-pod intermediate artifacts should be cleared before the
  bundle is shared externally.
- **ExternalEval is operator-mediated.** The cockpit only displays
  contact info. Do not auto-DM the ExternalEval operator from any agent surface.

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

Quality-eval scores (lm-eval-harness MMLU/GSM8K/etc.) are not directly
roofline-bound, so this skill does NOT add a `%SoL` column to its
output. Per `docs/METHODOLOGY.md` "Speed-of-light framing", the
methodology applies to *measurement-producing perf skills* - eval
accuracy is orthogonal. When eval pairs with perf
([`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)) for a
quant-quality-vs-throughput comparison, the perf side carries `%SoL`
and the eval side carries accuracy %.

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

## Source-of-truth references

- Pair: [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) - the
  perf counterpart for full pre-promotion validation.
- [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md)
 - ties eval scores to a perf-baseline registry entry (Phase C).
- `docs/METHODOLOGY.md` - full-context reporting + verdict rigor.
