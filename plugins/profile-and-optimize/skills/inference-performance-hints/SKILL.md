---
name: inference-performance-hints
license: Apache-2.0
compatibility: Requires a skills-compatible agent and configured MCP servers named in allowed-tools.
metadata:
  last-validated: "2026-08-16"
description: >-
  Applies Jeff Dean and Sanjay Ghemawat's Performance Hints to GPU inference.
  Builds a back-of-the-envelope cost ledger, identifies the hot-path class,
  bounds each optimization with profile share, and routes the highest-value
  hypothesis to the repo's benchmark, profiling, tuning, quantization, or
  capacity skill. Use for "performance hints", "back-of-the-envelope perf",
  "where should I optimize", "flat profile", "hot path review", "estimate
  before benchmarking", or "make this inference path faster". Estimates remain
  DRAFT until production-shaped measurement confirms them.
allowed-tools: "mcp__profile_and_optimize__search_runbooks mcp__profile_and_optimize__search_evidence Read"
---

# inference-performance-hints

## Purpose

Turn a vague performance question into one bounded hypothesis and one useful
experiment before spending GPU time. The workflow adapts Jeff Dean and Sanjay
Ghemawat's general performance guidance to GPU inference. It does not replace the
measurement and profiling skills that produce decision-grade evidence.

## When to use

- An operator asks where to begin optimizing a model, runtime, kernel, or data path.
- Several possible levers exist and a rough estimate can eliminate weak options.
- A CPU or GPU profile is flat and no single hotspot dominates.
- A design review needs performance reasoning before implementation.
- A benchmark plan needs an expected bottleneck and a stopping rule.

Do not use this skill to:

- claim a speedup without a matched measurement,
- capture a live profile, run a sweep, or mutate a deployment,
- replace vendor ceilings with the old "numbers everyone should know" table,
- optimize a numeric-correctness failure. Use `inference-kernel-whitebox-debug`.

## Example prompts

- "Apply Jeff Dean's performance hints to this decode path."
- "Do a back-of-the-envelope estimate before we reserve the GPUs."
- "The profile is flat. Where should we optimize next?"
- "Review this inference API for hot-path costs."
- `/inference-performance-hints --focus decode-latency --model <model>`

## Prerequisites

1. A named workload or design with enough shape information to estimate work.
2. A target metric such as TTFT, TPOT/ITL, tokens/s/GPU, cost/M tokens, loader
   time, or memory footprint.
3. For any claimed regression or win, a full-context baseline. If it is missing,
   the workflow stops at a DRAFT estimate and routes to measurement.

## Interaction style

Work one gate at a time. Show the estimate ledger and its assumptions before
recommending an experiment. Ask the operator to resolve only an assumption that
would change the top-ranked experiment. Never turn an estimate into a verdict.

## Workflow

### Phase 0: load the canonical hints

```text
mcp__profile_and_optimize__search_runbooks with:
  query: "back-of-the-envelope GPU inference"
  limit: 10
```

Read
[`performance-hints.md`](../../server/docs/performance-hints.md)
and use its current formulas and routing rules. The upstream guide is attribution
and general technique. This repo document is the GPU-inference contract.

### Phase 1: classify the path

Record:

- execution frequency: setup, per model load, per request, per prefill token, or
  per decode token,
- workload identity: model, active parameters, precision, TP/PP/EP/DP, engine,
  ISL/OSL distribution, concurrency, warm/cold regime,
- target metric and current value, if measured,
- likely resource: compute, HBM, interconnect, launch/host, storage/network, or
  queueing,
- evidence source and confidence.

Missing workload shape means `NEEDS_CONTEXT`. Missing measurement permits a DRAFT
estimate but blocks any claim of improvement.

### Phase 2: write the rough cost ledger

Estimate only relevant terms: weight and KV bytes, FLOPs, collective bytes,
launch count, host gaps, model-loading bytes, and queueing. Use achieved rates
from local evidence when possible. Otherwise cite the exact ceiling key and an
explicit efficiency assumption.

For overlapping work, use the maximum as the lower bound. Add serialized work.
Round uncertain inputs and show units. The checked-in synthetic example at
[`examples/performance-hints/`](../../../../examples/performance-hints) demonstrates the
JSON shape.

### Phase 3: compare the estimate with measurement

If a full-context baseline exists, compare it with the lower bound. A large gap
is not itself a diagnosis. Use it to choose the next profile:

| Suspected term | Route |
| --- | --- |
| End-to-end serving curve or missing baseline | `inference-perf-bench` |
| CPU/GPU attribution with low mutation cost | `analyze-zymtrace-workload` |
| c=1 decode host, kernel, and communication budget | `inference-decode-step-budget` |
| Absolute CUDA timeline or launch gaps | `inference-kernel-profile` |
| Per-kernel stalls, occupancy, bytes, and roofline | `inference-kernel-ncu-profile` |
| Config-space hypothesis | `inference-tune-sweep` |
| Precision or KV representation | `inference-quantize-calibrate` plus `inference-model-eval` |
| Fleet count or SLA feasibility | `inference-capacity-sizing` |

### Phase 4: apply the hint families

Check in this order:

1. Reduce asymptotic or active work: padding, active parameters, repeated decode
   steps, duplicated parsing, and unnecessary transfers.
2. Bulk and fuse: batch API crossings, tokenization, launches, and operator chains.
3. Improve representation: quantization, KV dtype, contiguous layout, compact hot
   metadata, and fewer indirections.
4. Remove allocations, copies, repeated work, logging, and stats from hot paths.
5. Reuse precomputed state, compiled graphs, stable prefixes, buffers, and
   descriptors, with cold and warm results separated.
6. Shorten synchronization, overlap pipeline stages, and parallelize at a grain
   large enough to amortize dispatch.
7. Inspect generated code or hand-specialize only when the profile proves that
   local code generation is the remaining limit.

For a flat profile, look higher in the call graph for a loop or representation
change. If none exists, rank several independently measurable small improvements.
Do not manufacture one dominant hotspot.

### Phase 5: rank and hand off

Bound each candidate by its measured profile share. Record the mechanism,
maximum plausible end-to-end impact, confidence, experiment cost, rollback, and
stop condition. Select one experiment, hold all unrelated dimensions fixed, and
route it to the sibling skill that can produce evidence.

Report and ask before any mutating or cluster-spending action. This skill itself
is read-only.

## Output template

```markdown
# Performance hint pass: <workload>

Status: DRAFT | VERDICT | NEEDS_CONTEXT
Target: <metric and goal>

## Estimate ledger
| Term | Work | Unit cost | Estimated time | Source | Confidence |

Estimated lower bound: <value and overlap/serialization rule>
Measured baseline: <value with full context, or missing>

## Ranked candidates
| Rank | Mechanism | Profile share | Max impact | Confidence | Experiment cost |

NEXT EXPERIMENT: <one controlled A/B>
PROVES: <metric and artifact>
STOP IF: <refutation condition>
ROUTE: <sibling skill>
```

## Verdict rigor

All rough costs, extrapolations, and single observations are DRAFT. A result can
become a VERDICT only through the variance, baseline, capture, and full-context
rules in [`docs/METHODOLOGY.md`](../../../../docs/METHODOLOGY.md). The estimate ledger may
prioritize a measurement. It cannot promote a claim.

## Safety

- Read-only. Do not launch jobs, edit deployments, or write production configs.
- Do not present the upstream latency table as current GPU or fabric data.
- Do not sum overlapping lower bounds or compare mismatched workload shapes.
- Do not recommend caching without separate cold and warm measurements.
- Do not hide a missing baseline, profiler gap, or quality gate.

## Source-of-truth references

- [`performance-hints.md`](../../server/docs/performance-hints.md)
- [`docs/METHODOLOGY.md`](../../../../docs/METHODOLOGY.md)
- [`configs/sol-ceilings.yaml`](../../../../configs/sol-ceilings.yaml)
- [`examples/performance-hints/README.md`](../../../../examples/performance-hints/README.md)
