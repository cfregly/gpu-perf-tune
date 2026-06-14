---
name: inference-perf-synthesize
last_validated: 2026-06-01
description: >-
  Fuse the four profilers -- nsys (timeline + kernel durations), ncu (roofline /
  occupancy / arithmetic intensity), zymtrace (CPU+GPU flamegraph), and DCGM
  (byte-grounded %SoL) -- into ONE ranked, data-backed recommendation ledger for a
  vLLM deployment. Every recommendation (including concrete vLLM source/config
  changes) cites its backing artifact + sol_rigor tier, and each is staged as an
  A/B experiment so it is proven, not asserted. Emits a findings doc + perftunereport
  view + publishes to the perf-lake. The holistic-analysis phase of
  inference-model-optimize (Phase 2.5 + 8.5), usable standalone. Triggers on
  "synthesize the profiles", "what should I change to make this faster",
  "holistic perf analysis", "fuse nsys ncu zymtrace dcgm", "ranked perf
  recommendations", "suggest vllm changes from profiles", "explain the bottleneck
  with data", or any combination of
  "synthesize / fuse / recommend / holistic / explain" with "profile / nsys / ncu
  / zymtrace / dcgm / roofline / bottleneck / vllm".
allowed-tools:
  - mcp__profile_and_optimize__findings_record
  - mcp__profile_and_optimize__evidence_init
  - mcp__profile_and_optimize__perf_tune_report_dcgm_correlate
  - mcp__profile_and_optimize__perf_tune_report_report_render
  - mcp__profile_and_optimize__perf_tune_report_publish_to_lake
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Read
  - Write
---

# inference-perf-synthesize

## Purpose

The four profiler skills each answer a *different slice* of "where does the time
go and why":

| Profiler | Skill | Answers | Rigor tier |
| --- | --- | --- | --- |
| nsys | [`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md) | timeline + absolute kernel durations + host-gap structure | L2-ish |
| ncu | [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md) | per-kernel occupancy / regs / DRAM BW / arithmetic intensity / tensor-core engagement | **L4** |
| zymtrace | [`analyze-zymtrace-workload`](/plugins/profile-and-optimize/skills/analyze-zymtrace-workload/SKILL.md) | CPU+GPU flamegraph cross-view (sample-share) | L1 |
| DCGM | [`inference-dcgm-correlate`](/plugins/profile-and-optimize/skills/inference-dcgm-correlate/SKILL.md) | byte-grounded workload-level %SoL over the bench window | L3 |

**Do NOT conclude "kernel at-roof / no lever" from SM-busy% (ncu `--set basic` or DCGM SM-active).**
An 88-92% SM-busy can hide a kernel at <15% of its FLOP-roofline (persistent/split-K spin). Decide
at-roof vs headroom only from ncu `--set full` achieved-TFLOPS vs the FLOP ceiling. Worked example:
a DeepSeek-V4-Flash FP8 GEMM read 88-92% SM but was 1-14.5% of FP8 FLOP-SoL -> the synthesized
recommendation was "throughput tier", not "kernel rewrite". (Canon: the ncu capture-hygiene section of
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md).)

Run in isolation they produce four disconnected artifacts. This skill **fuses
them into one ranked recommendation ledger**: it reconciles the four views
(e.g. zymtrace says category X is hot -> DCGM confirms X is byte-bound -> ncu
shows X reaches only Y% of the tensor-core roofline -> nsys shows a host gap
before it), turns that into a **ranked list of addressable levers** (config knobs
AND concrete vLLM source changes), and **stages each lever as an A/B experiment**
so the recommendation is proven on this exact workload before it is believed.

It does NOT replace the four skills -- it consumes their outputs. Run the
profilers first (or let [`inference-model-optimize`](/plugins/profile-and-optimize/skills/inference-model-optimize/SKILL.md)
Phase 2 drive them), then run this to synthesize.

## When to use

- After profiling a deployment with 2+ of the four profilers, to get a single
  prioritized "what to change" answer instead of four separate reports.
- `inference-model-optimize` Phase 2.5 (diagnose the baseline -> recommend levers
  that seed the tuning phase) and Phase 8.5 (post-champion: what's left on the
  table + future-work levers).
- When a stakeholder asks "why is it slow and what do we change" and the answer
  must be data-backed and explainable, not asserted.

Do **not** use this skill for:

- Capturing a profile -- that is the four profiler skills. This skill reads what
  they captured.
- A which-config sweep -- that is [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md)
  (this skill *feeds* it the candidate levers).
- A single-profiler reading where no cross-view is needed -- just use that
  profiler's skill.

## Example prompts

- "Synthesize the nsys + ncu + zymtrace + DCGM captures into ranked recommendations."
- "What should I change in vLLM to make this faster, backed by the profiles?"
- "Holistic perf analysis of the gemma4 baseline -- explain the bottleneck with data."
- `/inference-perf-synthesize --bundle <evidence-bundle> --focus latency`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. An evidence bundle with at least two of the four profiler artifacts present
   (`*.nsys-rep` / `*.ncu-rep` / zymtrace TSVs / `dcgm_correlation.json`). It will
   not synthesize from a single source (that is not a cross-view).
2. The capture-validation gates passed for each source (an empty nsys rep is a
   capture bug, not a finding -- see the nsys capture-validation gate). The skill
   refuses to draw a conclusion from an unvalidated/empty artifact.
3. For any recommendation it stages as an A/B: a writable experiment-prefixed
   overlay path + an idle GPU node (plain-K8s GB300 / legacy Slurm-on-K8s) (the A/B itself runs via `inference-tune-sweep`
   / the controlled-A/B harness).

## Capture orchestration (how to actually get the 4 artifacts)

Getting the four artifacts is itself constrained -- these rules are
hard-won. Encode them so a capture is right the first time:

- **One pod cannot run all four profilers.** `ncu` and the zymtrace CUDA injection
  (`CUDA_INJECTION64_PATH`) are **mutually exclusive**, `nsys` needs the sidecar
  image or vLLM `/start_profile`. So a full capture is **2-3 coordinated pods**:
  (A) the bench pod with the zymtrace injection ON + vLLM torch profiler
  (`VLLM_TORCH_PROFILER_DIR` + `/start_profile`/`/stop_profile` window), (B) a
  SEPARATE `nsys` launch-wrap pod (injection cleared, `--cuda-graph-trace=node`),
  (C) a SEPARATE `ncu` sister pod (`--set basic`, `--kernel-name` scoped). DCGM is
  correlated POST-HOC over the bench window (no in-pod agent).
- **Use a submit-and-queue self-driving Job on a contended cluster** (per
  `server/AGENTS.md`): the capture pod embeds serve + bench + window-stamp, writes to the
  RWX PVC, and EXITS to free the GPU, `backoffLimit` re-queues through preemption,
  guard idempotency on the **result file** (not a marker that could be touched on
  failure). Make the bench **variance-controlled** (>=3 trials/concurrency, same pod
  => same node) so the baseline is VERDICT-grade, not a single-shot DRAFT.
- **Offline-tokenizer gotcha (L1):** in an `HF_HUB_OFFLINE=1` pod, `vllm bench serve
  --model <hf-id>` will try to fetch the tokenizer from huggingface.co and fail with
  `LocalEntryNotFoundError`. ALWAYS pass `--tokenizer <local-weights-dir>` so the
  bench client uses the on-disk tokenizer. (This silently produced zero results until
  caught.)

## Workflow

### Phase 1: inventory + validate the captured artifacts

Read the bundle. List which of the four profilers produced a *valid* artifact
(apply each capture-validation gate). Record the `sol_rigor` available per source.
Refuse to proceed if fewer than two valid sources exist.

### Phase 2: reconcile the four views

Build a per-kernel-family / per-phase table that joins the sources:

- zymtrace sample-share (L1) -> which category/kernel is hot, CPU vs GPU side.
- DCGM byte/FLOP (L3) -> is that category byte-bound or compute-bound at the
  workload level.
- ncu (L4) -> for the dominant kernel: achieved roofline %, tensor-core engaged?,
  occupancy / stall reason / register or smem pressure.
- nsys -> absolute duration + whether a host gap precedes it (host-bound vs
  GPU-bound), cudagraph-captured or not.

Flag contradictions (e.g. zymtrace says hot but ncu says near-roofline -> not
addressable) -- a contradiction is itself a finding, not something to smooth over.

### Phase 3: emit the ranked recommendation ledger

For each addressable lever, write a ledger row:

```text
rank | lever | type(config|vllm-src|kernel) | expected_effect | evidence_artifacts[] | source_refs[] | sol_rigor | confidence(DRAFT)
```

`source_refs[]` is MANDATORY for any `vllm-src` / `kernel` lever: the exact
vLLM/SGLang branch + commit (resolved to a GitHub URL) and delivery
(image|overlay|patchedVllm|infr-patch) from the bundle's
`experiment_provenance_v1` block + the campaign's source registry, so
a reviewer can open the patch that produced the win. A source-code recommendation
with no `source_refs[]` is not actionable. **Code-under-test provenance
match:** the `expected_effect` number MUST come from an `evidence_artifacts[]` campaign whose
provenance `delivery`+`commit` matches the recommendation's `source_refs[]` -- an
`overlay`/offline-prepped campaign cited as the benefit of an `infr-patch` recommendation is a
cross-tier DRAFT defect (even when the kernels match). Cite the patch's own run.

- **type=config**: a `vllm.extraArgs` / cudagraph / kv-cache-dtype change ->
  hand to [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md).
- **type=vllm-src**: a concrete vLLM code change, staged as a runtime overlay per the
  delivery ladder (`server/AGENTS.md` "Experiment delivery ladder"): a `subPath` ConfigMap
  (`overlay_mode: subpath`) for a few files,
  or the initContainer patch-set (`overlay_mode: patchset-initcontainer`)
  for a whole patch
  set on a pullable base -- include the file + the diff sketch + which artifact motivates it.
- **type=kernel**: a custom-kernel opportunity -- classify it with the K/R/H/P/A
  rubric and name the production-representative baseline (a win over a
  strictly-lower-H/R baseline is a DRAFT, never a VERDICT).

Every row MUST list `evidence_artifacts[]` (the exact `.nsys-rep` / `.ncu-rep` /
zymtrace TSV / `dcgm_correlation.json` / `kernels.json` paths). A row with no
artifact citation is rejected -- this is the explainability contract.

### Phase 4: prove each lever (A/B), do not assert

Take the top-ranked levers and stage each as an experiment:

- config -> a `campaign_run` cell or a controlled A/B.
- vllm-src -> an overlay deploy (experiment-prefixed) + a same-node controlled
  A/B vs the unpatched arm, **both under matching cudagraph_mode**.

Gate each with [`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md) (or
`perf_baseline_diff`). A lever is promoted from DRAFT (predicted) to VERDICT
(proven) only after a same-node + >=3-trial + metric-isolated A/B confirms it.
A lever whose A/B comes back within noise is recorded as "tried, null" -- kept in
the ledger so the next operator does not re-explore it.

### Phase 5: persist + publish

- Write `findings/synthesis-<ts>.md` to the evidence bundle (the ranked ledger +
  the reconciliation table + the A/B verdicts), via `findings_record`.
- Feed the ledger into the perftunereport campaign as a recommendations view so it
  renders in the PDF and lands in the perf-lake under `campaign=<run-id>` with
  `focus` + `sol_rigor` recorded.
- Record the campaign + `s3://perf-lake/...` paths back into the bundle's
  `SOURCE.md`/`summary.md`.

## Explainability contract (mandatory)

Per `server/AGENTS.md` "All attribution claims must be matched with collected
profile data", **every** recommendation and every where/why verdict this skill
emits carries:

1. the backing artifact path(s),
2. the `sol_rigor` tier of that evidence (`L1` zymtrace-proxy / `L3` DCGM /
   `L4` ncu),
3. a DRAFT/VERDICT tier (DRAFT = predicted from profiles. VERDICT = proven by an
   A/B).

A claim missing any of the three is not emitted. An empty/failed capture is a
capture bug to fix, never evidence of "unprofilable" -- clear the capture-
validation gate first, then conclude. (For zymtrace specifically, an empty-now
right after the bench is usually ClickHouse INGEST LAG, not a bug -- wait + requery
for the freshest data. See [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md).)
(One genuine environmental exception seen on
GB300: a CUDA 12.x image on a CUDA 13.x driver skews CUPTI ->
`CUPTI_ERROR_INVALID_DEVICE` -> 0 kernels for ALL CUPTI clients regardless of
hygiene. Grep `CUDA versions. CUPTI/Runtime/Driver` -> use a CUDA-13 image or zymtrace.)

## Verdict rigor (DRAFT vs VERDICT)

Synthesis output is **predictive (DRAFT) by construction** -- it ranks levers from
profile evidence. A lever becomes a VERDICT only after Phase 4's A/B proves it
(same-node, >=3 trials, metric-isolated median TPOT/ITL for latency, against the
production-representative baseline). Never report a predicted speedup as achieved.

## Kernel rubric (K/R/H/P/A)

Any `type=kernel` recommendation records the candidate AND the named production
baseline's `(K,R,H,P,A)` coordinates per `server/AGENTS.md` "Kernel rubric".
The H + P proof (tensor-core engagement + roofline %) comes from the ncu artifact,
a win over a strictly-lower-H/R baseline is a DRAFT, never a VERDICT.

## Safety

- **No silent fallbacks / no fabricated evidence** -- refuses to synthesize from
  < 2 valid sources or from an unvalidated capture. Rejects any uncited claim.
- **A/B before believe** -- a `vllm-src` recommendation is staged as an
  experiment-prefixed overlay and proven on this workload. It is never applied to
  a standing deploy or asserted from first principles.
- **Experiment isolation + scheduling** -- all A/B arms are experiment-prefixed +
  `experiment=<id-slug>` labeled and torn down by label. Scheduled per the cluster
  profile (plain-K8s `default-scheduler` / the Slurm scheduler on legacy Slurm-on-K8s).
- **No external posting** of recommended vLLM changes (no upstream PR) without
  explicit per-turn operator approval.

## Source-of-truth references

- The four profiler skills (inputs):
  [`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md),
  [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md),
  [`analyze-zymtrace-workload`](/plugins/profile-and-optimize/skills/analyze-zymtrace-workload/SKILL.md),
  [`inference-dcgm-correlate`](/plugins/profile-and-optimize/skills/inference-dcgm-correlate/SKILL.md).
- [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md) +
  [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) -- where the
  staged A/Bs run and the recommendations get published. The ranked levers here
  become the cross-engine variant arms. The proven survivors flow to
  `perftunereport champion_select` (the baseline-vs-top-X production pick + page 8).
- `server/AGENTS.md` -- attribution-must-be-profiled, speed-of-light framing,
  DRAFT-vs-VERDICT, kernel rubric.

## Full-context reporting (no bare numbers)

Per the methodology canon "Every performance number carries its full context (no bare numbers)"
(`docs/METHODOLOGY.md`): every number this
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
Speed-of-Light and publish `--strict`. Canonical map: `server/AGENTS.md`
"Rigor discipline index" / `docs/METHODOLOGY.md`. Skills that
do not produce measurements are exempt (`server/AGENTS.md` "Speed-of-light framing").

## Asset validation (review + FAIL LOUD)

Every asset this skill emits (findings doc / recommendation ledger / perfreport view) is held to
`server/AGENTS.md` "Validate every generated asset"
(`docs/METHODOLOGY.md`): the generator **FAILS LOUDLY** on missing/bad data (a recommendation with no backing
profile artifact, an empty/degenerate fused view, `unknown`/null where a value is required) ->
raise / flag loudly, never a silent placeholder or an unbacked recommendation, and the agent
**REVIEWS** the rendered doc/ledger for human-sense + 100% accuracy (every recommendation cites
its backing artifact + sol_rigor tier, the A/B staging is sound) and **rebuilds** it if wrong --
never ships a wrong/confusing synthesis with a caveat.

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per "The Grind Mandate" (`server/AGENTS.md` + `docs/METHODOLOGY.md`). A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.
