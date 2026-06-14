---
name: inference-perf-bench
last_validated: 2026-06-07
description: >-
  Canonical inference perf-bench skill (formal name. The colloquial alias
  is `ai-bench` - identical behaviour). Drives NVIDIA AIPerf + the
  replay-playback dataset against an in-cluster vLLM endpoint to
  measure TTFT, ITL, throughput, tok/s/user, request latency, and prefix
  cache hit rate. Iterative 9-phase workflow. Use when promoting a model to
  staging/prod, after vllmArgs / vLLM / KV-cache-dtype changes, or for A/B
  comparison across configs. Triggers on "perf-bench", "AIPerf", "Replay
  Playback", "throughput sweep", "TTFT P95", "concurrency sweep on
  inference", "/run-perf-bench", "benchmark throughput", "benchmark
  latency", "run aiperf", "performance test", or any combination of "perf /
  latency / throughput / tps / TTFT / ITL" with "inference / vllm /
  serverless / bench".
allowed-tools:
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Bash(kubectl:*)
  - Bash(huggingface-cli:*)
  - Bash(curl:*)
  - Bash(jq:*)
  - Bash(date:*)
  - Read
  - Write
---

# inference-perf-bench

> **Fast model loading (always-applied):** when standing up a vLLM deploy, never load 100s-of-GB single-stream via s3fs FUSE (a GLM-5.1-sized model takes ~50 min that way. See `docs/METHODOLOGY.md`). Prefer, in order: a fast model-loading endpoint when reachable -> parallel multipart to local NVMe (`server/tools/stage-model-parallel.py`) -> `runai_model_streamer` (`--load-format runai_streamer`) -> tensorizer. Flag a slow load loudly: if effective rate < ~500 MB/s on a large model, STOP and switch. Details: `server/docs/inference-fast-model-loading.md`.

## Purpose

Drive inference performance benchmarks against an in-cluster inference
endpoint using NVIDIA's [AIPerf](https://github.com/ai-dynamo/aiperf)
tool and the `replay-playback` HuggingFace dataset of
recorded multi-turn agentic-coding conversations (the dataset may be
gated - confirm your HF token has access before starting). Runs from
an in-cluster bench pod for accurate latency measurement. Supports
concurrency sweeps, multi-model comparison, and server-side metric
capture.

> **Steady-state window (throughput trap):** drive `num_prompts >= 2*c` at each concurrency
> `c` (AIPerf/`vllm bench serve` measure throughput over the full run. Too few prompts make
> the window ramp/drain-dominated and undercount high-c throughput ~1.6-1.8x). Default
> sweeps use `num_prompts=2*c` (e.g. 128 @ c=64). See `docs/METHODOLOGY.md` trap 4.

The core benchmark loop is a standard 9-phase AIPerf workflow (Phase B
below). This file does not belabor the per-phase mechanics - it adds
the cockpit-specific cross-references (perf-baseline registry, perf-lake
export, evidence bundle scaffolding) that turn a bench run into a
durable, comparable result.

## When to use

- Before promoting a new model to staging or prod - validate
  performance on dev before promotion.
- After vllmArgs tuning, vLLM version bumps, or KV cache dtype
  changes - re-measure to confirm no regression.
- Side-by-side comparison of two model configs (NVFP4 vs FP8,
  EAGLE3 on/off, different tensor-parallel sizes, etc.).

Do **not** use this skill for:

- Training perf measurement (step-time / MFU / scaling-efficiency) -
  out of scope for this skill, which measures serving-side
  TTFT / ITL / throughput.
- NCCL collective-bandwidth measurement - use the upstream
  `nccl-tests` suite.
- Quality / accuracy evaluation (GPQA, MMLU-Pro,
  Terminal-Bench, SWE-Bench) - use
  [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md).
- Model deployment / promotion itself - creating or updating standing
  deployments is a platform operation outside this plugin's scope.

## Example prompts

- "Run perf-bench on the kimi-k25 dev deployment with concurrency
  sweep 1,4,8,16,32."
- "Benchmark throughput of glm-5-fp8 against the 2025_07 split."
- "Compare NVFP4 vs FP8 on the deepseek-v4 dev pods using AIPerf."
- "TTFT P95 sweep on the new minimax-m2.7 deployment."
- `/run-perf-bench --model kimi-k25 --concurrency 16 --split 2025_07`
- `/inference-perf-bench --models kimi-k25,glm-5-fp8 --concurrency 1,4,8,16,32`

## Prerequisites

Standard bench prerequisites plus the cockpit's provenance contract:

1. **`HF_TOKEN`** - a HuggingFace read token with access to the
   `replay-playback` dataset.
2. **`kubectl` context for a dev cluster** - run
   `kubectl config use-context <ctx>` before invoking.
3. **Namespace** - the namespace where the bench pod (e.g.
   `my-perf-bench`) lands.
4. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** - set by the bundled MCP server at install
   time. The cockpit writes the result bundle under
   `${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/inference-perf-bench/<UTC-ts>/`.

## Interaction style

Iterative - the 9-phase runbook pauses naturally at each
phase. After Phase 5 (the actual benchmark run) completes, hand off to
the bridge skills below before deleting the bench pod.

## Workflow

### Phase A: scaffold an evidence bundle (cockpit-side)

Optional but recommended. Use [`evidence-bundle-init`](/plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md):

```text
/evidence-bundle-init --family inference-perf-bench --intent "perf-bench on <model> at concurrency <C>"
```

This creates `experiments/artifacts/inference-perf-bench/<run-id>/` with
`SOURCE.md`, `summary.md`, `commands/`. Every shell command in the
9-phase workflow can then be captured as a four-file tuple
under `commands/` for the reproducibility-grade-evidence rule
(`server/AGENTS.md`).

### Phase A.5: pre-bench - quiet Slurm (Slurm-on-K8s clusters only)

If the inference deployment runs on a Slurm-on-K8s cluster where slurmd worker
pods (`slurm-b200-*`, etc.) co-host the vLLM replicas on the same GPU
nodes, drain the Slurm partition for
the bench window so a co-tenant Slurm job can't steal CPU, host RAM,
`/dev/shm`, PCIe BW, or IB fabric capacity from the inference pod
mid-measurement. Use the `slurm_quiet_window` MCP verb:

```text
mcp__profile_and_optimize__slurm_quiet_window with:
  args: ["--nodes", "<comma-list of slurm-* worker pod names>",
         "--cmd", "<your Phase B/C bench cmd>",
         "--bundle", "<your bundle>/slurm-quiet-window-<UTC>",
         "--json"]
  i_understand_this_substitutes_nodes: true
```

The orchestrator drains, runs your bench cmd, and ALWAYS resumes via a
Python try/finally so a Ctrl-C, exception, or non-zero bench-cmd exit
cannot leave the partition stuck in `drained`. Skip this phase when
the cluster has dedicated inference nodes (no Slurm-on-K8s co-tenancy) or when
the operator has already drained Slurm by hand.

### Phase B: run the 9-phase benchmark loop

The phases execute in order: (1) Dataset Check, (2) Bench Pod Setup, (3)
Identify Target Endpoints, (4) Capture Pre-Run Server Metrics, (5) Run
Benchmark, (6) Monitor, (7) Capture Post-Run Server Metrics, (8)
Download Results, (9) Report.

The four canonical vLLM Prometheus metrics captured in Phases 4 + 7
are also the canonical metric set
[`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md)
exposes as a worked example: `vllm:prefix_cache_hit_rate`,
`vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`,
`vllm:avg_generation_throughput_toks_per_s`.

### Phase C: register a baseline (cockpit-side)

After Phase 9 (Report), the operator has a
`perf-bench-report-<YYYY-MM-DD>.md` plus per-concurrency aiperf logs.
Register the result as a baseline for future regression diffs:

```text
/inference-perf-baseline-bridge record \
  --model <model> \
  --source experiments/artifacts/inference-perf-bench/<run-id>/
```

See [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md)
for the full contract.

### Phase D: land the result in the perf-lake (MANDATORY)

Per the `server/AGENTS.md` "Experiment Isolation &
Traceability" rule, a measurement is not a result until it lands in the
perf-lake with Speed-of-Light rooflines. The canonical path is the
`perftunereport` pipeline (see [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md)),
keyed by `campaign=<id>` where `<id>` is this bundle's run-id:

```text
perftunereport campaign_init  --config <campaign>.yaml      # campaign id = run-id; set focus: latency|throughput|mixed
perftunereport cell_run ...   --i-understand-this-submits-jobs
perftunereport atlas_aggregate --campaign <id>
perftunereport dcgm_correlate  --campaign <id> --cell-id <cell> --frozen-yaml <dcgm-frozen>.yaml  # raises sol_rigor to L3 (pages 6/6b)
perftunereport import_roofline_sweep --campaign <id> --bundle <roofline-out> --hardware GB300 --tensor-parallel <tp> --cache-mode cold  # page 7
perftunereport report_render   --campaign <id>              # SoL roofline pages; sets sol_complete + sol_rigor
perftunereport publish_to_lake --campaign <id>              # atlas_v1 + campaign_v1 parquet (always lands; records focus + sol_rigor)
```

**Always-on prefill/decode roofline.** Before publish, also run the gated
`*-deploy/profiling/roofline-sweep.sh` (decode-concurrency + prefill-ISL sweep with
per-cell in-pod `dcgmi` PROF) and `import_roofline_sweep` so the campaign carries **page 7**
(per-GPU roofline + HBM%/tensor%/SM%-vs-concurrency - the "what C maxes the TFLOPs / is decode
>=75% HBM / which sharding degree" answers. Per-(c,ISL) DCGM lands in `atlas_v1.extra_json`).
Sweep every candidate config (TP / KV-dtype) so page 7 overlays them. See
[`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) Phase D3.

**Always-publish with focus + sol_rigor.** EVERY run publishes a
`sol_complete` roofline and records `focus` (set it in the campaign YAML) +
`sol_rigor` (`L4` ncu | `L3` DCGM | `L1` zymtrace-proxy | `none`). The DCGM +
zymtrace capture below RAISES `sol_rigor` toward L3/L4 - it is not a
publish gate. `publish_to_lake` **never refuses** by default: a latency-bound /
proxy / `dcgm_grounded=false` run is a first-class published result, with the
gap RECORDED on `campaign_v1` + warned (pass `--strict` only when you want
publish to refuse). Still capture DCGM (SM/DRAM/tensor/GR + NVLINK bytes)
concurrently with the bench window and fold it in via `dcgm_correlate` (or the
[`inference-dcgm-correlate`](/plugins/profile-and-optimize/skills/inference-dcgm-correlate/SKILL.md) skill) for a
tighter (L3) roofline.

This pipeline records gaps loudly instead of leaving silent
blanks: `import_perf_bench` prints a `WARNING:` for any cell that imports as
STATUS_FULL but lacks `Median TTFT (ms)` / `Request throughput (req/s)`
(such a cell produces no scatter point), `atlas_aggregate` warns on 0
plot-ready / full-but-unplottable cells, `report_render` records every
omitted SoL page (why + how-to-fix) on a completeness page +
`report_status.json`, and `publish_to_lake` records the gap on the lake row
(it lands by default, `--strict` refuses). If a cell's bench output
is missing those lines, re-run the bench so it prints them, then
re-import + re-aggregate + re-render.

## Safety

- **Read-only on the cluster except for the bench pod.** The
  workflow creates and (in Phase Cleanup) destroys a single
  bench pod (e.g. `my-perf-bench`) via `kubectl run` / `kubectl delete`.
  No other cluster mutation.
- **No public-gateway traffic by default.** Phase 5 runs against the
  cluster-internal service DNS, never a public gateway -
  a public gateway adds variable cross-region overhead that
  contaminates measurements (dev-vs-prod throughput skews of 3x have
  been traced to exactly this).
- **Bench pod cleanup is required.** The Cleanup
  step (`kubectl delete pod my-perf-bench`) MUST run before the
  evidence bundle is finalized. Orphan bench pods consume cluster
  capacity for hours.
- **No credential commit.** `HF_TOKEN` and any provider API keys
  stay in env vars. Never written into the evidence bundle's
  `SOURCE.md` or commit history.

## Experiment isolation & traceability (mandatory)

Per the `server/AGENTS.md` "Experiment Isolation &
Traceability" rule (and `docs/METHODOLOGY.md`):

- Any disposable serve/bench deployment this workflow creates MUST use
  experiment-unique names derived from the run-id and carry the label
  `experiment=<id-slug>`. NEVER reuse standing/platform/migration names
  (standing deploys, shared `*-cache` PVCs, anything labeled
  `migration=*`). Cluster-scoped PV
  names are global - a collision silently breaks another owner's PVC.
- Tear down by label (`kubectl delete deploy,pod,pvc,secret -l
  experiment=<id-slug>`). Pre-clear the attacher finalizer on `Retain`
  experiment PVs before deleting. Verify standing/migration objects are
  untouched + Ready afterward.
- **DCGM + zymtrace capture is required during the Phase 5 bench window**
  (DCGM via the Prometheus MCP PromQL for the window+node. Zymtrace via the
  always-on DaemonSet flamegraph for the window+node) so the perf-lake
  roofline pages render. zymtrace flushes to ClickHouse asynchronously, so an
  empty L1 right after the window is **ingest lag, not absence** -- wait + requery
  for the freshest data (see
  [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md)).
  Record the created object names + perf-lake `campaign=<id>` in the bundle `SOURCE.md`.

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

Every result table this skill produces MUST carry a `%SoL` column
alongside the absolute throughput / latency numbers. Per the
`server/AGENTS.md` "Speed-of-light framing" section:

- Workload-level: peak `output_tps_per_gpu` compared to the
  HBM-bandwidth ceiling (`b200_sm100.hbm3e_tbps = 8.0 TB/s` ÷
  per-token footprint). Per-token footprint depends on model,
  e.g. GLM-5.1 NVFP4 at ISL=4096 / OSL=512 is ~7 GB/token.
- Source of all peak numbers: `configs/sol-ceilings.yaml`
 - cite by key path (`b200_sm100.hbm3e_tbps`, etc.). Never inline.
- Per-run sol-summary doc lives at `<campaign>/sol-summary.md` and
  carries the workload-level %SoL row. The perf-report PDF page 4
  picks it up automatically when zymtrace per-category data is also
  present in the bundle.

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

## Verdict rigor (DRAFT vs VERDICT)

Per `server/AGENTS.md` "Verdict rigor: DRAFT vs VERDICT", tier every bench number.
A single sweep is a **DRAFT**. Promote to a **VERDICT** only when variance-controlled
(same-node, >=3 trials, mean +/- std), metric-isolated (median TPOT/ITL for
decode-latency claims - output tok/s at small num_prompts is TTFT-dominated, NOT
decode), and against a production-representative baseline (cudagraph-on, shipped
backend. Never an eager strawman). Mark the campaign `verdict_tier` accordingly. The
perf-lake writer gates `verdict_tier=verdict` on this provenance.

**Quant-format / serve-backend claims are a MATRIX, not a single cell** (per
`server/AGENTS.md` "Validate the matrix, never generalize from one cell" +
`docs/METHODOLOGY.md`). When A/B-ing quant formats or serve
backends (NVFP4 marlin vs cutlass, FP8 compressed-tensors, ...), a single
(backend, concurrency) point is NEVER a universal verdict -- the winner is
concurrency-dependent (e.g. on Qwen3: FP8 wins c1 latency, NVFP4-marlin the c16 knee,
NVFP4-cutlass throughput at c64-256. No universal best). MANDATORY for such a verdict:
run the **serve-backend x concurrency matrix same-node** (one pod, all backends, 3
trials) and aggregate the cells. Root-cause any failed/degenerate cell by
reading the `(EngineCore pid=...)` worker traceback instead of
assuming the backend is blocked. Report the winner PER concurrency regime.

## Source-of-truth references

- Pair: `ai-bench` (colloquial alias of this
  skill), [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md)
  (quality-side counterpart).
- Bridge: [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md).
- `server/AGENTS.md` - fail-fast + provenance rules.
