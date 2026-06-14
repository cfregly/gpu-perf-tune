---
name: inference-perf-baseline-bridge
last_validated: 2026-05-24
description: >-
  Bridge between the inference-perf-bench / ai-bench output bundle and the
  profile-and-optimize perf-baseline registry. Knows the canonical inference perf
  metric shape (TTFT p50/p95/p99, ITL p50, throughput, tok/s/user, request
  latency p50, prefix cache hit rate delta, GPU cache utilization peak),
  records it under inference_perfbench_v1 schema, and on diff applies
  per-metric tolerances tuned for inference workloads (TTFT regressions
  <5 percent, throughput <3 percent, cache hit rate <2 absolute points).
  Wraps the existing perf-baseline-record and perf-baseline-diff MCP verbs,
  no new MCP verbs introduced. Triggers on "register inference baseline",
  "inference perf-baseline", "diff inference perf", "is this inference
  regression real", "perf-bench baseline", "ai-bench baseline", or any
  combination of "register / record / diff / compare" with "inference /
  ai-bench / perf-bench / vllm / kimi / glm / deepseek / minimax".
allowed-tools:
  - mcp__profile_and_optimize__perf_baseline_record
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__search_evidence
  - mcp__profile_and_optimize__search_runbooks
  - Bash(jq:*)
  - Bash(sha256sum:*)
  - Bash(date:*)
  - Read
  - Write
---

# inference-perf-baseline-bridge

## Purpose

Translate the output of an
[`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) (or its
colloquial alias `ai-bench`) run into the
shape the workload-agnostic
[`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md) /
[`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md) registry
expects. The bridge fixes three pain points:

1. The upstream `perf-bench` skill's report is a markdown table per
   model - not a JSON payload the registry can hash and diff.
2. The metric set (TTFT p50/p95/p99, ITL p50, tok/s, tok/s/user,
   request latency p50, prefix cache hit rate delta, GPU cache
   utilization peak) needs a stable schema name so future diffs find
   apples-to-apples baselines.
3. Per-metric regression tolerances differ - TTFT regressions of 5%
   are usually real, throughput regressions of 3% may be capacity
   contention, prefix cache hit rate movements should be expressed as
   absolute percentage points (not percent-of-percent).

This bridge does NOT add a new MCP verb. It wraps the existing
`mcp__profile_and_optimize__perf_baseline_record` and
`mcp__profile_and_optimize__perf_baseline_diff` verbs with a fixed schema kind
(`inference_perfbench_v1`) and operator-tunable per-metric tolerances.

## Schema: `inference_perfbench_v1`

```json
{
  "schema": "inference_perfbench_v1",
  "model": "<served-model-name>",
  "model_path": "<HF-org/repo>",
  "deployment": {
    "namespace": "<k8s-namespace>",
    "cluster": "<kube-context>",
    "service": "<svc-name>",
    "vllm_version": "<v0.NN.M>",
    "tensor_parallel_size": <int>,
    "quantization": "<NVFP4 | FP8 | BF16 | FP4 | INT4>",
    "kv_cache_dtype": "<fp8_e4m3 | bf16 | ...>",
    "speculative_decoding": "<eagle3 | mtp | none>"
  },
  "workload": {
    "dataset": "<hf-org>/replay-playback",
    "split": "<2025_07 | 2025_04 | 2025_01 | all>",
    "conversation_num": <int>,
    "concurrency_sweep": [<int>, ...],
    "input_token_band": "<1k | 10k | 100k | mixed>",
    "output_token_band": "<1k | 1.5k | 2k | mixed>"
  },
  "metrics": {
    "<concurrency>": {
      "throughput_toks_per_s": <float>,
      "ttft_p50_ms": <float>,
      "ttft_p95_ms": <float>,
      "ttft_p99_ms": <float>,
      "itl_p50_ms": <float>,
      "tps_per_user": <float>,
      "request_latency_p50_ms": <float>,
      "errors": <int>
    }
  },
  "server_metrics_delta": {
    "prefix_cache_hit_rate_pre": <float>,
    "prefix_cache_hit_rate_post": <float>,
    "gpu_cache_usage_perc_peak": <float>,
    "num_requests_running_peak": <int>,
    "avg_generation_throughput_toks_per_s": <float>
  },
  "kernel_class_gpu_pct": {
    "fp8_bmm": <float|null>,
    "tp_all_reduce": <float|null>,
    "cuda_event_sync": <float|null>,
    "mla_attention": <float|null>,
    "moe_routing_finalize": <float|null>,
    "per_token_group_quant": <float|null>,
    "<other-class>": <float|null>
  },
  "cpu_spinpoll_pct": <float|null>
}
```

## Optional fields

The two top-level fields `kernel_class_gpu_pct` and `cpu_spinpoll_pct`
are **optional**. They surface profile-grade signals that the upstream
`perf-bench` runbook does not produce on its own - they are populated
when the operator also captures a [zymtrace](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md)
or Nsight-Systems profile against the same time window.

Field shapes and origins:

- **`kernel_class_gpu_pct`** is an **open map** of kernel-class name to
  percent of GPU time. The canonical class set is the one typically
  observed in a plain-FP8 MoE serving profile:
  `fp8_bmm`, `tp_all_reduce`, `cuda_event_sync`, `mla_attention`,
  `moe_routing_finalize`, `per_token_group_quant`. Classes outside this
  set go under arbitrary `<other-class>` keys (free-form). The map is
  derived from a GPU-side `event_kind = 'cuda'` zymtrace query grouped
  by kernel-name regex. The source bucket-rules live in the
  [zymtrace-anchored-query](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md) skill's
  "Kernel-class bucketing" appendix.
- **`cpu_spinpoll_pct`** is a single scalar: the percent of CPU on-cpu
  samples that fall inside vLLM's spin-poll IPC path
  (`shm_broadcast.py::{acquire_read,wait,memory_fence,timeout_ms,should_warn,dequeue}`
  + `utils.py::sched_yield`). Populated from the same zymtrace bundle's
  CPU-side `event_kind = 'on_cpu'` query.

**Diff semantics on null**: when either side of a per-metric comparison
is `null`, the bridge skips that metric in `Phase 2: per-metric diff
with tolerances` and emits a `NULL_LEGACY_BASELINE` row in the diff
output so the operator can see which fields were unmeasured. This keeps
existing `inference_perfbench_v1` baselines registered before these
fields existed diff-compatible with new bundles that include them - the
registry never breaks.

When a true breaking change is needed (e.g. metric-shape restructure of
`metrics.<concurrency>`), the schema name will bump to
`inference_perfbench_v2`. Additive-only changes stay on `_v1`.

## Default per-metric tolerances

| Metric | Default tolerance | Direction | Rationale |
| --- | --- | --- | --- |
| `throughput_toks_per_s` | 3% | regression = current < baseline | Throughput is the headline metric. Small regressions are real |
| `ttft_p50_ms` | 5% | regression = current > baseline | TTFT is sensitive to scheduler / cache cold-start, 5% is the usual signal floor |
| `ttft_p95_ms` | 10% | regression = current > baseline | Tail latency wider band - outlier-prone |
| `ttft_p99_ms` | 15% | regression = current > baseline | Same |
| `itl_p50_ms` | 3% | regression = current > baseline | ITL is decode-bound. Small regressions are real |
| `tps_per_user` | 3% | regression = current < baseline | Per-user throughput. Same band as overall throughput |
| `request_latency_p50_ms` | 5% | regression = current > baseline | Same as TTFT p50 |
| `prefix_cache_hit_rate_post` | 2 absolute pts | regression = current < baseline | Cache hit rate moves in absolute pts, not percent-of-percent |
| `gpu_cache_usage_perc_peak` | 5 absolute pts | regression = current > baseline | Capacity ceiling proxy |
| `kernel_class_gpu_pct.<any>` | 3 absolute pts | regression = current > baseline | Kernel-class %-of-GPU drift, 3pp is the usual signal floor for hot kernels (e.g. TP all-reduce drifting from an expected ~10% to an observed ~15% is an actionable 5pp delta) |
| `cpu_spinpoll_pct` | 5 absolute pts | regression = current > baseline | CPU spin-poll waste, >5pp jump indicates `VLLM_USE_SHM_BROADCAST_BLOCKING=1` regressed off, or a new vLLM release re-introduced busy-wait IPC |

Operator can override any of these via `--tolerance.<metric> <value>`
on the command line. The two optional fields' tolerances are only
evaluated when both sides of the diff have non-null values. Otherwise
the metric is reported as `NULL_LEGACY_BASELINE` and skipped (see
"Optional fields" above).

## When to use

- After an [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)
  run completes (Phase 9 of the upstream runbook).
- Periodic regression sweep - weekly cron diffs each registered
  baseline against the most recent run.
- Pre-promotion gate - block staging or prod promotion if the diff
  verdict is RED.
- Pairing perf vs quality - register the baseline with `--notes
  "GPQA=<score>. MMLU-Pro=<score>"` from the
  [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md) run so
  future diffs can confirm the regression is not masked by a quality
  gain (or vice versa).

Do **not** use this skill for:

- MLPerf training step-time / MFU baselines - use
  [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md) directly
  with `--family <bench>` and a custom schema.
- One-off diagnostic measurements - register a baseline only when the
  measurement is worth remembering.

## Example prompts

- "Register the kimi-k25 perf-bench result as the new baseline."
- "Diff this morning's deepseek-v4 ai-bench against last week's baseline."
- "Inference perf regression check on glm-5-fp8 with tolerance.tps=5%."
- `/inference-perf-baseline-bridge record --model kimi-k25 --source experiments/artifacts/inference-perf-bench/<run-id>/`
- `/inference-perf-baseline-bridge diff --baseline experiments/artifacts/perf-baselines/inference/kimi-k25/2026-05-19T20:00:00Z/ --current experiments/artifacts/inference-perf-bench/<run-id>/`

## Prerequisites

1. **Source bundle path** - `--source <path>` pointing at an
   `experiments/artifacts/inference-perf-bench/<run-id>/` directory.
2. **Model name** - `--model <served-model-name>` (matches the value
   returned by the model's `/v1/models` endpoint).
3. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for the registry path.

## Interaction style

Mostly autonomous. One pause: confirm the per-metric tolerances and
the registry path before write (record) or before computing the diff
verdict (diff).

## Workflow

### Mode 1: record

#### Phase 1: parse the perf-bench bundle into the schema shape

Read the upstream perf-bench output:

- `experiments/artifacts/inference-perf-bench/<run-id>/perf-bench-report-*.md` - parse the per-concurrency rows.
- `experiments/artifacts/inference-perf-bench/<run-id>/aiperf-c<N>.log` - extract the headline metrics per concurrency.
- `experiments/artifacts/inference-perf-bench/<run-id>/<model>-metrics-pre.prom` and `<model>-metrics-post.prom` - extract `vllm:prefix_cache_hit_rate`, `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`, `vllm:avg_generation_throughput_toks_per_s`.

If the operator also captured a paired profile bundle (zymtrace or
Nsight) against the same window, source the two optional fields:

- `experiments/artifacts/zymtrace-bundles/<run-id>/response.json` (or any sibling bundle that follows the [zymtrace-anchored-query](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md) layout) - derive `kernel_class_gpu_pct` from the GPU-side `event_kind = 'cuda'` query response, bucketed per the kernel-class regex appendix in that skill. Derive `cpu_spinpoll_pct` from the CPU-side `event_kind = 'on_cpu'` response by summing samples whose `py_file` matches `shm_broadcast.py` or `utils.py::sched_yield`.
- If no paired profile bundle exists, emit `null` for both optional fields. The diff verb will skip them per the "Optional fields" rule above. The registry record stays valid.

Emit `inference_perfbench_v1.json` to a scratch path, `sha256sum` it.

#### Phase 2: register

```text
mcp__profile_and_optimize__perf_baseline_record with:
  args: ["--family", "inference",
         "--measurement", "<model>-perfbench-v1",
         "--source", "<scratch-path-to-inference_perfbench_v1.json>",
         "--unit", "structured-json",
         "--notes", "<operator-supplied notes; e.g. GPQA + MMLU-Pro scores from inference-model-eval>",
         "--json"]
```

The verb writes the registry entry under
`experiments/artifacts/perf-baselines/inference/<model>-perfbench-v1/<UTC-ts>/`
(per the registry layout the upstream verb already implements).

#### Phase 3: report

Print:

- Registry path.
- Source SHA-256.
- Headline metrics at concurrency 16 (the canonical operating point
  for routine comparison).
- Cross-link to [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md) `diff`
  for future regression checks.

### Mode 2: diff

#### Phase 1: resolve baseline + current

- `Read <baseline>/baseline.json` - confirm `family == "inference"`
  and `measurement == "<model>-perfbench-v1"`.
- `Read <baseline>/source-snapshot.<ext>` - load the baseline's
  `inference_perfbench_v1.json`.
- Parse the current bundle into the same schema shape (Phase 1 of
  record mode).

#### Phase 2: per-metric diff with tolerances

For every concurrency cell in `metrics.<concurrency>`, compute:

- `delta = current - baseline`
- `delta_pct = 100 * delta / baseline` (for percent metrics)
- `delta_abs = current - baseline` (for absolute metrics: cache hit rate)
- Per-metric verdict: GREEN if `|delta|` is within the tolerance for
  that metric (regression direction respected), RED otherwise.

For `server_metrics_delta`, apply the tolerances at the absolute
level: a `prefix_cache_hit_rate_post` drop of more than 2 percentage
points is RED. A peak `gpu_cache_usage_perc` rise of more than 5
percentage points is RED.

#### Phase 3: overall verdict

- GREEN: every metric at every concurrency is within tolerance.
- YELLOW: 1-2 metrics exceed tolerance OR exactly one concurrency
  level shows multiple regressions.
- RED: 3+ metrics exceed tolerance OR the headline metric
  (`throughput_toks_per_s` at concurrency 16) regresses by more than
  its tolerance.

#### Phase 4: write the diff bundle

Delegate to `mcp__profile_and_optimize__perf_baseline_diff` (the existing verb
already writes the diff under
`experiments/artifacts/perf-baseline-diffs/inference/<model>-perfbench-v1/<UTC-ts>/`):

```text
mcp__profile_and_optimize__perf_baseline_diff with:
  args: ["--baseline", "<baseline-registry-entry-dir>",
         "--current", "<scratch-path-to-current-inference_perfbench_v1.json>",
         "--tolerance-percent", "5",
         "--json"]
```

The wrapper supplies `5` as the umbrella tolerance for the underlying
verb's structured-JSON shape. The per-metric verdict above is the
authoritative classification this skill returns.

#### Phase 5: recommend

- GREEN: "no inference perf regression detected. Safe to promote."
- YELLOW: "spot regression on N metrics. Recommend re-running the
  perf-bench against the same dev cluster cohort to rule out noise,
  then re-diff."
- RED:
  - If headline `throughput_toks_per_s` regressed: prompt the
    operator to check for cohort contention via
    [`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md)
    on `vllm:num_requests_running` for the same time window.
  - If TTFT regressed but throughput didn't: suggest checking
    scheduler / queue depth (`vllm:num_requests_running` peak +
    `vllm:gpu_cache_usage_perc` peak deltas).
  - If prefix cache hit rate regressed: suggest the
    `--connection-reuse-strategy sticky-user-sessions` flag - a
    hit-rate drop is often a routing problem (cross-region hops
    breaking session affinity), not a model-side issue.

## Safety

- **Read-only on the registry.** Diff mode never writes to
  `experiments/artifacts/perf-baselines/`. It writes only to
  `experiments/artifacts/perf-baseline-diffs/`.
- **No new MCP verbs.** This skill wraps existing
  `mcp__profile_and_optimize__perf_baseline_record` and
  `mcp__profile_and_optimize__perf_baseline_diff` verbs. The bundled MCP
  surface tool count stays unchanged.
- **Schema mismatch fails fast.** If the baseline's `schema` field
  is not `inference_perfbench_v1`, diff mode stops and asks. No
  cross-schema fuzzy diffing.
- **Provenance preserved.** Every record / diff carries the perf-bench
  bundle's source SHA-256 + operator + cluster + profile_and_optimize SHA, per
  the `server/AGENTS.md` reproducibility-grade-evidence rule.

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
- **Stack:** image/vllm commit, **delivery** (image|overlay|patchedVllm|infr-patch), bench backend, serving engine. A number is evidence only for its own `delivery` -- never cite cross-tier.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named.
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

See `server/AGENTS.md` "Speed-of-light framing". When this bridge
records / diffs a perf-bench run, the baseline record SHOULD carry a
`sol_pct` field per the
[`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md) schema, and
the diff SHOULD report SoL delta alongside the absolute throughput /
latency delta. Peaks are sourced from
`configs/sol-ceilings.yaml` - never inlined.

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

## Source-of-truth references

- The pair this bridge wraps: [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md),
  [`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md).
- Perf-bench input: [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)
  (also reachable via the `ai-bench` colloquial alias).
- Quality counterpart: [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md).
- `server/AGENTS.md` - fail-fast + provenance rules.
