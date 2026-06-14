---
name: inference-kernel-profile
last_validated: 2026-05-29
description: >-
  Capture per-kernel CUDA profile data from a live vLLM inference pod via an
  nsys debug sidecar (no production image rebuild). Outputs `.nsys-rep` +
  summary CSV + top-kernels table that joins with the zymtrace
  per-kernel breakdown and the inference-perf-bench
  bundle's `inference_perfbench_v1.json`. Triggers on "kernel profile",
  "nsys profile", "what's in the native bucket", "per-kernel breakdown",
  "ncu", "Nsight Systems", "kernel-level analysis", or any combination of
  "profile / capture / nsys / ncu" with "vllm / kimi / glm / deepseek".
allowed-tools:
  - Bash(kubectl:debug,*)
  - Bash(kubectl:exec,*)
  - Bash(kubectl:cp,*)
  - Bash(kubectl:get,*)
  - Bash(nsys:*)
  - Bash(jq:*)
  - Bash(sha256sum:*)
  - Read
  - Write
---

# inference-kernel-profile

## Purpose

Capture per-kernel CUDA **timeline + absolute-duration** profile data
from a **live vLLM inference pod** without rebuilding the production
image. Uses `kubectl debug --share-processes` to attach an nsys-enabled
sidecar to an existing pod. The sidecar profiles the running vllm
engine PID and writes `.nsys-rep` + `summary.csv` into a sidecar
emptyDir that the operator extracts via `kubectl cp`.

## What nsys is for vs the sibling skills

| Question | Tool | Skill |
|---|---|---|
| What kernels run + relative sample-share? | zymtrace | `zymtrace-anchored-query` (free, no sidecar - kernel names resolve fully when the ClickHouse join is correct) |
| **Absolute per-kernel duration (ns), CUDA graph capture/replay timeline, NVTX ranges** | **nsys** | **this skill** |
| Per-kernel occupancy / regs / smem / DRAM-BW / arithmetic intensity / warp stalls | ncu | `inference-kernel-ncu-profile` |
| Where does c=1 (low-concurrency) decode TIME go (GPU-busy vs host-gap vs comm) + kernel-vs-host-vs-comm verdict | vLLM profiler endpoint (torch) or nsys | `inference-decode-step-budget` |

**Fast path for the decode/latency tier (prefer over a restart):** if the
question is "where does c=1 decode time go", use
[`inference-decode-step-budget`](/plugins/profile-and-optimize/skills/inference-decode-step-budget/SKILL.md) - it
drives vLLM's `/start_profile` + `/stop_profile` HTTP endpoints so you capture
restart-free in seconds, and it bakes in the correctness gates (clean
single-stream driver. GPU-busy must include CUDA-graph execution. Reconcile
against driver TPOT). This skill (nsys sidecar/timeline) is for prefill /
throughput-tier hot-spots and the deep CUDA-graph timeline.

**Note on zymtrace symbol resolution**: zymtrace DOES resolve
SASS-level kernel symbols - the per-kernel join returns fully-resolved
kernel names directly from zymtrace ClickHouse via `interp_funcs`. A
high "native unresolved" share usually means the query joined wrong
(filtered out the kernel-name match), not a tooling gap. Use
`zymtrace-anchored-query` first for "what kernels run" - it's free + no
cluster mutation. Use nsys for the things zymtrace CAN'T give you (timeline,
absolute ns, NVTX, cuda graph events).

## When to use

- Investigating a specific bottleneck where the operator already ran a
  perf-bench sweep + zymtrace and wants per-kernel detail beyond the
  category breakdown.
- A/B'ing two helm configs to see which kernels shifted.
- One-shot release-readiness check before a major deploy change.

Do **not** use this skill for:

- Continuous always-on capture - that's what zymtrace is for. nsys is
  on-demand only because it adds ~5% per-pod overhead during capture.
- Multi-pod fleet-level profiling - this skill targets ONE pod at a time.
  For fleet capture use the zymtrace profiler DaemonSet.

## Sidecar image

Two paths:

- **Path A (recommended)**: pre-built
  `ghcr.io/cfregly/nsys-sidecar:0.1.0` (publicly
  readable. Manifest digest
  `sha256:3146de96f6022a8cc36f86d1b8c0281cb940e51e2c3dc49c315646ad66ede43d`).
- **Path B (zero-build)**: `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu22.04`
  with `apt install nsight-systems-cli` at attach time

## Recipe

### 1. Identify the target vllm pod

```bash
NS=inference
TARGET_POD=$(kubectl -n $NS get pods -l app=basic-inference -o jsonpath='{.items[0].metadata.name}')
echo "Target pod: $TARGET_POD"
```

### 2. Attach the sidecar (one-shot, non-disruptive)

```bash
SIDECAR=nsys-debug-$(date +%s)
kubectl -n $NS debug $TARGET_POD \
  --image=ghcr.io/cfregly/nsys-sidecar:0.1.0 \
  --container=$SIDECAR \
  --share-processes --target=basic-inference \
  -- sleep 3600
```

`--share-processes` is the critical flag: it puts the sidecar in the
target pod's PID namespace so `nsys profile --attach-pid <vllm-pid>` can
see the engine process.

### 3. Capture (120s default duration)

```bash
VLLM_PID=$(kubectl -n $NS exec $TARGET_POD -c $SIDECAR -- \
  pgrep -f "vllm serve" | head -1)

kubectl -n $NS exec $TARGET_POD -c $SIDECAR -- bash -c "
  nsys profile \
    --output=/profiling/capture \
    --capture-range=cudaProfilerApi \
    --duration=120 \
    --force-overwrite=true \
    --attach-pid=$VLLM_PID \
    --sample=cpu --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --sampling-frequency=1000
"
```

**Gotcha 1 - CUDA graphs (must-read).** vLLM runs decode (and often prefill)
inside CUDA graphs. Without `--cuda-graph-trace=node` the graph executes as one
opaque `GRAPH_TRACE` span and the constituent kernels do NOT appear in
`cuda_gpu_kern_sum` - so a KERNEL-only "GPU-busy" sum **under-counts** real busy
time and a kernel ranking misses the in-graph forward kernels. Always pass
`--cuda-graph-trace=node`, and when computing GPU-busy from the sqlite, union
`CUPTI_ACTIVITY_KIND_KERNEL` with `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` (+ `MEMCPY`).
Detect graphs via the `cudaGraphLaunch` vs `cudaLaunchKernel` API counts.

**Gotcha 2 - `--capture-range=cudaProfilerApi` needs a trigger.** It only starts
collecting when the app calls `cudaProfilerStart`. vLLM does that **only** when
launched with `--profiler-config.profiler=cuda` and you POST `/start_profile`.
Without that, either set `--profiler-config.profiler=cuda` (preferred - precise
window, no restart per capture) or drop `--capture-range` and use a timed window
(`--delay`/`--duration`) - but a fixed window is brittle (it can miss
steady-state, and a long window can OOM at gpu-mem-util >= 0.85, truncating the
report). See [`inference-decode-step-budget`](/plugins/profile-and-optimize/skills/inference-decode-step-budget/SKILL.md)
for the endpoint-triggered recipe.

**Gotcha 3 - `--attach-pid` caveats.** Attaching to a running PID for CUDA
tracing works only when the process has no conflicting CUDA injection
(production zymtrace-instrumented pods set `CUDA_INJECTION64_PATH` and block it -
see the ncu skill's "Blocker 1". Latency canaries usually have none). For the
*decode* tier prefer the profiler-endpoint path over attach.

**Capture-quality gate (decode captures).** Before trusting a decode trace,
verify it (a) contains CUDA-kernel data, (b) shows a repeating decode-step
pattern (a dominant inter-step idle-gap bucket), and (c) is not prefill-
contaminated - drive a CLEAN single-stream workload (tiny prompt + long gen +
`ignore_eos`), not `vllm bench serve --random-input-len N`.

**Gate 0 (precondition) - CUPTI must even initialize (CUDA image-vs-driver skew).** On GB300,
a CUDA 12.9 serving image against the node's CUDA 13.0 driver makes CUPTI fail to init
(`CUPTI_ERROR_INVALID_DEVICE`), so EVERY CUPTI client (nsys, ncu, vLLM `--profiler-config.profiler=torch`)
records **0 kernels REGARDLESS of the four gates below**. This is NOT capture hygiene and the gates won't
fix it - it is a CUDA major-version toolkit-vs-driver skew (NOT permission: `RmProfilingAdminOnly:0`. NOT
missing-lib: `libcupti.so.12` present + linked. NOT wrong-process). If you get 0 kernels on GB300,
grep the kineto/nsys log for `CUDA versions. CUPTI/Runtime/Driver` FIRST: a 12.x-toolkit / 13.x-driver
split needs a CUDA-13-aligned image or zymtrace (non-CUPTI), not more capture tuning.
**SOLVED PATH (GB300 default):** use launch-wrap templates that copy a Blackwell-capable nsys
2026.x (CUDA-13) from an NGC CUDA image into the serve container - the model-bringup template
ships `nsys-ngc.yaml` for vLLM and `nsys-sglang.yaml` for SGLang
(`sglang.launch_server`). These are the GB300 default. The in-image / apt nsys is superseded there.

**An EMPTY `cuda_gpu_kern_sum` is a CAPTURE-HYGIENE bug, NOT a "cudagraph blind
spot" - DO NOT conclude the stack is unprofilable until you validate four
things** (only after Gate 0 passes). A single empty rep has been wrongly declared a
"CUDA-graph blind spot" before. A re-capture on the same production cudagraph stack with
fixed capture hygiene returned **tens of millions of kernel rows**:
1. **Flag** - `--cuda-graph-trace=node` is in the nsys argv (Gotcha 1 above),
   without it graph-resident kernels are opaque `GRAPH_TRACE` and `kern_sum` is
   empty at c>=64.
2. **Traffic** - a bench is DRIVING load during the capture window. At the
   **throughput tier (c>=64)** drive continuous `--max-concurrency >=64` traffic
   during `[delay, delay+duration]`. An idle/untrafficked window yields an empty
   rep. (At the decode/c=1 tier this is the CLEAN single-stream workload above.)
3. **Rep-size** - a real c>=64 rep is hundreds of MB to GB (421 MB @ c=128,
   1.0 GB @ c=192). `.nsys-rep << ~10 MB` => idle/failed capture, RETRY. Never run
   stats on a tiny rep.
4. **Sqlite probe** - `nsys export --type sqlite` then
   `SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL`. A `kern_sum SKIPPED` message
   ALONE is not proof. It fires on a near-empty rep. Only after all four hold and
   it is STILL empty may you escalate to a genuine tooling limit.
Mechanical gate: `scripts/nsys-validate-capture.sh`
(checks the flag, rep-size, and KERNEL row count. PASS/RETRY).

**Two launch-wrap gotchas that also hit nsys (canon: the ncu capture-hygiene section of
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md) +
`docs/METHODOLOGY.md`):** (1) `-- env VAR=v python3` makes the profiler
target the `env` process not its python child - set the env on the profiler process itself
(`VAR=v nsys … -- python3 …`), (2) profiler-start-based scoping is unreliable when the harness
imports a vLLM `direct_register_custom_op` module - prefer launch-index/delay scoping.

### 4. Extract artifacts

```bash
BUNDLE=experiments/artifacts/inference-perf-bench/<bundle>/
mkdir -p $BUNDLE/profiles
kubectl -n $NS cp $TARGET_POD:/profiling/capture.nsys-rep \
  $BUNDLE/profiles/capture.nsys-rep -c $SIDECAR
kubectl -n $NS exec $TARGET_POD -c $SIDECAR -- bash -c "
  nsys stats --report cuda_api_sum,gpu_kern_sum,osrt_sum \
    /profiling/capture.nsys-rep --format=csv \
    --output=/profiling/summary
"
kubectl -n $NS cp $TARGET_POD:/profiling/summary_gpu_kern_sum.csv \
  $BUNDLE/profiles/gpu_kern_sum.csv -c $SIDECAR
```

### 5. Populate `kernel_profile` field in `inference_perfbench_v1.json`

```python
import json, pathlib
bundle = pathlib.Path("$BUNDLE")
ipb = json.loads((bundle / "inference_perfbench_v1.json").read_text())
ipb["kernel_profile"] = {
    "captured_at": "<UTC>",
    "nsys_rep_path": "profiles/capture.nsys-rep",
    "summary_csv_path": "profiles/gpu_kern_sum.csv",
    "duration_s": 120,
    "vllm_pid": $VLLM_PID,
    "sidecar_image": "ghcr.io/cfregly/nsys-sidecar:0.1.0",
    "method": "kubectl-debug-share-processes",
    # populate top_kernels from the CSV's "name" + "samples" columns
}
(bundle / "inference_perfbench_v1.json").write_text(json.dumps(ipb, indent=2))
```

The `inference_perfbench_v1` schema treats `kernel_profile` as an
optional dict - bundles without it are unaffected.

### 6. Cleanup

```bash
kubectl -n $NS exec $TARGET_POD -c $SIDECAR -- rm -rf /profiling/*
# The sidecar container will terminate after the 3600s sleep; pod
# auto-cleans the ephemeral container.
```

## Output schema

The capture produces three files under `<bundle>/profiles/`:

| File | Format | Use |
|---|---|---|
| `capture.nsys-rep` | binary nsys | open in Nsight Systems GUI for full timeline |
| `gpu_kern_sum.csv` | CSV | per-kernel total time / samples / instances |
| `cuda_api_sum.csv` | CSV | per-CUDA-API total time (for host-side overhead diagnosis) |

The CSV columns from `nsys stats --report gpu_kern_sum`:

```
Time (%), Total Time (ns), Instances, Avg (ns), Med (ns), Min (ns), Max (ns), StdDev (ns), Name
```

The renderer + analyze-zymtrace-workload skill (Phase 4d) join this CSV
with the zymtrace per-category breakdown via the `Name` column's
canonical-substring matching.

## Cross-skill join

The `analyze-zymtrace-workload` skill (Phase 4d) reads the
`kernel_profile` field when present and joins:

- zymtrace categorical share: e.g. "FMHA = 14.2%"
- nsys per-kernel share: e.g. "fmha_v2_kernel_<sm100>: 11.8%,
  fmha_v2_kernel_paged<sm100>: 2.4%"

Showing the operator both numbers at once closes the loop on "is the
hot category one kernel or three?" - answerable from zymtrace alone is
"unknown beyond the category".

## Cost + risk

- ~5% per-pod overhead during the 120s capture window
- nsys-rep files are typically 50-200 MB (manageable)
- Sidecar attach is non-disruptive (no vllm restart)

Pair this skill with `evidence-bundle-init` when starting a new
investigation so the `profiles/` subdir lands in the right place from
the start.

## Pairs with

- [`inference-decode-step-budget`](/plugins/profile-and-optimize/skills/inference-decode-step-budget/SKILL.md) - FAST, restart-free c=1/low-concurrency decode-step budget via vLLM's `/start_profile` endpoint. Use it (not this skill) when the question is "where does decode time go / is it kernel- or host-bound".
- [`zymtrace-anchored-query`](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md) - kernel-name resolution + sample-share from ClickHouse, no cluster mutation. Run FIRST. An empty result right after a bench is usually ClickHouse ingest lag (async flush), not absence - wait + requery for the freshest data (see [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md)).
- [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md) - per-kernel hardware counters (occupancy, regs, smem, DRAM BW, roofline) using the SAME sidecar image, scoped via `--kernel-name` to avoid the 10-100x kernel-replay slowdown. Run AFTER nsys to interpret hot-kernel ROOT CAUSE (memory-bound vs compute-bound vs occupancy-limited).

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

Per `docs/METHODOLOGY.md` "Speed-of-light framing", kernel-time
hot-spot interpretation MUST cite the natural ceiling each kernel
family is bound by, sourced from
`configs/sol-ceilings.yaml`:

- NCCL kernels → `nvlink5_tbps` (1.8 TB/s bi-dir per GPU on B200)
- MoE / BMM-NVFP4 → `nvfp4_dense_pflops` (9 PFLOPS dense on B200)
- FMHA → `hbm3e_tbps` (8 TB/s on B200)
- cuBLAS / Triton-fused → `bf16_dense_pflops` (2.25 PFLOPS on B200)

nsys gpu_kern_sum gives wall-clock time per kernel. The %SoL conversion
needs FLOPS / bytes per kernel which **ncu** captures, not nsys. When
a nsys finding flags a hot kernel that needs proper %SoL interpretation,
the next step is
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)
on that kernel name - see the "Pairs with" entry above.

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

## Kernel rubric (K/R/H/P/A)

When this nsys capture backs a **custom-kernel** comparison, apply the
kernel rubric (`docs/METHODOLOGY.md` "Kernel-work classification").
**Record `(K,R,H,P,A)` for the candidate AND the named baseline** in the bundle's
`SOURCE.md`/`summary.md`. nsys gives you the K and R axes directly (which op, library
vs Triton vs CUDA-graph-captured) and the absolute per-kernel duration that feeds the
P-axis comparison - but it **cannot** prove the H axis (tensor-core engagement) or a
true %SoL. Defer the H + P proof to
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md), which is
the gate's enforcement point. Reminder: a win over a strictly-lower-H/R baseline (e.g.
beating generic Triton when production runs the `sm100f` tensor-core library) is a
**DRAFT, never a VERDICT** - it fails the "Fair baseline" clause. When the campaign
reaches L4 (an ncu roofline renders), the structured `krhpa:` block in `config.yaml`
is **required** by `publish_to_lake` (see the YAML example in
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)). Prose in
`SOURCE.md`/`summary.md` alone no longer satisfies the gate.
