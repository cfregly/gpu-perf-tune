---
name: inference-kernel-ncu-profile
last_validated: 2026-05-29
description: >-
  Capture per-kernel CUDA hardware-counter data (occupancy, achieved warps
  active, regs/thread, smem/block, DRAM throughput, arithmetic intensity,
  branch divergence %, warp-stall reason %) from a live vLLM inference pod
  via an `ncu` debug sidecar (no production image rebuild). Outputs
  `.ncu-rep` + per-kernel CSV. Scoped via `--kernel-name` to avoid the
  10-100x kernel-replay slowdown that an unscoped ncu attach would impose
  on serving. Pairs with `inference-kernel-profile` (nsys: absolute kernel
  duration + cuda-graph timeline) and `zymtrace-anchored-query` (sample-
  share + kernel-name resolution). Triggers on "ncu", "nsight compute",
  "kernel internals", "occupancy", "register pressure", "shared memory",
  "DRAM bandwidth", "roofline", "arithmetic intensity", "warp stalls",
  "branch divergence", or any combination of "ncu / nsight-compute /
  occupancy / regs / smem / dram / roofline" with "vllm / kimi / glm /
  deepseek / inference".
allowed-tools:
  - Bash(kubectl:debug,*)
  - Bash(kubectl:exec,*)
  - Bash(kubectl:cp,*)
  - Bash(kubectl:get,*)
  - Bash(ncu:*)
  - Bash(jq:*)
  - Bash(sha256sum:*)
  - Read
  - Write
---

# inference-kernel-ncu-profile

## Purpose

Capture per-kernel **hardware-counter** data from a **live vLLM
inference pod** without rebuilding the production image. Uses
`kubectl debug --share-processes` to attach an `ncu`-enabled sidecar
to an existing pod. The sidecar replays a small number of launches of
each named kernel with the hardware-counter set enabled and writes
`.ncu-rep` files into a sidecar emptyDir that you extract via
`kubectl cp`.

The three sibling skills cover complementary kernel-profile dimensions:

| Question | Tool | Skill |
|---|---|---|
| What kernels run + relative sample-share? | zymtrace | `zymtrace-anchored-query` |
| Absolute per-kernel duration (ns), CUDA graph timeline, NVTX ranges | nsys | `inference-kernel-profile` |
| **Per-kernel occupancy / regs / smem / DRAM-BW / arithmetic intensity / warp stalls** | **ncu** | **this skill** |
| c=1 decode-step budget (GPU-busy vs host-gap) + kernel-vs-host-vs-comm verdict | vLLM profiler endpoint / nsys | `inference-decode-step-budget` |

**Sequencing note:** for the decode/latency tier, run
[`inference-decode-step-budget`](/plugins/profile-and-optimize/skills/inference-decode-step-budget/SKILL.md)
FIRST. If at c=1 it returns "host-bound" (GPU idle >> busy per step, as GLM-5.1
does), per-kernel SoL is **moot** - the kernels are a small fraction of TPOT and
are launch-latency-scale at one token. Only escalate to this skill when the
budget says kernel-bound, or for the prefill/throughput tier.

## When to use

- Investigating WHY a specific kernel is hot. The `zymtrace-anchored-query`
  + `inference-kernel-profile` skills tell you WHICH kernels are hot and
  HOW LONG they take. Ncu tells you whether they're memory-bound vs
  compute-bound, whether occupancy is the ceiling, whether warps are
  stalling on memory or sync, etc.
- Roofline analysis to prove a kernel is at-the-roof vs has-room.
- Pre-PR-submission justification: when proposing a custom kernel or a
  kernel-tuning change, ncu evidence (e.g. "this kernel achieves only
  47% theoretical occupancy due to register pressure") is the kind of
  argument upstream reviewers expect.

Do **not** use this skill for:

- Continuous always-on capture - kernel replay is 10-100x slow per
  replayed kernel. Ncu is a per-shot tool.
- Untargeted "give me everything" sweeps - without `--kernel-name`
  scoping, ncu will replay EVERY kernel, which on a busy vLLM pod is
  a heavy operation that risks Kubelet liveness-probe failures.
- Multi-pod fleet capture - this skill targets ONE pod at a time.

## Sidecar image

Same image as `inference-kernel-profile`:
**`ghcr.io/cfregly/nsys-sidecar:0.1.0`** (publicly readable).
Image includes nsys 2025.6.3 + **ncu 2026.1.1** + py-spy 0.4.1 +
python3 + jq.

Image is 11.87 GB. First-pull on a new GPU node is ~3-7 min from
ghcr.io. Subsequent attaches on the same node are instant.

## Safety: live-serving impact

ncu's `--set full` capture flag enables every hardware-counter group,
which forces kernel replay (the kernel is launched, ncu reads the
counters, the kernel is launched AGAIN, ncu reads the next group, ...).
Replay multiplies the kernel's wall-clock by 10-100x.

**On a single-replica deploy**, the replay
slowdown can spike pod CPU to the point of Kubelet liveness-probe
failure. Mitigate by:

- Using `--launch-count 3` not 5
- Using `--set basic` instead of `--set full` for the first capture
  (basic captures only the metric groups required for roofline, ~3x
  slowdown instead of 30x)
- Scoping to ONE kernel per capture, not 5

**On a multi-replica deploy**, a temporary per-replica capacity dip
during ncu replay is usually acceptable, so
`--launch-count 5` + `--set full` is safe - the slow replica is just
1 of N and the Service load-balancer distributes around it.

**Fallback**: if even the conservative path is too risky, spin a sister
deploy `<deploy>-ncu` at 0 service routing (a separate Helm release
serving the same model, NOT in the Service selector), and ncu against
THAT pod. Adds ~30 min for the sister helm install, but zero risk to
production serving.

**Memory caveat:** at `gpu-mem-util >= 0.85` there is little
headroom for a profiler's host/device buffers. A long capture window (observed
with a 40s nsys window on a GLM-5.1 canary) can OOM-crash vllm mid-capture and
truncate the report. Keep windows short, scope ncu to one kernel + low
`--launch-count`, and prefer the endpoint-triggered bounded windows in
[`inference-decode-step-budget`](/plugins/profile-and-optimize/skills/inference-decode-step-budget/SKILL.md).

## Gate 0 (precondition): CUPTI must even initialize (CUDA image-vs-driver skew)

ncu is a CUPTI client, so on GB300 nodes it can record **0 kernels** for the same reason nsys + the
vLLM torch profiler do: a CUDA 12.9 serve image against the node's CUDA 13.0 driver
makes CUPTI fail to init (`CUPTI_ERROR_INVALID_DEVICE`). This is NOT permission
(`RmProfilingAdminOnly:0`), NOT a missing lib (`libcupti.so.12` present + linked), and NOT a
wrong-process attach - it is a CUDA major-version toolkit-vs-driver skew. If ncu returns 0 kernels,
grep the launch log for `CUDA versions. CUPTI/Runtime/Driver`: a 12.x-toolkit / 13.x-driver
split needs a CUDA-13-aligned image or zymtrace (non-CUPTI), not more attach tuning.

**SOLVED PATH (GB300 default): pull a Blackwell-capable ncu from NGC** instead of using the
in-image ncu: copy ncu **2026.x** from `nvcr.io/nvidia/pytorch:26.05-py3` (arm64/sbsa) into the
serve container via an initContainer (no prod rebuild. Needs an NGC image-pull secret in the
namespace). Use this NGC-tools path on GB300 for both nsys and ncu,
the legacy in-image / apt paths are superseded there. Verified end-to-end on
a GB300 MoE deploy (per-kernel Compute(SM)/Occupancy resolved).

## ncu capture-hygiene (empty-rep / non-deterministic gates) - empty != tooling limit

Beyond Gate 0 (CUPTI), three ncu traps on a vLLM stack each yield "No kernels were profiled" or a
non-deterministic-count failure. NONE is a tooling limit (each reproduced + fixed on a
DeepSeek-V4-Flash GB300 deploy). Canon: `docs/METHODOLOGY.md` "Capture hygiene".

1. **`cudaProfilerStart` breaks under any vLLM `direct_register_custom_op` import.** If your harness
   imports a module that runs `direct_register_custom_op` at load (`mhc.tilelang`, `fp8_utils`, …),
   ncu `--profile-from-start off` + an in-script `torch.cuda.cudart().cudaProfilerStart()` captures
   **0 kernels** (the custom-op/triton registration perturbs the CUPTI profiler-start hook). FIX: do
   NOT rely on cudaProfilerStart - profile from start and exclude warmup with
   `--launch-skip <warmup> --launch-count N` + `--kernel-name` scoping:
   ```bash
   M=256 N=4096 K=8192 ncu --kernel-name 'regex:sm100_fp8_fp4_gemm' \
     --launch-skip 3 --launch-count 1 --set full --target-processes application-only \
     --export out -- python3 one_op.py     # one_op.py warms the op 3x then loops; ncu skips the 3 warmups
   ```
2. **`-- env VAR=v python3` defeats `--target-processes application-only`** - ncu profiles the `env`
   process, not its python child → 0 kernels. FIX: set the env on the ncu process itself
   (`VAR=v ncu … -- python3 …`), never via an `env` wrapper after `--`.
3. **TP=N MoE+sparse in-engine `--replay-mode application` is non-deterministic.** A large MoE +
   sparse-attention model relaunched per metric-pass varies its kernel-launch count →
   `==ERROR== Unexpected number of profiled kernels`, `--app-replay-mode relaxed` + no-op
   `profile_run` + prefix-caching-off + eager + fixed-blocks do NOT fix it. The canonical L4 path is
   an **isolated single-op kernel-replay harness**: call the production library op directly (e.g.
   `vllm.utils.deep_gemm.fp8_gemm_nt` / `tf32_hc_prenorm_gemm`), TP=1, **NO `mhc.tilelang` import**,
   synthetic fixed-shape inputs (AI/%SoL is shape-structural). Mirror the production caller's quant +
   SF layout (e.g. `per_token_group_quant_fp8` + `per_block_cast_to_fp8` +
   `deepgemm_post_process_fp8_weight_block`) so the exact kernel template fires.

## Two blockers on prod-direct attach

An older `ncu --attach <PID>` recipe attached ncu to a running vllm
process directly. That recipe is **broken** on current production
deploys for two independent reasons. Read this section before
attempting any live capture.

### Blocker 1: zymtrace already owns `CUDA_INJECTION64_PATH`

Production vllm deploys instrumented with zymtrace run with:

```
CUDA_INJECTION64_PATH=/var/lib/zymtrace/profiler/libzymtracecudaprofiler.so
```

This is the zymtrace profiler agent injecting itself as the CUDA
driver-level injection library. The CUDA driver only honors **one**
injection library per process. Ncu cannot stack on top. There's no
way to add ncu's `libcompute.so` to a running zymtrace-instrumented
process.

**Exception: latency/canary deploys often have NO zymtrace
injection.** A canary that runs with no `CUDA_INJECTION64_PATH` in its env
escapes Blocker 1 - ncu launch-mode (or the preload+TCP attach below) works
directly on it without a separate sister-deploy. Check the target's
env (`kubectl get deploy <name> -o jsonpath='{...env...}'`) before assuming a
sister is required. The sister-deploy is only mandatory for the
zymtrace-instrumented production releases.

### Blocker 2: ncu 2026.1.1 dropped `--attach <PID>`

The deprecated `--attach <PID>` flag was removed in ncu 2026.x:

```
==ERROR== unrecognised option '--attach'. Use --help for further details.
```

The modern ncu interaction modes (per `ncu --help`) are:

- `--mode launch-and-attach` (default) - wrap the application at
  process startup
- `--mode launch` - launch the application and suspend for a later
  attach
- `--mode attach` - connect to a process pre-instrumented with
  ncu's injection library, via TCP (`--hostname` + `--port`)

The only path that works against an **already-running** vllm process
is `--mode attach`, and it requires `NV_NCU_INJECTION64_PATH` to have
been set when vllm started so the process exposes the ncu profiling
endpoint. Production vllm (instrumented with zymtrace instead) does
not expose this endpoint.

## Recipe: sister-deploy with preload + TCP attach

Both blockers are dodged by spinning up a parallel **sister-deploy**
helm release running the same vllm image but with:

- zymtrace `CUDA_INJECTION64_PATH` unset
- `NV_NCU_INJECTION64_PATH` pointing at ncu's `libcompute.so`
  (mounted from the nsys-sidecar image via an initContainer + emptyDir
  volume)
- `replicas: 1`
- a Service selector / label that **excludes** prod traffic (the
  prod Service matches `app: basic-inference`, the sister uses
  `app: basic-inference-ncu`, so prod LB routes around it)

The vllm process on the sister pod exposes ncu's TCP profiling
endpoint on `127.0.0.1:49152`. From a `kubectl debug` sidecar with
`--share-processes`, run:

```bash
ncu --mode attach --hostname 127.0.0.1 --port 49152 \
    --kernel-name '<KERNEL_REGEX>' \
    --launch-count 5 \
    --set full \
    --target-processes all \
    --export /profiling/ncu-$K.ncu-rep
```

Multiple captures against the same long-lived sister pod are
supported - vllm stays warm. The sidecar runs ncu N times sequentially.

### Reference helm + capture scaffolding

A reusable sister-deploy scaffold (one directory per model) contains:

- `values-ncu-sister.yaml` - helm values overrides
- `install.sh` - `helm upgrade --install` wrapper
- `capture.sh` - full capture flow (wait-ready, warmup, sidecar
  attach, N-kernel loop, extract, helm uninstall). Flags:
  `--launch-count N`, `--set basic|full`, `--roofline-min` (AI-roofline
  `--metrics` set only), `--replay-mode kernel|application|range`, and
  `--campaign <slug> --cell-id <cell>` to auto-import the result into a
  perf-report campaign cell.
- `README.md` - runbook + canonical 5-kernel list
- a replay-mode-application runbook - the ack-gated path past the
  TP=8 NVFP4 `ContextSaveFailed` kernel-replay blocker (use
  `--replay-mode application --roofline-min`).

### Identifying hot kernels

ncu `--kernel-name` accepts substring or regex match against the
demangled or mangled kernel name. Demangle a mangled symbol via
`c++filt` (available in the nsys-sidecar image):

```bash
echo "_ZN66_GLOBAL__N__51f43292_25_CUDASymmetricMemoryOps_cu_c81adf72_277485226multimem_all_reduce_kernelIN3c108BFloat16ELi16EEEvPT_mPPjmm" | c++filt
# -> (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(...)
```

Either form is fine for `--kernel-name`:

```bash
--kernel-name multimem_all_reduce_kernel
--kernel-name "regex:multimem_all_reduce.*BFloat16"
```

Canonical 5-kernel set for B200 NVFP4 inference (Kimi K2.6, GLM-5.1,
DeepSeek):

| Kernel substring | Category | Expected bottleneck |
| --- | --- | --- |
| `multimem_all_reduce_kernel` | NCCL | NVLink5 bandwidth |
| `bmm_E2m1` | BMM-NVFP4 | NVFP4 Tensor Core compute |
| `fmhaSm100fKernel` | FMHA | HBM3e bandwidth |
| `nvjet_tst_.*_2cta` | MoE / GEMM | NVFP4 Tensor Core compute |
| `triton_red_fused` | Triton-fused | Mixed |

## Recipe (deprecated - DO NOT USE on prod-direct)

An earlier recipe used `ncu --attach $VLLM_PID` against the
zymtrace-instrumented production pod. That recipe is preserved below
ONLY as historical reference. It WILL FAIL on every current
inference deploy. Use the sister-deploy section above.

<details>
<summary>Historical recipe (broken. Click to expand)</summary>

```bash
# THIS WILL FAIL:
# ERROR: unrecognised option '--attach'  (ncu 2026.1.1)
# ERROR: CUDA injection slot taken by zymtrace
ncu --kernel-name '$KERNEL' --launch-count 5 --set full \
    --export /profiling/ncu-$KERNEL.ncu-rep \
    --attach $VLLM_PID
```

</details>

### 5. Export human-readable CSV

```bash
kubectl -n $NS exec $TARGET_POD -c $SIDECAR -- bash -c "
  ncu --import /profiling/ncu-${KERNEL}.ncu-rep \
    --csv --page raw > /profiling/ncu-${KERNEL}-raw.csv
  ncu --import /profiling/ncu-${KERNEL}.ncu-rep \
    --csv --page details --section SpeedOfLight \
    > /profiling/ncu-${KERNEL}-sol.csv
"
```

Key columns from `--page details --section SpeedOfLight`:

| Column | Meaning |
|---|---|
| `Memory Throughput [%]` | DRAM throughput vs theoretical peak |
| `DRAM Throughput [%]` | Same, restricted to DRAM (vs L2) |
| `Compute (SM) Throughput [%]` | SM utilization |
| `Achieved Occupancy` | Actual warps-active / theoretical-max-warps-active |
| `Theoretical Occupancy` | What the kernel COULD achieve given regs+smem |
| `Achieved Active Warps Per SM` | Average warps active per SM |
| `Block Limit Registers` | Limit imposed by registers/thread |
| `Block Limit Shared Mem` | Limit imposed by smem/block |
| `Block Limit Warps` | Limit imposed by max warps per SM |

These are the roofline + occupancy diagnostics. A kernel with
`DRAM Throughput % = 92%` is memory-bound. One with
`Compute SM Throughput % = 88%` is compute-bound. One with
`Achieved Occupancy = 0.31` and `Block Limit Registers = limited` is
suffering register pressure.

### 6. Extract artifacts

```bash
BUNDLE=experiments/artifacts/inference-perf-bench/<bundle>/
mkdir -p $BUNDLE/ncu-profiles
for kernel in multimem_all_reduce_kernel flashinfer_trtllm_allreduce_fusion fmhaSm100fKernel concat_and_cache_mla_kernel bmm_E2m1; do
  kubectl -n $NS cp $TARGET_POD:/profiling/ncu-${kernel}.ncu-rep $BUNDLE/ncu-profiles/ -c $SIDECAR
  kubectl -n $NS cp $TARGET_POD:/profiling/ncu-${kernel}-sol.csv $BUNDLE/ncu-profiles/ -c $SIDECAR
done
```

### 7. Populate `kernel_internals` field in `inference_perfbench_v1.json`

```python
import json, csv, pathlib
bundle = pathlib.Path("$BUNDLE")
ipb = json.loads((bundle / "inference_perfbench_v1.json").read_text())
ipb["kernel_internals"] = {
    "captured_at": "<UTC>",
    "method": "ncu-sidecar-kernel-scoped",
    "sidecar_image": "ghcr.io/cfregly/nsys-sidecar:0.1.0",
    "vllm_pid": $VLLM_PID,
    "per_kernel": {
        # one entry per kernel captured:
        # "<kernel-name>": {
        #     "achieved_occupancy_pct": ...,
        #     "theoretical_occupancy_pct": ...,
        #     "dram_throughput_pct_peak": ...,
        #     "sm_throughput_pct_peak": ...,
        #     "regs_per_thread": ...,
        #     "smem_per_block_bytes": ...,
        #     "block_limit_factor": "registers" | "shared_mem" | "warps",
        # }
    },
}
(bundle / "inference_perfbench_v1.json").write_text(json.dumps(ipb, indent=2))
```

The `inference_perfbench_v1` schema treats `kernel_internals` as
optional - bundles without it are unaffected.

### 8. Cleanup

```bash
kubectl -n $NS exec $TARGET_POD -c $SIDECAR -- rm -rf /profiling/ncu-*
# The sidecar container will terminate after the 3600s sleep; pod
# auto-cleans the ephemeral container.
```

## Cross-skill join

Roofline interpretation is most useful when joined with the absolute
kernel-time from `inference-kernel-profile` (nsys) and the relative
sample-share from `zymtrace-anchored-query`. Example join:

```
Kernel                                  zymtrace samples %  nsys total ms  ncu DRAM %  ncu SM %  Verdict
flashinfer trtllm_allreduce_fusion      14.2%               1,847          88%          12%      Memory-bound; comm-bound (good - kernel is at-the-roof)
fmhaSm100fKernel...Persistent           8.7%                1,144          47%          81%      Compute-bound (room for compute-side tuning)
multimem_all_reduce_kernel              5.3%                694            91%           9%      Memory-bound; comm-bound (at-the-roof)
bmm_E2m1_E2m1E2m1                       6.9%                901            61%          74%      Mixed (canonical NVFP4 BMM: balanced)
```

Without ncu, you have shares (zymtrace) + times (nsys) but no
attribution to the dimension (memory vs compute vs occupancy). ncu
closes that loop.

## Cost + risk

- ~10-100x per-replayed-kernel slowdown during capture. Bounded by
  `--launch-count` (5 replays at 100x = 0.5s of kernel time = sub-second
  for typical kernels)
- ncu-rep files are typically 5-50 MB per kernel (manageable)
- Sidecar attach is non-disruptive (no vllm restart)
- **Live-pod risk**: see Safety section above

Pair this skill with `evidence-bundle-init` when starting a new
investigation so the `ncu-profiles/` subdir lands in the right place.

## Full-context reporting (no bare numbers)

This skill is the preferred follow-up for the proper roofline scatter. Per the canon
"Every performance number carries its full context (no bare numbers)"
(`docs/METHODOLOGY.md` "Full-context reporting"): every number this skill emits MUST carry its full
measurement-context descriptor, and every comparison MUST be matched on it. A bare number is a
defect - it cannot set a default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named.
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

This skill is the **preferred path** to a real per-kernel
arithmetic-intensity-vs-roofline scatter plot per
`docs/METHODOLOGY.md` "Speed-of-light framing". The other two siblings
(`zymtrace-anchored-query`, `inference-kernel-profile`) deliver
time-share + wall-clock duration but cannot compute %SoL per kernel
because they lack the FLOPS + bytes counters this skill captures
(`sm__sass_thread_inst_executed_op_*_pred_on.sum`, `dram__bytes.sum`).

Workflow when proper %SoL is needed:

1. Identify the dominant hot kernel via
   [`zymtrace-anchored-query`](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md) +
   [`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md).
2. Run THIS skill against that kernel name with the roofline-min
   counter set (see Reference below).
3. Compute arithmetic intensity = FLOPS / bytes. Place the point on
   the B200 / GB300 / H100 roofline using peaks from
   `configs/sol-ceilings.yaml` (`b200_sm100.nvfp4_dense_pflops`
   for compute ceiling, `b200_sm100.hbm3e_tbps` for bandwidth ceiling).
   Cite by key path, never inline magic numbers.
4. Record the resulting %SoL in the per-run `sol-summary.md` doc for
   the campaign / cluster probe this kernel came from.

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per `docs/METHODOLOGY.md` "Always be grinding (next-lever framing)". A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.

## Wiring ncu into the perf-report page-5 roofline scatter

The capture's `ncu-profiles/*-sol.csv` + `*-raw.csv` are ingested into a
perf-report campaign cell as `cells/<cell-id>/ncu_kernels.json` (the page-5
input) via the `import_ncu` verb:

```bash
perftunereport import_ncu --campaign <slug> --cell-id <cell-id> --bundle <bundle-dir>
perftunereport report_render --campaign <slug>   # -> page 5 populated
```

(`capture.sh --campaign <slug> --cell-id <cell>` runs this automatically at
the end.) Notes:

- **Two CSV shapes are auto-handled.** ncu's `--page raw` is wide (one row per
  kernel, metric columns). Ncu-2026's `--page details --section SpeedOfLight`
  is long/melted (`Metric Name` / `Metric Value` rows). The importer detects
  and pivots the long shape, so either export works.
- **Counter set determines what page 5 can show.** `--set=full` or
  `--roofline-min` collect `dram__bytes.sum` + the
  `sm__sass_thread_inst_executed_op_*_pred_on.sum` FLOP op-counts, so the
  importer computes arithmetic intensity + achieved TFLOPS and page 5 plots a
  real roofline point. `--set=basic` collects neither, so AI is null and page
  5 renders a hollow "%SoL only / AI unmeasured" marker placed at SM% x the
  kernel family's compute ceiling (`category_ceiling_map` in
  `configs/sol-ceilings.yaml`) - the y is honest, the x is a
  placeholder. AI is never fabricated.
- **SM-busy% is NOT %-of-FLOP-SoL - never conclude "at-roof" from `--set basic`.** The
  `--set basic` `sm__throughput.avg.pct_of_peak` ("SM busy") can read 88-92% while the kernel is at
  <15% of its FLOP-roofline - a persistent / split-K / spinning kernel keeps SMs busy doing little
  tensor work. To decide at-roof vs headroom you MUST run `--set full` and compare achieved TFLOPS
  to the FLOP ceiling (`fp8_dense_pflops` / `nvfp4_dense_pflops` / `bf16_dense_pflops`). WORKED
  EXAMPLE: the DeepSeek-V4-Flash FP8 GEMM (`sm100_fp8_fp4_gemm_1d1d`, ~70% of decode compute) read
  88-92% SM (`--set basic`, which mislabeled it "at-roof") but `--set full` showed only **1-14.5% of
  the FP8 FLOP-SoL** (M=1→256) - the kernel is concurrency-starved, not exhausted. The throughput
  tier was the real headroom, not a kernel rewrite.
- **TP=8 NVFP4 blocker.** Multi-pass kernel-replay fails at TP=8 NVFP4
  (`ContextSaveFailed`). See the replay-mode-application runbook in the
  sister-deploy scaffolding above for the `--replay-mode application` path that
  gets to a full AI-grounded point.

## Kernel rubric (K/R/H/P/A) - this skill is the H + P enforcement point

When this capture backs a **custom-kernel** comparison (a candidate kernel vs a
baseline), apply the kernel rubric (see `docs/METHODOLOGY.md`
"Kernel-work classification"). ncu is the **only** profiler in the
sibling set that can prove the two axes a kernel win actually turns on:

- **H (hardware specialization)** - the `Compute (SM) Throughput [%]` and tensor-pipe
  active % (from the SpeedOfLight section) reveal whether a kernel engages the
  frontier path. On SM100 the production libraries are H4 (`sm100f` tensor cores /
  NVFP4 tensor cores via `bmm_E2m1`, `nvjet_tst_*` in the canonical 5-kernel set). A
  candidate kernel that shows **near-zero tensor-pipe activity / SM% dominated by FMA**
  is H1 - and an H1 candidate cannot beat an H3-H4 baseline on a K3-K4 op no matter
  how it schedules. Read tensor-core engagement off ncu BEFORE believing any win.
- **P (performance target)** - the roofline (%SoL via the FLOPS + `dram__bytes.sum`
  counters, "Speed-of-light reporting" above) is the P-axis: P4 means at/above the
  best library. A kernel far below its family ceiling that still "wins" an e2e A/B is
  winning on something other than GPU work (re-check methodology).

**The gate, operationally:** record `(K,R,H,P,A)` for the candidate AND the named
baseline in the bundle's `SOURCE.md`/`summary.md`. A speedup over a baseline at
strictly lower H or R (e.g. a tensor-core candidate vs a generic-Triton baseline when
production runs the `sm100f` library) is a **DRAFT, never a VERDICT** - it fails the
"Fair baseline" clause. The canonical worked failure: warp-decode (K4/R2/H1) beat
generic Triton in microbench but ncu/zymtrace confirmed it never engaged tensor cores,
and it lost 1.51x to FlashInfer-TRTLLM (K4/R1/H4) on real GPU time.

**Emit it as a structured `krhpa:` block (not just prose).** When the campaign
renders an L4 ncu roofline (page 5, `sol_rigor=L4`), `perftunereport publish_to_lake`
FAILS CLOSED unless the campaign `config.yaml` carries a `krhpa:` block
classifying both arms - prose in `SOURCE.md`/`summary.md` alone does not satisfy
the gate for an L4 campaign. Add to the campaign config:

```yaml
krhpa:
  candidate: {K: 4, R: 2, H: 1, P: 2, A: 1, name: "warp-decode (Triton FMA)"}
  baseline:  {K: 4, R: 1, H: 4, P: 4, A: 1, name: "FlashInfer-TRTLLM bmm_*_sm100f"}
```

Each axis is an int `1..4` (= `L1..L4`), `baseline.name` must name the
production-representative kernel/library. The gate (`lake_writer.krhpa_problems`)
refuses a missing or malformed block under `--strict` and records + warns otherwise.

## Reference: full hardware-counter set names

ncu's `--set full` enables ~60 metric groups. For roofline analysis the
minimum set is:

```
sm__sass_thread_inst_executed_op_*_pred_on.sum  (per-op work counts)
dram__bytes.sum                                  (DRAM bytes read+written)
gpc__cycles_elapsed.max                          (peak cycles)
smsp__warps_active.avg.pct_of_peak_sustained_active  (occupancy)
launch__registers_per_thread                     (regs/thread)
launch__shared_mem_per_block_static              (smem/block)
```

These are also available under the named section group
`--section SpeedOfLight_HierarchicalTensorRooflineChart` for direct
roofline plotting.
