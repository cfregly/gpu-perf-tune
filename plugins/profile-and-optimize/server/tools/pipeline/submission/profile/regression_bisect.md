Status: Active
Audience: operators triaging an MFU / step-time / convergence regression on a GPU training fleet.

# Performance regression bisection decision tree

This is the symptom-to-culprit decision tree that pairs with
[`profile_run.sh`](/plugins/profile-and-optimize/server/tools/pipeline/submission/profile/profile_run.sh), [`profile_diff.py`](/plugins/profile-and-optimize/server/tools/pipeline/submission/profile/profile_diff.py),
and [`host_overhead.py`](/plugins/profile-and-optimize/server/tools/pipeline/submission/profile/host_overhead.py). It turns a noisy "step time
got slower" complaint into a short list of likely culprits, each anchored
on a concrete recent NVIDIA-side commit so the operator knows what kind
of fix to look for. Pair it with the strategy memo at
[`docs/profiling-and-perf-discovery.md`](/plugins/profile-and-optimize/server/docs/profiling-and-perf-discovery.md)
and your per-target SOPs under `runbooks/`.

The tree is **not** an automated tool. Each branch is a hypothesis the
operator confirms with a profile diff or a feature-flag bisect before
committing to a fix.

## Step 1: classify the symptom

Pick the row that best matches the actual measurement. The "first probe"
column tells you the cheapest single command that confirms the
classification.

| Symptom | First probe | Likely bucket |
| --- | --- | --- |
| Mean step-time grew across all NEXPs | summarize per-step times over baseline + candidate logs | A. Recipe / config drift |
| Step-time variance grew (mean OK, p99 worse) | replay the per-step MFU stream and check z-scores | B. CUDA-graph capture race |
| MFU dropped only at scale (e.g. 64N OK, 256N bad) | compare scaling efficiency against the prior tier | C. Comm-overlap / fabric ordering |
| Loss diverged, step-time stable | MLLOG `tracked_stats.reduced_train_loss` curve | D. Quantization correctness |
| Hangs at iter 1 (no `tracked_stats` events) | `slurm-<jobid>.out` + NCCL log | E. NCCL env wiring / init |
| DSv3-only regression in MoE path | NVTX ranges around `moe_routing` / `alltoall` | F. MoE feature interaction |
| Step-time stable, GPU mostly idle in nsys | nsys timeline + `host_overhead.py top` | G. CPU / Python host overhead |

Any symptom can land in multiple buckets. Work them in order. The probe
in the second column is meant to be cheap (no new cluster job).

## Step 2: capture a profile if the symptom is not already tracked

Run your per-target capture SOP from `runbooks/`. Inputs to
the diff harness are:

- The candidate `.nsys-rep` from the new run.
- The baseline `.nsys-rep` from your recorded best run for the same
  target (see the `perf-baseline-record` skill).

```bash
python3 tools/pipeline/submission/profile/profile_diff.py \
    --baseline experiments/artifacts/<baseline-family>/<baseline-run>/profiling/<file>.nsys-rep \
    --candidate experiments/artifacts/<candidate-family>/<candidate-run>/profiling/<file>.nsys-rep \
    --out experiments/artifacts/<candidate-family>/<candidate-run>/profiling/profile-diff.md
```

The top deltas in the NVTX / kernel / NCCL tables map to the buckets
below.

## Step 3: bucket-by-bucket bisection

### A. Recipe / config drift

**Signal**: mean step-time grew across all NEXPs. Profile-diff usually
shows a uniform fattening of forward + backward NVTX ranges, not a
single hot kernel.

**Likely culprits** (most common first):

1. Recompute defaults flipped by an upstream merge. Cf. Megatron-Bridge
   `090da658c96a` "no recompute default"
   (commit moved the default away from full recompute). Probe:
   `MEGATRON_EXTRA_ARGS` carrying `--recompute-granularity` should match
   the baseline's. If it changed, the probe template at
   [`tuning/proposals/template-patches/llama31_405b/recompute_granularity_none.json`](/plugins/profile-and-optimize/server/tuning/proposals/template-patches/llama31_405b/recompute_granularity_none.json)
   shows how to pin the value.
2. Quant dtype drifted: cf. Bridge
   `ec5e8c3b0fab` "mxfp8 to fp8_cs for
   h100 gpt-oss". Probe: NVFP4 recipe paths must remain on
   `MixedPrecisionConfig` with `fp4_param_gather` enabled (cf. Bridge
   `ecbd4ead5c1d`).
3. Grouped-GEMM swizzle / checkpoint layout mismatch: cf. Bridge
   `4a4e35a4df03` "add checkpoint
   swizzling" - if the candidate loaded a checkpoint that was swizzled
   under a different convention, GEMMs go to the slow path.

**Fix shape**: pin the recipe value or revert the toggle. Re-run the
candidate. The diff harness should show the regression cleared.

### B. CUDA-graph capture race / missing API leftover

**Signal**: variance grew (median stable, p99 worse). Profile-diff
shows non-deterministic kernel ordering between similar iterations.

**Likely culprits**:

1. M4 leftover: cf. Megatron-LM
   `546a448b4bf0` "M4 leftover for TE cuda
   graph" and `2be925cabe69` "restore
   missing CudaGraphScope import". Probe: `git log --since='2 weeks
   ago' -- megatron/core/transformer/cuda_graphs.py
   megatron/core/full_cuda_graph.py` against the candidate's pinned SHA.
2. Bridge-side TE CUDA-graph cleanup gone: cf. Bridge
   `6a56e1519c79` "cleanup TE cuda
   graphs with the right api". Probe: `train.py` in the candidate's
   bridge SHA.
3. mHC + cuda graph interaction: cf. Megatron-LM
   `722664008bfa` "support mHC with cuda
   graph and activation offloading". DSv3 only.

**Fix shape**: cherry-pick the upstream fix into your Megatron-LM or
Megatron-Bridge fork branch, smoke at the lowest viable scale, then
sync.

### C. Comm-overlap / fabric ordering

**Signal**: MFU dropped only at scale. Profile-diff shows NCCL
collective time grew while compute did not, or kernels reordered such
that compute now waits on NCCL instead of overlapping.

**Likely culprits**:

1. TE `f2ed86bb` "Ordering
   enforcement to split_overlap_rs gemms (#2056)". This is the canonical
   "delay kernel inserted to fix CUDA-graph capture race" commit. Probe:
   nsys timeline at the boundary of `tp_comm_overlap` and the next GEMM.
2. TE `077e26c3` "Use userbuffers for
   MXFP8 wgrad all-gather overlap". Probe: `userbuffers` env / config in
   the candidate.
3. Fabric env regression on the `nccl_ub` path: cf. Bridge
   `6927c9fed204` "use direct
   assignment for NCCL env vars when nccl_ub enabled". Probe: rank-0
   `NCCL_*` env in the candidate's `slurm-<jobid>.out`.

**Fix shape**: pin the upstream TE / Bridge SHA, or roll forward the
NCCL env override. Confirm with a profile-diff at the regression's
scale.

### D. Quantization correctness

**Signal**: loss diverged with step-time stable. Profile-diff often
looks unchanged. The fault is numerical.

**Likely culprits**:

1. TE `5d947a037757` "Fix race in
   dbias kernel (MXFP8 group-quantize)". Probe: NVTX `dbias` range +
   the numerics tests under upstream `TransformerEngine/tests/cpp/operator/`.
2. TE `2d92aa6aae02` "Fix cuteDSL
   kernel numerics when K is 64 aligned". Probe: feature flags around
   the cuteDSL grouped-GEMM path.
3. Mxfp8 weight-quant cache mismatch: cf. Megatron-LM
   `94c6b505b1e5` "Support mxfp8 proj
   gemm weight quant caching". Probe: the candidate's mxfp8 config /
   recipe.

**Fix shape**: cherry-pick the upstream numerics fix. Re-run the
quality eval (`final_log_ppl`) against the recipe target before any
throughput claim.

### E. NCCL env wiring / init

**Signal**: hangs at iter 1, no `tracked_stats` events at all. Often
mistaken for a fabric issue.

**Likely culprits**:

1. NCCL env regression on the `nccl_ub` path: Bridge
   `6927c9fed204` (same commit as
   bucket C). Probe: `NCCL_DEBUG=INFO` log shows `nccl_ub` enabled but
   the userbuffer plugin is not loaded.
2. NCCL_IB_TC / NCCL_P2P_DISABLE drift on the v6.0 image. Some image
   versions cannot init at all with the wrong values. Probe:
   `NCCL_IB_TC` and `NCCL_P2P_DISABLE` in the candidate's launcher env
   vs the launcher defaults.

**Fix shape**: cluster-side env fix on the launcher. Not an upstream
bisect.

### F. MoE feature interaction (DSv3-only)

**Signal**: DSv3 671B regression that does not reproduce on 8B / 405B.
Profile-diff almost always shows growth in `moe_routing` / `alltoall`
NVTX ranges.

**Likely culprits**:

1. mHC + cuda graph + activation offload interaction: cf. Megatron-LM
   `722664008bfa` (same commit as bucket
   B for the cuda-graph axis).
2. Triton dbias_dprob path on MoE experts: cf. Megatron-LM
   `2436e3df7b6d` "Enable dbias_dprob
   triton kernel in TE". Probe: confirm the candidate took the new
   path. If it fell back, NVTX shows the slower bias/dropout kernel.
3. MoE alltoall vs allgather dispatcher default: cf. Bridge
   `1513a102` "GPT-OSS GB200
   dispatcher default to alltoall". DSv3 inherits the same dispatcher
   selection.

**Fix shape**: usually a feature-flag pin. Sometimes a local patch to
the experts path in your Megatron-LM fork.

### G. CPU / Python host overhead

**Signal**: step-time grew but the GPU is mostly idle in nsys. The
diff harness's "CUDA API" delta will look unchanged or smaller.

**Likely culprits**:

1. Megatron-LM `9d976bcd` "reduce CPU
   overhead in modules" - the canonical case. Host_overhead.py top-N
   should show the regression's hot Python frame.
2. Dataloader contention: NCCL traffic from the dataloader colliding
   with training comm.
3. Optimizer / grad-bookkeeping: cf. Megatron-LM
   `f43bc6cb6` "perf(distopt): Cache
   shard buffer".

**Fix shape**: cherry-pick the perf commit, or split the host work off
the hot loop.

## Step 4: confirm and capture

A bisect is only "done" when the diff harness, run again on the fixed
candidate vs the same baseline, shows the regression at or below the
baseline's noise floor (e.g. a cross-NEXP median spread of ~0.018% on
the 8B target). Save the resulting
`profile-diff.md`, `mfu-stream.jsonl`, and `host-overhead-flame.svg`
under
`experiments/artifacts/<family>/<run-id>/profiling/` and link them from
the run's `summary.md`.

## Anti-patterns

- "We saw step-time grow on one NEXP, must be a regression." Cross-NEXP
  noise on the 8B path is ~0.018%,
  do not chase ghosts under that. Run >=3 NEXPs before declaring a
  regression unless the delta is more than 5%.
- "The profile shows kernel X grew, ship a kernel patch." A kernel
  delta is necessary but not sufficient. Confirm the recipe is the
  same. Many "kernel X regressed" reports turn out to be "the
  candidate is taking a different path through TE entirely".
- "We are profiling rank 0 only at scale and not seeing anything." On
  GB300 405B and DSv3 671B, the comm-overlap regressions live on the
  MoE-expert ranks or the NVLink-peer ranks. Profile those ranks too,
  not just rank 0.
