---
name: inference-kernel-whitebox-debug
last_validated: 2026-06-07
description: >-
  White-box debug a custom CUDA/CUTLASS kernel producing a WRONG numeric result
  (over-amplification, NaN, coherence) after black-box bisection is
  EXHAUSTED: every external lever varied (grid, pipeline depth, layout, dtype),
  the result deterministic + lever-independent, yet the structure reads correct.
  Two tracks: (A) an in-kernel operand + accumulator trace
  (`if constexpr`-gated to survive megakernel pipelining), (B) a standalone
  reproducer .cu instantiating the EXACT kernel template + TMA descriptors with
  controlled inputs (all-ones then real-dump) vs a host GEMM. Localizes the
  defect to operand-load vs MMA vs epilogue vs descriptor. Escalation tier above
  `inference-kernel-ncu-profile` when H/P are fine but the output is numerically
  wrong. Triggers on "white-box kernel", "kernel
  over-amplifies / NaN", "standalone reproducer", "in-kernel trace", "operand
  dump", "cutlass UMMA bug", or any combination of "white-box / reproducer /
  in-kernel / operand-trace" with "kernel / cutlass / UMMA / megakernel / vllm".
allowed-tools:
  - Bash(kubectl:exec,*)
  - Bash(kubectl:get,*)
  - Bash(nvcc:*)
  - Bash(cuda-gdb:*)
  - Bash(jq:*)
  - Bash(sha256sum:*)
  - Read
  - Write
---

# inference-kernel-whitebox-debug

## Purpose

Localize a NUMERIC defect inside a custom CUDA/CUTLASS kernel (not a thin error -- a real
wrong result) to a specific internal stage: operand-load / MMA-accumulate / epilogue /
descriptor. This is the deepest tier of the rigor ladder, above
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md) (proves H +
P -- tensor-core engagement + %SoL) and the "Read the source" rule (reads the kernel
statically). White-box engineering goes further: it instruments or re-runs the kernel's
INTERNAL data path with controlled inputs. (The same **isolated single-op harness** pattern --
call the production library op directly, TP=1, no engine-side imports, synthetic fixed-shape
inputs -- is also the canonical L4 ncu path when in-engine app-replay is non-deterministic on a
TP=N MoE+sparse model. See the ncu capture-hygiene section of the ncu-profile skill.) It is a **standing, first-class option** --
reach for it whenever a K3-K4 kernel is numerically wrong and the mechanism is invisible
from outside. It is not a last resort. Canon: `docs/METHODOLOGY.md`.

## When to use

Escalate here when ALL of:
- the kernel produces a WRONG numeric output (over-amplification, NaN, a coherence /
  magnitude failure), AND
- **black-box bisection is exhausted** -- you varied every external lever (grid,
  `NUM_AB_STAGE`/pipeline depth, layout/shuffle, dtype, the calling config) and the wrong
  output is **deterministic and lever-independent**, AND
- the structural read of the kernel (the cute setup, the MMA loop, the epilogue) looks
  correct yet the output is still wrong.

Do NOT use it as a first move (cheaper: black-box A/B, the structural read, `ncu` for
H/P). Use it precisely when those are spent and the bug is invisible from outside.

## Recipe A -- in-kernel single-tile / operand numerical trace

Instrument the kernel itself to dump the operands as the compute unit consumes them + the
accumulator, then compare to a host/torch hand-compute on the SAME inputs.

1. **Add a `__device__` trace buffer + dump, gated `if constexpr` on the EXACT template
   instantiation under test.** A compile-time gate (e.g. `if constexpr (REDUCTION_SIZE ==
   6144)`) compiles the probe ONLY into the instantiation you care about, survives
   aggressive megakernel cross-layer pipelining (a runtime flag set by another op races
   and dies), and touches no other kernel. Read the SMEM operands the MMA actually
   consumes (for an SS-UMMA, the smem tensors `tCsA`/`tCsB`, not the `tCrA`/`tCrB`
   descriptor iterators) with `atomicMax`(`__float_as_int`) for absmax, plus a one-shot
   `printf` of the first N values to see the layout. Dump the accumulator (`tTR_rAcc`) in
   the epilogue. Read back via the runtime's finalize hook (e.g. `cudaMemcpyFromSymbol`
   in `persistent_kernel.py`), rebuild/JIT, smoke.
2. **Control the three confounds (each manufactures a false "operand is over" reading):**
   - **batch/N padding** -- when `BATCH < MMA_N` the TMA box is clamped, so the padding
     rows (`N >= BATCH`) are stale/zero and an all-elements absmax is confounded. Gate to
     real rows (`N < BATCH`) or print the values to see the real-vs-padding split.
   - **layer/config** -- an all-invocation absmax mixes layers. Late-layer
     massive-activation inputs are legitimately larger. Pin the layer or use a controlled
     input before calling the operand "over".
   - **cross-op flags** -- a runtime flag set by op X and read by op Y dies to pipelining,
     use the compile-time `if constexpr` keyed on the instantiation.
3. **Verdict logic:** operands correct + accumulator ~Nx => the MMA/compute over-counts,
   operand ~Nx (after de-confounding) => operand-load/TMA bug. Accumulator correct =>
   downstream (epilogue/store).

## Recipe B -- standalone reproducer

Author a self-contained `.cu` that instantiates the EXACT kernel template + the EXACT
TMA/descriptor types the production path uses, feeds CONTROLLED inputs, diffs vs a host
GEMM. This removes ALL production-context confounds (interleave, pipeline, layer-varying
input) and definitively localizes scale-vs-layout-vs-stride.

1. **Copy the instantiation + descriptors VERBATIM from the codegen site** (e.g.
   `task_register.cc`'s `register_*_task` `code.e(...)` block) -- the template params, the
   `tma_2d<...>` types (gmem/smem shape, strides, swizzle bits), the bias tensor. Do not
   re-derive them. Transcribe them.
2. **Controlled inputs, analytic check first.** all-ones weight + input => a GEMM output
   MUST equal `K` everywhere. If the kernel gives `~Nx*K`, the compute over-counts. Then
   feed the real dumped input + checkpoint weight and diff vs a host GEMM (`relL2`,
   `kernel/ref_ratio`).
3. **Build with the SAME flags the production JIT uses** -- read them off a real compile
   line (the JIT log): include paths (`-I<tree>/include -I<tree>/include/<pkg> -I.../deps/
   cutlass/include ...`), arch (`-gencode=arch=compute_103a,code=sm_103a` for GB300),
   `-std=c++20`, the project `-D` defines. Produce an EXECUTABLE (drop `-shared` /
   nvshmem-runtime. The task impl is self-contained), launch one CTA (the m_tile loop
   covers all output tiles), set the dynamic smem `cudaFuncSetAttribute` above
   `sizeof(SharedStorage)`.
4. **Runtime-hang caveat (heavier build).** A structurally-valid harness can still HANG
   (100% GPU spin) because the bare 1-CTA launch does not perfectly reproduce the
   production smem/barrier/scheduler context (a TMA-descriptor or named-barrier deadlock).
   Resolve with `cuda-gdb` on the spin (`thread apply all bt`). Commit the `.cu` + build
   script regardless -- they are reusable once the context is reproduced.

## Reconcile + verify

Reconcile Track A + Track B -> name the exact stage/line -> apply the minimal fix ->
verify against the reference trajectory (the golden numeric the kernel must match). A
white-box localization is a **VERDICT** only when a controlled-input reproducer (or an
unconfounded in-kernel trace) isolates it. A padding/layer-confounded absmax is a
**DRAFT** -- walk it back per DRAFT-vs-VERDICT (that confound is exactly how an
over-confident "the input is Nx" / "the router is also over" claim sneaks in).

## Cross-skill join

| Question | Tool | Skill |
|---|---|---|
| What kernels run + sample-share? | zymtrace | `zymtrace-anchored-query` |
| Absolute per-kernel duration / cuda-graph timeline | nsys | `inference-kernel-profile` |
| Per-kernel occupancy / regs / smem / DRAM-BW / roofline (H + P) | ncu | `inference-kernel-ncu-profile` |
| Task-graph wiring (declared-but-unconsumed tensor) | task-graph audit | `mirage-graph-coverage` |
| **Why is the kernel's NUMERIC OUTPUT wrong (operand vs MMA vs epilogue)** | **in-kernel trace + standalone reproducer** | **this skill** |

Run `ncu` first to confirm H/P are fine (tensor cores engaged, near-ceiling) -- if H/P
are the problem it is a K/R/H/P/A mismatch, not a numeric bug. This skill is the
escalation when H/P are fine but the output is numerically wrong.

## Scaffolding (MCP)

`perf_tune_report_kernel_reproducer_scaffold` generates the Track-B `.cu` + build script from a
kernel signature (header, kernel name, template params, input source), so the boilerplate
(includes, instantiation, controlled inputs, host-GEMM diff, the GB300 build flags) is not
hand-retyped each time. Hand-edit the generated descriptors to match the codegen site
exactly.

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

A white-box defect hunt is almost always on a K3-K4 op (dense GEMM / attention / NVFP4 MoE
GEMM) at R1-R3 (library / CuTe DSL). Record `(K,R,H,P,A)` for the kernel under test in the
bundle's `SOURCE.md`. White-box engineering is orthogonal to the H/P win-gate (that is
`inference-kernel-ncu-profile`'s job) -- it answers CORRECTNESS, not competitiveness: a
kernel can be H4/P4 (tensor cores, near-ceiling) and still be numerically wrong, which is
exactly when you escalate here.

## Worked example: an apparent ~10x over-amplification refuted by the reproducer

A shared-expert gate_up GEMM (BF16, K=6144/OUTPUT=1024) appeared to over-amplify ~10x.
Black-box bisection refuted ~13 mechanisms and the structural read was clean. Track A
(an `if constexpr (REDUCTION_SIZE==6144)` operand trace of `tCsA`/`tCsB` + accumulator)
confirmed the weights load correctly (absmax 1.30) and the real input rows load
faithfully -- an all-layer input absmax of 11-15 was late-layer massive activations (the
padding/layer confound, walked back) -- pointing at the MMA itself. Track B (a standalone
reproducer with the exact instantiation + TMA descriptors, all-ones input, sm_103a build)
first hit the runtime-hang friction. The hang was the `BATCH=8 < MMA_N=16` clamped-box
path. At `BATCH=16` the reproducer ran clean and the GEMM was **numerically correct**
(all-ones -> out==K=6144, ratio 1.000. Non-uniform -> relL2 0.00182 vs a host GEMM): the
over-amplification was REFUTED. The "~10x" was the `BATCH<MMA_N` padding confound (stale
smem rows read by an all-rows absmax) -- the exact DRAFT-vs-VERDICT lesson this skill
encodes: a VERDICT needs the controlled-input reproducer, not an in-kernel absmax.
