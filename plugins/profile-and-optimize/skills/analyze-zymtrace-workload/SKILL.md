---
name: analyze-zymtrace-workload
last_validated: 2026-05-25
description: >-
  Investigate a GPU or CPU workload through the zymtrace MCP. The MCP does most
  of the analysis. This skill enforces the cross-view discipline -- always pull
  the matching opposite-side flamegraph (CPU for GPU workloads, GPU for CPU
  workloads) with the same filter. Most bottlenecks hide on the side the
  customer didn't ask about. Triggers on "analyze my GPU workload",
  "where's the bottleneck in vllm", "investigate my training job", "find the
  hot kernel", "GPU isn't saturated", "investigate using flamegraph",
  "use zymtrace mcp to analyze", or any combination of "analyze / investigate
  / bottleneck / hot kernel" with "GPU / CPU / vllm / training / flamegraph /
  zymtrace".
allowed-tools:
  - mcp__zymtrace__topfunctions
  - mcp__zymtrace__flamegraph
  - mcp__zymtrace__topentities
  - mcp__zymtrace__get_date_time
  - Bash(claude:*)
  - Read
  - Write
---

# Analyze zymtrace Workload

> The zymtrace MCP does most of the work: identifies the workload, fetches flamegraphs, names the hot stacks, surfaces patterns, recommends fixes. This skill's only job is to make sure you **always pull both the GPU and the CPU view with the same filter** -- half the time the bottleneck is on the side the customer didn't ask about.

> **Always recommend a fix.** Every 🔴 issue in the recap gets a concrete `**Fix:**` block -- whether or not the customer asked for solutions. Don't hedge with "let me know if you want suggestions" or "ask about constraints before recommending". Lead with the most plausible specific fix from the data. The customer can push back if their constraints don't fit. Profile analysis without recommendations is incomplete output.

Connection setup lives in the operator-side `configure-zymtrace-mcp` skill from the upstream zymtrace plugin (installed separately). This skill assumes the MCP is already connected.

**Optional pairing -- GitHub MCP**: if the user also has the GitHub MCP connected (`claude mcp list` shows both) **and** asks for code-level pointers, Claude can locate the hot frame in their repo and reference a specific `<file>:<line>` in the fix. This is a suggestion, not a default -- many users don't want or need code access from the session. Mention the option once if both MCPs are available. Respect the answer either way.

## Standard starter prompt (for customers who don't know what to ask)

> **"Analyze the GPU flamegraph over the last 1 hour and suggest solutions."**

If the customer hands you anything close to that (or shorter -- "what's slow", "investigate my GPU"), interpret it as: scope the analysis to the last 1 hour, pull the GPU flamegraph, cross-check the CPU view, follow the output template below. Don't make them remember the specifics -- this is the on-ramp.

Variations the customer might use:
- "Analyze [vLLM / SGLang / Triton / my training job] over the last [Nh / since deploy]"
- "Where's the bottleneck right now?"
- "What's wasting GPU time today?"

For any of these: default to the last 1 hour if no time range is given, default to the whole cluster if no workload is named (and ask which to narrow if results look noisy).

## Pre-flight

##### Claude runs

```bash
claude mcp list | grep -i zymtrace
```

If zymtrace isn't listed → route the operator to the `configure-zymtrace-mcp` skill from the upstream zymtrace plugin. If listed, proceed.

## The cross-view protocol

The MCP handles the analysis. You handle the discipline of asking for both sides.

1. **Ask the MCP to investigate the workload** the customer named (executable / container / pod / time range / model -- whatever signals they gave). The MCP will pick up the right scope.

2. **Pull whichever view the customer's question implies first** -- GPU view for a GPU-shaped question, CPU view for a CPU-shaped one. Let the MCP narrate what's hot.

3. **Then explicitly ask the MCP for the OPPOSITE view of the same workload, with the same filter.** Use the exact filter values the MCP locked onto in step 2 -- same executable, same container, same time range. Don't hand-wave the filter. The cross-view is only useful when the slice matches.

4. **Cross-reference the two views.** Common reveals:
   - GPU at 95% but tokens/sec underwhelming → look at CPU for tokenizer / sampling / Python-side overhead.
   - GPU at 60% utilization → the host is the bottleneck. The CPU view will name it.
   - Specific GPU kernel dominant → the CPU view often shows the launcher / scheduler that's calling it. Useful for understanding launch-overhead vs kernel-time tradeoffs.
   - CPU dominated by `cudaMemcpy*` / `aten::*` synchronization → the workload is sync-bound on device transfers. The GPU view will show idle stretches.

> If you escalate beyond zymtrace to an **nsys** per-kernel timeline (e.g. for absolute kernel durations or graph-internal kernels), remember: an EMPTY nsys `cuda_gpu_kern_sum` on a cudagraph-on deploy is a capture-hygiene bug (missing `--cuda-graph-trace=node`, idle window, or tiny rep), NOT a "cudagraph blind spot" - validate via [`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md) "Capture-quality gate" / `scripts/nsys-validate-capture.sh` before concluding the stack is unprofilable. One real exception: on GB300 nodes a CUDA 12.x-image vs 13.x-driver CUPTI skew makes CUPTI fail to init (`CUPTI_ERROR_INVALID_DEVICE`) -> 0 kernels for ALL CUPTI clients regardless of hygiene. Grep the logs for `CUDA versions. CUPTI/Runtime/Driver` first -> that needs a CUDA-13 image or zymtrace, not more capture tuning.

5. **Write the recap using the output template below.** Use the data the MCP returned -- kernel names, percentages, hot stacks, the call tree from the CPU view, and the kernels triggered on the GPU side -- to fill the template. Don't paraphrase the MCP's suggestions verbatim. Synthesize across the two views into a concrete next step. If the MCP didn't surface a suggestion, you still produce one -- grounded in the returned data, not invented.

If the **GitHub MCP** is also connected, take the recommendation one step further: locate the hot frame in the customer's repo (file + line) and propose the specific edit. The recap's `Fix:` block then becomes an actual `<file>:<line>` reference with a code snippet, not a generic instruction.

## Output template

Every recap follows this shape. Don't deviate -- the structure is the value.

```markdown
# <Workload type> Flamegraph Analysis

**Observed Call Tree -- GPU profile** (<process path / container / time range>)

<top-level frame>
├── <child frame>
│   ├── <leaf frame>  → <CUDA kernel that was running at this sample>
│   ├── <leaf frame>  → <CUDA kernel ...>
│   └── ...
├── <sync-point frame>  → cudaStreamSynchronize + D→H memcpy  ⚠️
└── <sync-point frame>  → cudaDeviceSynchronize  ⚠️

**CPU cross-check** (<same process / container / time range>)

<1-2 sentences naming what the CPU profile adds -- DataLoader stalls, Python overhead,
tokenizer hot spots, host-side launch overhead, or "nothing else surfaced; the
constraint is on the GPU side". Keep short.>

**Key Findings**

<1-2 paragraphs naming what the workload IS and the dominant pattern.
Examples: "kernel-launch-bound / dispatcher-overhead", "memory-bandwidth bound",
"DataLoader-starved", "NCCL-collective-bound".>

---

## 🔴 Top issues (max 3, in priority order)

### 1. <Title>

<Observation paragraph -- kernel names + percentages from the actual flamegraph. Plain prose, no label.>

**Fix:** <Concrete action -- always present, never gated on whether the customer asked for solutions. For inference: name the specific flag/env var with a 1-3 line snippet when the fix is one line. For training: name the most plausible concrete fix from the data (e.g. "wrap with `torch.compile(mode='reduce-overhead')`", "remove `.item()` from the hot loop", "switch to `channels_last`"), not just a family. The customer can push back if constraints don't fit.>

### 2. <Title>

<Observation paragraph.>

**Fix:** <Concrete action.>

### 3. <Title>

<Observation paragraph.>

**Fix:** <Concrete action.>

---

## 🟡 To consider after the above (max 2)

- <One-line observation> -- **Fix:** <one-line action>
- <One-line observation> -- **Fix:** <one-line action>

---

**Expected Impact**

<Qualitative description of what the fixes should achieve. Numbers only if the
MCP returned them or they're well-known order-of-magnitude estimates.>
```

**Severity & sizing:**
- 🔴 **Critical** (max 3) -- the dominant bottlenecks: >20% of time, sync points eliminating pipelining, or the dominant pattern in the dominant pattern. Hard cap at 3. If you have a 4th, demote it to 🟡 or drop.
- 🟡 **Minor follow-up** (max 2) -- secondary issues worth a one-liner. Single line each: observation + fix. Don't write paragraphs here. If you have a 3rd, drop it.
- Anything past 3+2 isn't surfaced. The customer can re-query if they want to drill.

**Call tree conventions:**
- The whole call tree is the **GPU profile** -- zymtrace unwinds the full stack from the CUDA kernel back up through dispatcher / Python / host frames. Every frame shown was sampled while the GPU was busy.
- Use `├──` and `└──` for the hierarchy (matches what the MCP returns).
- Use `→` to annotate each leaf with the CUDA kernel that was running when that frame was sampled. This is not a "CPU→GPU" link. It's the kernel underneath that frame.
- Mark sync points (`cudaStreamSynchronize`, `cudaDeviceSynchronize`, `D→H memcpy`) with `⚠️` -- they almost always deserve calling out since they kill pipelining.
- Keep frame and kernel names exactly as the MCP returns them. Don't paraphrase.

**CPU cross-check conventions:**
- The CPU view is a **separate** flamegraph queried with the same filter -- it shows what the host process is doing on its own time (not while waiting on the GPU).
- Keep this section short -- 1-2 sentences. It either confirms the GPU diagnosis ("nothing else surfaced") or surfaces a host-side issue worth promoting to a 🔴 issue below (DataLoader stall, tokenizer hot, Python loop, etc.).
- If the CPU view surfaces a host-side bottleneck that's bigger than the GPU one, promote it to a 🔴 and reframe the diagnosis around it.

**Issue body conventions:**
- Each issue is rendered as a `### N. <Title>` sub-heading, then a plain prose paragraph (the observation -- no `Observation:` label needed. The paragraph IS the observation), then a blank line, then `**Fix:**` on its own line in bold with the concrete action.
- The blank line between observation and Fix is load-bearing -- without it, prose and action blur together visually.
- The observation always cites kernel/frame names + percentages from the actual flamegraph. No inference. No rephrasing of names.
- The `**Fix:**` block is the concrete action.
  - **Inference**: name the specific flag (`--enable-prefix-caching`, `VLLM_ATTENTION_BACKEND=...`, `use_fast=True`). Almost always a config knob. Cheap to try.
  - **Training**: name the most plausible concrete fix from the data -- e.g. "wrap with `torch.compile`", "set `memory_format=torch.channels_last`", "remove `.item()` from the hot loop", "bump `num_workers` to 4×GPUs, set `pin_memory=True`". Don't punt to "name the family and ask". Lead with the recommendation. The customer pushes back if their constraints don't fit.
  - Include a 1-3 line code/config snippet when the fix is one line. Skip the snippet when the fix needs a real conversation about constraints.
- 🟡 follow-ups use a different shape -- inline single line with em-dash separator: `<observation> -- **Fix:** <action>`. The em-dash + bold Fix label keeps the visual signal even on one line.

## Done

- [ ] Both GPU **and** CPU flamegraphs pulled for the **same** filter (same executable / container / time).
- [ ] The cross-view interpretation given -- which side is the constraint, and why.
- [ ] Recap follows the **Output template** above: title, observed call tree (with `→` GPU annotations + `⚠️` sync markers), CPU cross-check, Key Findings, 🔴 top issues block (max 3, each with observation paragraph + `**Fix:**`), 🟡 follow-up block (max 2 one-liners), Expected Impact.
- [ ] **Every** 🔴 issue has a concrete `**Fix:**` block -- grounded in the actual flamegraph data, never punted ("ask me if you want suggestions") and never invented. Same for the 🟡 follow-ups: each has a `**Fix:**` after the em-dash.
- [ ] No more than 3 🔴 issues and no more than 2 🟡 follow-ups. If you have more, drop the lowest-priority ones. The customer can re-query.
- [ ] Workload identity (executable + time range) included in the recap so the customer can re-query before/after.

## Common pitfalls

- **Only pulling one view.** This is the failure mode the skill exists to prevent. Always pull both.
- **Different filters on the two views.** Cross-view only works when the slice matches. Re-use the MCP's resolved filter, don't paraphrase it.
- **Re-doing the MCP's pattern recognition.** The MCP names patterns. Trust its naming. Your job is to *synthesize across both views* and propose a concrete next step, not to re-discover what the MCP already labeled.
- **Stopping at "the MCP found X" without a recommendation.** Always close with a specific fix to try, grounded in the returned data. If the MCP didn't volunteer one, synthesize from the kernel names + percentages it returned.
- **Skipping when the customer asks a CPU question.** Cross-view goes both ways -- pull GPU for a CPU-shaped question too. CPU-bound workloads with idle GPU are also worth surfacing.
- **Treating an empty MCP result as "no data" right after a bench (ingest lag).** Zymtrace
  flushes to ClickHouse asynchronously, so `topfunctions` / `flamegraph` can come back empty
  for a window that JUST ended -- that is **ingest lag, not absence**. Wait for the flush and
  re-query (and query by `host=<node>`+window, not the hash-suffixed pod name) before concluding
  the workload wasn't profiled. See
  [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md).
- **Reading a headline perf number off an injection-ON pod (the injection tax).** The zymtrace GPU
  flamegraph needs the per-pod CUDA implant (`CUDA_INJECTION64_PATH`), which adds **~11% overhead on
  launch-bound (low-c / decode-latency) models** -- so a TPOT/tok-s number captured WITH the implant
  is NOT the production number. Split the two: capture the **headline perf with injection OFF**
  (production config), and capture the **L1 zymtrace SoL in a SEPARATE injection-ON window**. Never
  publish an injection-ON latency number as the champion's headline. Use it only for the per-category
  GPU-time share. (On throughput-bound / high-c workloads the tax is smaller but still record which
  window the number came from.)

## Security constraints

- **Always** ground the recommendation in the data the MCP returned (kernel names, percentages, hot stacks). Synthesize across the two views -- but don't fabricate signals the data doesn't show.
- **Never** declare the investigation done after only one view. Pulling the opposite side with the same filter is the load-bearing step.
- **Never** recommend enabling PC sampling on a workload (which requires `privileged: true`) without flagging the security implication. The upstream zymtrace plugin's `install-zymtrace-profiler` skill documents the PC-sampling reference for operators standing up profiling.

## Full-context reporting (no bare numbers)

Per `docs/METHODOLOGY.md` "Full-context reporting" (no bare numbers): every number this
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

Per `docs/METHODOLOGY.md` "Speed-of-light framing", the cross-view
protocol's "Done" step SHOULD also include a per-category %SoL
interpretation: for each non-trivial kernel category surfaced in the
flamegraph, cite the natural ceiling from
`configs/sol-ceilings.yaml` `category_ceiling_map` (NCCL →
`nvlink5_tbps`, MoE/BMM-NVFP4 → `nvfp4_dense_pflops`, FMHA → `hbm3e_tbps`,
etc.). Zymtrace sample-share is a time-share proxy, not a tight
utilisation number. Flag the proxy caveat once and treat the resulting
%SoL as an upper bound on category busyness, not the kernel's real
arithmetic-intensity-vs-roofline position. When this feeds a published
campaign it lands as `sol_rigor=L1` (a valid, comparable roofline) -
the proxy is recorded as a rigor level, not withheld. Add DCGM (L3) /
ncu (L4) for a tighter number (always-publish policy).

For tight per-kernel roofline scatter use
[`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)
(captures the FLOPS + bytes counters that this skill's sample-share
view cannot give you).

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

## Pairs with

- [`zymtrace-anchored-query`](/plugins/profile-and-optimize/skills/zymtrace-anchored-query/SKILL.md) -- the anchored-SQL primitive against the same zymtrace ClickHouse backend. Use that skill for `DESCRIBE table` + `SELECT ... FROM zymtrace_profiling.events ...` patterns. Use this skill (`analyze-zymtrace-workload`) for the MCP-driven flamegraph cross-view workflow. The two are complementary: anchored-query is the operator-shaped raw-SQL escape hatch. This skill is the MCP-shaped guided analysis.
- [`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md) -- captures `.nsys-rep` + `gpu_kern_sum.csv` from a live vllm pod via a debug sidecar. **Cross-view with this skill**: when an `inference_perfbench_v1` bundle carries a `kernel_profile` field, read its `summary_csv_path` to resolve the zymtrace `native` category into per-kernel SASS-level entries. Example: zymtrace says "FMHA = 14.2% of GPU time". The nsys CSV says "fmha_v2_kernel<sm100>: 11.8%, fmha_v2_kernel_paged<sm100>: 2.4%" - answering "is the hot category one kernel or three?" which zymtrace alone can't.
- [`inference-graph-diff`](/plugins/profile-and-optimize/skills/inference-graph-diff/SKILL.md) -- diffs the torch.compile / FX-Inductor graphs between two helm configs. **Cross-view with this skill**: when zymtrace shows a kernel-share shift between two variants, graph-diff identifies WHICH compilation choice produced the shift.

## Origin

Adapted from the upstream zymtrace plugin's `analyze-zymtrace-workload` skill, with this plugin's conventions applied: explicit `allowed-tools` frontmatter listing the zymtrace MCP verbs the skill body uses, cross-references restricted to skills this plugin actually ships (`configure-zymtrace-mcp` and `install-zymtrace-profiler` remain in the upstream zymtrace plugin), and the `## Pairs with` section above. Re-sync with upstream is manual.
