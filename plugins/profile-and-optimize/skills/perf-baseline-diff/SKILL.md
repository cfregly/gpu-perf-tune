---
name: perf-baseline-diff
last_validated: 2026-05-20
description: >-
  Diff a current performance measurement against a registered baseline from
  the perf-baselines registry. Works for any measurement type (NCCL BW, MFU,
  step-time, latency, throughput, per-kernel times, structured heatmaps,
  nsys-rep profiles). Honors the operator-supplied
  tolerance, reports per-dimension deltas, and classifies the overall verdict
  as GREEN / YELLOW / RED. Pairs with perf-baseline-record. Triggers on
  "diff baseline", "diff against baseline", "perf regression check",
  "is this regression real", "compare to baseline", "perf-baseline-diff",
  "regression diff", or any combination of "diff / compare / regression /
  check" with "baseline / golden / reference / perf-of-record".
allowed-tools:
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__profile_profile_diff
  - mcp__profile_and_optimize__search_evidence
  - Bash(sha256sum:*)
  - Bash(jq:*)
  - Bash(diff:*)
  - Read
  - Write
---

# perf-baseline-diff

## Purpose

Take a current perf measurement (any kind) and a registered baseline (from [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md)), produce a structured diff with per-dimension deltas, and classify the overall verdict. Works for scalar, structured, and nsys-rep measurements.

> This skill is backed by a native MCP verb: `mcp__profile_and_optimize__perf_baseline_diff`. The verb handles scalar + structured-JSON shapes natively and delegates nsys-rep diffs to `mcp__profile_and_optimize__profile_profile_diff`. The Bash-tool path documented below remains supported as a fallback.

Output:

- A **per-dimension delta table** (every (key, baseline-value, current-value, delta, delta-%) row).
- An **overall verdict**: GREEN (within tolerance), YELLOW (one or two dimensions outside tolerance), RED (multiple dimensions outside tolerance OR the headline dimension regressed beyond tolerance).
- A **next-action recommendation** (bisect, ignore, file regression, etc.).

Handles three measurement shapes:

1. **Scalar** - single number (e.g. step-time-ms). Delta is `current - baseline`, delta-% is `100 * (current - baseline) / baseline`.
2. **Structured key->value** (e.g. per-pair NCCL BW). Computes delta per key. Reports worst-N regressions.
3. **Nsys-rep** - delegates to `mcp__profile_and_optimize__profile_profile_diff` (the profile-diff harness shipped with the `profile_and_optimize` MCP server).

### Inference-perf use

For inference perf-bench output specifically, prefer the [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md) skill. It is a structured-shape diff (Phase 2b below) keyed on the `inference_perfbench_v1` schema, with per-metric tolerances tuned for inference workloads (TTFT 5%, throughput 3%, cache-hit-rate 2 absolute pts) and a verdict-classification step that knows the headline metric is `throughput_toks_per_s` at concurrency 16. No new MCP verb is introduced. The bridge wraps `perf_baseline_diff` directly.

## When to use

- After re-running a fabric benchmark (e.g. upstream `nccl-tests`) post-driver-upgrade, diff against the pre-upgrade baseline.
- After re-running an `nvbandwidth` link sweep after a hardware swap, diff against the prior heatmap.
- After a candidate training run, diff step-time / MFU against the registered best run.
- Periodic regression sweep: weekly cron, diff each measurement against the most recent registered baseline.

Do **not** use this skill for:

- Comparing two arbitrary measurements neither of which is in the registry - that's `diff` or `jq` directly. Register one of them as a baseline first.
- One-shot debugging - too much ceremony for a quick eyeball check.

## Example prompts

- "Diff this morning's nccl-tests bundle against the baseline from last Tuesday."
- "Is this MFU regression real? Diff against the registered baseline."
- "Compare this nvbandwidth heatmap to the one we registered before the firmware upgrade."
- "Perf-baseline-diff family=llama31_8b measurement=nccl_busbw current=<bundle> tolerance=5%"
- `/perf-baseline-diff --baseline experiments/artifacts/perf-baselines/llama31_8b/nccl_busbw/<UTC-ts>/ --current experiments/artifacts/nccl-tests/<run-id>/results.json --tolerance-percent 5`

## Prerequisites

1. **Baseline path** - `--baseline <registry-path>` (directory from `perf-baseline-record`).
2. **Current measurement** - `--current <path>` (file or directory).
3. **Tolerance** - `--tolerance-percent <N>` for scalar measurements, `--tolerance-absolute <N>` for structured. Default: `5%`.
4. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for writing the diff bundle.

## Interaction style

Fast and autonomous. One pause: confirm the baseline + current paths + tolerance before writing the diff bundle. The verdict + recommendation is the last step.

## Workflow

### Phase 0: resolve paths

- `Read` `<baseline>/baseline.json` to extract `family`, `measurement`, `unit`, `source_sha256`, `value`, `source_path`.
- `Bash(sha256sum <current>)` to hash the current source.

### Phase 1: determine shape

- If `<baseline>/source-snapshot.nsys-rep` exists -> shape = `nsys-rep` (delegate to Phase 2c).
- Else if `baseline.value` is set -> shape = `scalar` (Phase 2a).
- Else if `<baseline>/source-snapshot.*` parses as JSON dict -> shape = `structured` (Phase 2b).
- Else: error and stop.

### Phase 2a: scalar diff

**Preferred (MCP verb):**

```text
mcp__profile_and_optimize__perf_baseline_diff with:
  args: ["--baseline", "<baseline-registry-entry-dir>",
         "--current", "<current-source-path>",
         "--tolerance-percent", "5",
         "--json"]
```

The verb auto-detects shape (scalar / structured JSON / nsys-rep), runs the diff, writes the diff bundle to `experiments/artifacts/perf-baseline-diffs/<family>/<measurement>/<UTC-ts>/`, and returns the verdict + bundle path.

**Fallback (Bash-tool):**

```text
delta = current_value - baseline.value
delta_pct = 100 * delta / baseline.value
verdict = "GREEN" if |delta_pct| <= tolerance_percent else "RED"
```

### Phase 2b: structured (key->value) diff

```text
For each key in baseline AND current:
  baseline_v = baseline[key]
  current_v  = current[key]
  delta      = current_v - baseline_v
  delta_pct  = 100 * delta / baseline_v

Sort by |delta_pct| descending; top 20 in the report.

verdict:
  GREEN  if no key exceeds tolerance
  YELLOW if 1-2 keys exceed tolerance
  RED    if 3+ keys exceed OR headline key (operator-flagged) exceeds
```

### Phase 2c: nsys-rep diff (delegate)

```text
mcp__profile_and_optimize__profile_profile_diff with:
  args: ["--baseline", "<baseline-nsys-rep>",
         "--candidate", "<current-nsys-rep>",
         "--out", "<diff-bundle>/profile-diff.md",
         "--json"]
```

Then translate the harness output into the same verdict shape (GREEN/YELLOW/RED) for consistency.

### Phase 3: write diff bundle

```
${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/perf-baseline-diffs/<family>/<measurement>/<UTC-ts>/
  diff.json                 # structured diff: baseline_sha, current_sha, per-key delta, verdict
  diff.md                   # human-readable: top 20 deltas, verdict, recommendation
  baseline-ref.txt          # absolute path to the baseline directory
  current-snapshot.<ext>    # copy of the current source for audit
```

### Phase 4: recommend

Based on verdict:

- **GREEN**: "no regression detected. No action needed."
- **YELLOW**: "spot regression on N keys. Recommend re-running the measurement on the same cohort to rule out noise, then re-diff."
- **RED**:
  - If shape == `nsys-rep`: drill into the `profile_profile_diff` per-kernel bucket output to name the regressed bucket.
  - If shape == `structured` and the regressions cluster on specific keys (e.g. all on one node-pair, or all on one collective): name the suspect axis.
  - If shape == `scalar`: name the absolute + percent delta and ask the operator whether to bisect.

## Safety

- **Read-only on the registry.** This skill never writes to `experiments/artifacts/perf-baselines/`. It writes only to `experiments/artifacts/perf-baseline-diffs/`.
- **Provenance preserved.** The diff bundle records `baseline_sha256` and `current_sha256` so the comparison is auditable.
- **No silent shape conversion.** If the baseline is structured and the current is scalar (or vice versa), the skill stops and asks. No fuzzy diffing.
- **Tolerance is explicit.** Default 5% but operator must agree. The skill prints the resolved tolerance in Phase 0 and writes it into the diff bundle.

## Full-context reporting (no bare numbers)

Per `docs/METHODOLOGY.md` "Full-context reporting": every number this
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

Per `docs/METHODOLOGY.md` "Speed-of-light framing", when the baseline
carries `sol_pct` / `sol_ceiling_key` / `sol_ceiling_value` (the
optional fields added by
[`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md)), the diff
output MUST also report **SoL delta** alongside the absolute delta.
Format:

```
measurement: nccl_all_reduce_busbw @ 1GB
  baseline:  310 GB/s   (17.2% of nvlink5_tbps SoL)
  current:   295 GB/s   (16.4% of nvlink5_tbps SoL)
  delta:     -15 GB/s   (-4.8% absolute, -0.8 pp SoL)
  verdict:   YELLOW (within 5% tolerance, but trending away from peak)
```

The SoL delta surfaces ceiling-distance shifts that absolute deltas
hide: a 4.8% drop on a measurement already at 90% SoL is structurally
different from the same drop at 30% SoL (former leaves no recovery
room. Latter is well within recoverable headroom).

When the baseline lacks `sol_*` fields (older baselines, or
non-roofline-bound measurements), the diff skips the SoL line silently -
no synthesised retrofit.

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

## Verdict rigor (DRAFT vs VERDICT)

Per `docs/METHODOLOGY.md` "Verdict rigor: DRAFT vs VERDICT", tier the diff verdict.
A GREEN/YELLOW/RED call is a **DRAFT** unless it is variance-controlled (same-node,
>=3 trials per arm, mean +/- std - a single-trial or cross-node delta is provisional),
metric-isolated (median TPOT/ITL for decode-latency claims, not output tok/s at small
num_prompts), and against a production-representative baseline. A "which kernel/path
regressed" claim additionally needs nsys/ncu per-kernel data, not a DCGM regime %.
Supersede a DRAFT everywhere once a controlled VERDICT overturns it.

## Source-of-truth references

- [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md) - the pair skill.
- [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) - measurement canon (full-context reporting, verdict rigor, speed-of-light framing).
- [`server/AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) - fail-fast and provenance rules.
