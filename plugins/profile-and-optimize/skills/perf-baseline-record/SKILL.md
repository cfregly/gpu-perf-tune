---
name: perf-baseline-record
last_validated: 2026-05-20
description: >-
  Capture any performance measurement (NCCL bandwidth, MFU, step-time,
  latency, throughput, kernel-time, anything operator-defined) into a
  versioned baseline registry under experiments/artifacts/perf-baselines/.
  Workload-agnostic. Records full provenance (operator, cluster, cohort,
  git SHA, capture command, raw payload sha256) so future regressions can
  be diffed deterministically by the paired perf-baseline-diff skill.
  Triggers on "record baseline", "register baseline", "baseline this
  measurement", "save as baseline", "perf-baseline-record", "baseline
  perf-of-record", or any combination of "record / register / save /
  capture / store" with "baseline / golden / reference / perf-of-record".
allowed-tools:
  - mcp__profile_and_optimize__perf_baseline_record
  - mcp__profile_and_optimize__search_runbooks
  - Bash(sha256sum:*)
  - Bash(date:*)
  - Bash(git:*)
  - Bash(uname:*)
  - Bash(hostname:*)
  - Bash(jq:*)
  - Read
  - Write
---

# perf-baseline-record

## Purpose

Make perf measurements **durable, comparable, and diff-able** by recording them into a versioned registry with full provenance. Replaces the hand-maintained "best run of record" table pattern: any operator can register any measurement (NCCL BW, MFU, latency, throughput, kernel-time, anything that produces a number or structured JSON) and have future regressions diff against it deterministically.

Pairs with [`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md). Together they replace the historical "I'll just remember last week's number" anti-pattern with a registry that survives the operator changing.

### Inference-perf use

For inference perf measurements specifically (TTFT / ITL / throughput / tok-per-user / prefix-cache-hit-rate captured by [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)), prefer the [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md) skill. It wraps this verb with the canonical `inference_perfbench_v1` schema, parses the upstream perf-bench bundle automatically, and supplies per-metric tolerances tuned for inference workloads (TTFT 5%, throughput 3%, cache-hit-rate 2 absolute pts). No new MCP verb is added. The bridge is a thin frontmatter-fluent wrapper around `perf_baseline_record` + `perf_baseline_diff`.

> This skill is backed by a native MCP verb: `mcp__profile_and_optimize__perf_baseline_record`. The verb does the registry write atomically (provenance + source snapshot + INDEX.md update in one shot). The Bash-tool path documented below remains supported as a fallback for environments without the MCP server.

## Why a registry, not a one-shot

A single perf measurement is a number with no shoulder. A registered baseline carries:

- **Provenance**: who recorded it, on what cluster, on what cohort, with what git SHA, with what capture command.
- **Raw payload hash**: `sha256sum` of the source data the measurement was derived from. Lets `perf-baseline-diff` confirm that two measurements were derived from the same shape of input.
- **Versioning**: every registration is a timestamped immutable directory. Nothing is ever overwritten. Going back N versions is `ls` + `cat`.
- **Schema check**: the measurement shape (units, dimensions, ranges) is validated against an operator-supplied JSON schema. Diffing apples-to-oranges fails fast.

## When to use

- After a fabric benchmark run (e.g. the upstream `nccl-tests` suite) on a fresh cluster / driver / firmware release - register the per-N busBW numbers as the new baseline.
- After an `nvbandwidth` link sweep - register the per-node heatmap.
- After a green MLPerf training run - register the step-time + MFU.
- After a kernel-level profile capture - register the top-N kernel-time table.
- Anytime the operator hits a perf number worth remembering.

Do **not** use this skill for:

- One-shot debugging measurements that don't need to be remembered - those go in `experiments/artifacts/<family>/<run-id>/` directly without registration.
- Per-run results that change with every run by design (e.g. live goodput counters) - record those as ordinary run artifacts, not baselines.

## Example prompts

- "Register this nccl-tests bundle as the new B200 8B busBW baseline."
- "Save this MFU number as the perf-of-record for llama31_8b on cohort X."
- "Baseline the nvbandwidth heatmap I just captured."
- "Record this latency number as the inference-baseline for the deepseek-v3 image."
- "perf-baseline-record family=llama31_8b measurement=nccl_busbw value=480 unit=GB/s source=<bundle-path>"
- `/perf-baseline-record --family llama31_8b --measurement nccl_busbw --source experiments/artifacts/nccl-tests/<run-id>/results.json`

## Prerequisites

1. **Source data** - operator names `--source <path>` (file or directory). Must be readable.
2. **Family + measurement-name** - `--family <name>` (e.g. `llama31_8b`, `gb300-cluster`, `deepseek-v3-inference`) + `--measurement <name>` (e.g. `nccl_busbw`, `step_time`, `nvlink_pairwise_bw`).
3. **Value** - `--value <number>` for scalar baselines, or `--source` for structured baselines (the source file IS the baseline payload).
4. **Optional units** - `--unit <gb/s | ms | tokens/s | mfu>`.
5. **Optional schema** - `--schema <path-to-json-schema>` for structured baselines. Skill validates the source against the schema.
6. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for the registry path.

## Interaction style

Mostly autonomous. One pause: confirm the registry path + provenance before write.

## Workflow

### Phase 0: resolve baseline metadata

Resolve and report:

- **Family** (`<family>` becomes the top-level registry directory).
- **Measurement name** (`<measurement>` becomes the per-measurement subdirectory).
- **Value / source** (scalar OR structured).
- **Units** (free-text, operator-supplied).
- **Schema** (optional. If provided, validate the source).
- **Registry path**: `${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/perf-baselines/<family>/<measurement>/<UTC-timestamp>/`.

Ask: "Register?" One confirmation.

### Phase 1: gather provenance

Collect in parallel:

- `Bash(date -u +%Y-%m-%dT%H:%M:%SZ)` - registration time.
- `Bash(hostname)` - workstation that recorded it.
- `Bash(uname -a)` - workstation kernel.
- `Bash(git -C ${PROFILE_AND_OPTIMIZE_REPO_ROOT} rev-parse HEAD)` - current SHA of the bundled server checkout.
- `Bash(sha256sum <source>)` - content hash of the source payload.
- Operator identity: `${USER}` - recorded for the audit trail, not as an ownership claim.

### Phase 2: schema validate (if provided)

If `--schema <path>` was provided, parse the source as JSON and validate against the schema. If invalid, stop and report.

### Phase 3: write

**Preferred (MCP verb):**

```text
mcp__profile_and_optimize__perf_baseline_record with:
  args: ["--family", "<family>",
         "--measurement", "<measurement>",
         "--source", "<source>",
         "--value", "<value>",      # omit for structured baselines
         "--unit", "<unit>",
         "--notes", "<notes>",
         "--json"]
```

The verb does the entire write atomically: creates the immutable directory, snapshots the source, hashes it, writes `baseline.json` + `SOURCE.md`, appends to `INDEX.md`. Returns the entry directory path and the source SHA-256.

**Fallback (Bash-tool path, no MCP server):**

```text
mkdir -p ${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/perf-baselines/<family>/<measurement>/<UTC-ts>/

Write baseline.json with:
  {
    "family": "<family>",
    "measurement": "<measurement>",
    "value": <value or null>,
    "unit": "<unit or null>",
    "source_path": "<source>",
    "source_sha256": "<hash>",
    "schema_path": "<schema or null>",
    "registered_at_utc": "<ts>",
    "registered_by": {"team": "<team-or-project>", "operator_user": "<USER>"},
    "workstation": {"hostname": "<host>", "uname": "<uname>"},
    "profile_and_optimize_sha": "<git-sha>",
    "notes": "<operator-supplied note or empty>"
  }

Copy <source> into the registry directory as source-snapshot.<ext>
Write SOURCE.md with human-readable provenance + the capture command (the operator's original prompt that produced the source).
```

### Phase 4: update the per-measurement index

```text
${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/perf-baselines/<family>/<measurement>/INDEX.md
```

Append a row to the index (one line per registered baseline). Index becomes the chronological history of this measurement.

### Phase 5: confirm + hand off

Print the registry path. Recommend:

- "Diff a future measurement against this baseline" -> [`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md) `--baseline <registry-path>`.
- "Make this the documented run-of-record" -> cite the registry path in your project docs. The `INDEX.md` row is the canonical history.

## Registry layout

```
${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/perf-baselines/
  <family>/                              # e.g. llama31_8b, gb300-cluster
    <measurement>/                       # e.g. nccl_busbw, step_time
      INDEX.md                           # chronological history
      <UTC-timestamp>/                   # immutable per-registration dir
        baseline.json
        SOURCE.md
        source-snapshot.<ext>            # exact source data
      <UTC-timestamp>/
        ...
```

## Safety

- **Never overwrite.** Every registration is a new timestamped directory. The registry is append-only by construction.
- **Schema validation fails fast.** If the source doesn't match the schema, the registry write does not happen. No corrupt baselines.
- **Provenance is mandatory.** A baseline without a `source_sha256` + `profile_and_optimize_sha` + `registered_at_utc` is rejected by the writer.
- **Attribution is audit-only.** `baseline.json` captures the operator's `${USER}` for the audit trail. It is never asserted as a "lead" / "owner" claim.

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

Per `docs/METHODOLOGY.md` "Speed-of-light framing", `baseline.json`
SHOULD carry an optional `sol_pct` field per measurement, computed at
record time and frozen with the baseline (so future diffs can report
SoL deltas against an immutable reference). Schema fragment:

```json
{
  "measurement_value": 310.0,
  "units": "GB/s",
  "sol_pct": 17.2,
  "sol_ceiling_key": "b200_sm100.nvlink5_tbps",
  "sol_ceiling_value": 1800.0
}
```

- `sol_ceiling_key` is the YAML path used to source the peak (e.g.
  `b200_sm100.nvlink5_tbps`, `gb300_nvl72.nvfp4_dense_pflops`) - sourced
  from `configs/sol-ceilings.yaml`. Never inline.
- `sol_ceiling_value` is the value snapshotted at record time so future
  diffs are stable even if the YAML's peak number is later updated.
- `sol_pct` is computed at record time = `measurement_value /
  sol_ceiling_value * 100` for bandwidth measurements. For compute
  measurements use the analogous TFLOPS / peak_TFLOPS ratio.

Baselines for measurements that are not roofline-bound (eval scores,
acceptance rates, etc.) MAY omit the `sol_*` fields.

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

Per `docs/METHODOLOGY.md` "Verdict rigor: DRAFT vs VERDICT", tier every number this
skill records. Default to **DRAFT** (provisional). A baseline is only VERDICT-grade
when it is variance-controlled (same-node, >=3 trials, mean +/- std), metric-isolated
(median TPOT/ITL for decode-latency, not output tok/s at small num_prompts), and
captured against a production-representative config. Record the tier + provenance in
`baseline.json` so a downstream diff can be promoted to a VERDICT.

## Source-of-truth references

- [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) - measurement canon (full-context reporting, verdict rigor, speed-of-light framing).
- [`server/docs/perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md) - raw-payload provenance rules.
- [`server/AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) - fail-fast + artifact-durability rules.
