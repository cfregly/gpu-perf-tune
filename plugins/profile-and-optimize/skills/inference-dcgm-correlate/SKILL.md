---
name: inference-dcgm-correlate
last_validated: 2026-05-27
description: >-
  Correlate DCGM Prometheus byte-traffic counters with an inference-perf-bench
  sweep window to compute byte-grounded workload-level Speed-of-Light. The
  third tier of the SoL rigor hierarchy (after zymtrace
  sample-share and ncu per-kernel arithmetic intensity). Reads the sweep
  window from `inference_perfbench_v1.json.bench`, queries Prometheus via
  the Prometheus MCP for the DCGM PROF group (`DRAM_ACTIVE`, `NVLINK_TX/RX_BYTES`,
  `PIPE_TENSOR_ACTIVE`, `PIPE_FP16_ACTIVE`), falls back to `DCGM_FI_DEV_*`
  counter-tier metrics when PROF is not exported, and writes
  `<cell>/dcgm_correlation.json` with per-resource %SoL = real byte traffic
  / (peak * duration * n_gpus). Triggers on "dcgm correlate", "dcgm sol",
  "byte-grounded sol", "workload %sol", "dram active over sweep",
  "nvlink bytes", "tensor pipe active", "real-vs-peak workload bandwidth",
  or combinations of "dcgm / prometheus" with
  "sweep / window / sol / workload / byte-traffic".
allowed-tools:
  - mcp__prometheus_mcp__query_prometheus
  - mcp__prometheus_mcp__query_observability_knowledge_base
  - mcp__profile_and_optimize__perf_tune_report_dcgm_correlate
  - Read
  - Write
  - Bash(jq:*)
  - Bash(sha256sum:*)
  - Bash(perftunereport:*)
---

# inference-dcgm-correlate

## Purpose

Lift workload-level Speed-of-Light from "first-principles estimate" to
"measured byte traffic over the sweep window". The methodology canon
(`docs/METHODOLOGY.md` "Speed-of-light framing") names three levels of SoL rigor:

> **Steady-state window:** per-(c) DCGM correlation is only meaningful when the bench cell
> sustained steady state, i.e. `num_prompts >= 2*c`. At `num=c+4` the window is ramp/drain-
> dominated so BOTH the throughput AND the high-c DCGM utilization read low (see
> `docs/METHODOLOGY.md` "Capture hygiene"). `import_roofline_sweep` WARNs on any cell `< 2c`.

1. **Sample-share proxy** (page 4 in the perf-report PDF) - zymtrace
   per-category time-share read as a coarse upper bound on category
   busyness. (zymtrace flushes to ClickHouse asynchronously, so an empty L1
   right after the window is **ingest lag, not absence** - wait + requery for
   the freshest data. See
   [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md).)
2. **ncu per-kernel arithmetic intensity** (page 5) - proper roofline
   scatter from ncu DRAM bytes + SM FLOPS counters.
3. **DCGM workload-level byte traffic** (page 6) - this skill. Real GB
   transferred across NVLink / HBM / Tensor pipe during the drive_load
   sweep, divided by peak × duration × n_gpus.

This skill is the page-6 producer.

## When to use

- After a campaign's `drive_load.py` sweep completes and the bundle
  has an `inference_perfbench_v1.json.bench.captured_at +
  duration_effective_s` pair recording the sweep window.
- When you want a workload-level %SoL number anchored in
  measured byte counters, not a first-principles HBM-roofline
  estimate.
- When refreshing a campaign's `sol-summary.md` with byte-grounded
  numbers replacing the time-share proxies.

Do **not** use for:

- Per-kernel arithmetic-intensity questions - that's
  [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)'s
  domain (DCGM has no per-kernel attribution).
- Real-time cluster health - DCGM scrape interval (~10-30 s) is too
  coarse for sub-minute incident response. Use
  [`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md)
  directly with the regular dashboard panels.

## Prerequisites

- The campaign / cell directory exists at
  `campaigns/<campaign>/cells/<cell>/`.
- The source `inference_perfbench_v1.json` is reachable (either inside
  the cell dir or passed via `--bundle-path` explicitly).
- Prometheus is reachable via the Prometheus MCP server
  (`prometheus_mcp`). The `query_observability_knowledge_base` tool is used
  FIRST to confirm the DCGM metrics exist with the expected cardinality.
- The cluster's DCGM exporter is configured to export the
  `DCGM_FI_PROF_*` group. If not, the skill falls back to
  `DCGM_FI_DEV_*` counter-tier and flags the result with
  `dcgm_group_level: "counter"`.

## Workflow

### Phase 0 - pre-flight (knowledge-base probe)

```python
from tools.perf_tune_report.dcgm_correlate import (
    DcgmCorrelateInputs,
    correlate,
    read_sweep_window_from_bundle,
)
```

1. Resolve the bundle path + cell directory.
2. Load `configs/sol-ceilings.yaml`. **On a GB300 cluster use
   `hw_key=gb300_nvl72`, `n_gpus`=the deploy's TP (GB300 node = **4**, NOT 8), and the tensor
   `peak_key=nvfp4_dense_pflops` for NVFP4 weights** (`bf16_dense_pflops` / `fp8_dense_pflops`
   otherwise). Stamp these from the deploy, not from habit: the `b200_sm100` / `n_gpus: 8` /
   `bf16` defaults apply only to B200 clusters and will mis-scale a GB300/NVFP4 %SoL if reused.
3. Call `query_observability_knowledge_base` for the metrics listed
   in `dcgm_config.prof_group_probe_metrics`. Confirm:
   - Each metric exists on the target cluster.
   - Labels include at least `{namespace, pod, gpu, device}` (the
     default DCGM label set).
   - Cardinality is bounded (e.g. < 1000 series per metric across the
     target deploy).

### Phase 1 - sweep window

Read `(start_utc, end_utc)` either by calling
`read_sweep_window_from_bundle(bundle_path)` or from explicit
input (when the bundle's `captured_at` is the END of the sweep, not the
start - older bundles).

### Phase 2 - build queries (dry-run)

```python
inputs = DcgmCorrelateInputs(
    bundle_path=bundle,
    cell_dir=cell,
    sweep_start=start,
    sweep_end=end,
    hw_key="b200_sm100",          # GB300: "gb300_nvl72"
    pod_label_selector="app=basic-inference",
    namespace="inference",
    expected_n_gpus=8,            # GB300 node = 4 (the deploy TP), NOT 8
)

# Dry-run first to print the PromQL the correlator WILL fire:
result = correlate(inputs, ceilings, prom_client, dry_run=True)
for q in result.queries:
    print(q["peak_key"], "->", q["promql"])
```

### Phase 3 - execute the correlation

```python
result = correlate(inputs, ceilings, prom_client, dry_run=False)
out_path = write_correlation(result, inputs.cell_dir)
```

The result's `resources` list has one row per peak that mapped to a
DCGM metric, each with:

- `measured_bytes_total`, `measured_bytes_per_s` (bandwidth peaks) or
- `measured_tflops_avg` (compute peaks)
- `sol_pct` (the headline number)
- `notes[]` (short-sweep / missing-data flags)

### Phase 4 - patch sol-summary.md

Update the campaign's `sol-summary.md` "Workload-level SoL" table to
reference the byte-grounded numbers, replacing the previous
first-principles estimate. Cite the `dcgm_correlation.json` path so
future readers can re-derive the math.

### Phase 5 - re-render + re-publish to raise `sol_rigor` to L3

Emitting `dcgm_correlation.json` per cell is not the end - the campaign's
published lake row + report PDF only reflect the byte-grounding after a
**re-render then re-publish**:

```text
perftunereport report_render   --campaign <slug>            # draws pages 6 + 6b; sets dcgm_grounded + sol_rigor=L3
perftunereport publish_to_lake --campaign <slug> --if-exists overwrite
```

**Byte-grounding RAISES `sol_rigor` to L3 - it is RECORDED, not a gate
(always-publish policy).** A `sol_complete=true` campaign that is
`dcgm_grounded=false` (no `dcgm_correlation.json`, pages 6/6b absent) still
publishes at `sol_rigor=L1` (zymtrace proxy) - the gap is RECORDED on the
`campaign_v1` row + warned, never a refusal. `dcgm_grounded` + `sol_rigor` flow
`report_status.json` -> `report_render` envelope -> `campaign_v1` columns. Run
this skill (or the CLI verb below) for **every** plot-ready cell, then
re-render + re-publish so the lake row is `dcgm_grounded=true` / `sol_rigor=L3`
 - a tighter roofline. Pass `publish_to_lake --strict` only when you want an
`dcgm_grounded=false` campaign to refuse instead of land.

### Phase 5b - offline / CI path: the `dcgm_correlate` CLI verb + frozen snapshot

The live `correlate()` python path above needs a `PrometheusClient` (the
agent wires it to the Prometheus MCP). For an offline / re-runnable /
CI context that cannot reach Prometheus, capture the DCGM means into a
**frozen YAML** (schema `dcgm_frozen_v1`) once, then fold it in
deterministically with the CLI verb:

```text
perftunereport dcgm_correlate --campaign <slug> --cell-id <cell> \
  --frozen-yaml <cell>/dcgm-frozen.yaml \
  [--kernels-json <cell>/kernels.json]   # default: the cell's own kernels.json (page 6b)
```

This wraps `correlate_from_frozen` + `write_correlation` and is the path the
campaign orchestrator's `step_dcgm_correlate` runs (it consumes a
`cells/<id>/dcgm-frozen.yaml` per cell). **Always snapshot a frozen YAML even
when you used the live path**, so the byte-grounding is reproducible offline
and survives deploy teardown (the DCGM time-series may age out of Prometheus
retention, but the frozen means do not).

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

This skill IS the byte-grounded SoL producer. Its output is the
authoritative third-tier evidence per `docs/METHODOLOGY.md`
"Speed-of-light framing":

- Peaks live in `configs/sol-ceilings.yaml`. This skill
  reads them by key (`b200_sm100.hbm3e_tbps`,
  `gb300_nvl72.nvfp4_dense_pflops`, etc.) - never inline.
- DCGM metric anchors live in the same YAML under each peak's
  `dcgm_metric` / `dcgm_metrics_bytes` / `dcgm_fallback_metric` fields.
- The renderer's page 6 (`dcgm_sol.py`) consumes the emitted
  `dcgm_correlation.json` and draws workload-level resource bars
  showing measured-vs-peak × duration × n_gpus.
- This skill's output drives the `dcgm_grounded` flag + the campaign's
  `sol_rigor`: with a `dcgm_correlation.json` the campaign is
  `dcgm_grounded=true` / `sol_rigor=L3` (or `L4` if ncu is also present),
  without one it is `dcgm_grounded=false` / `sol_rigor=L1`. Under the
  **always-publish policy this is RECORDED, not a gate** - `publish_to_lake`
  lands the campaign either way, with the gap on the `campaign_v1` row + a loud
  warning. Run this skill per cell to raise rigor to L3 (a tighter roofline),
  pass `publish_to_lake --strict` only when you want an ungrounded campaign to
  refuse instead of land.

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

## Safety

- **Read-only.** Every Prometheus call is a read. No mutation of any
  cluster state.
- **Knowledge-base FIRST.** Skill MUST call
  `query_observability_knowledge_base` before any `query_prometheus`
  to confirm cardinality bounds. This is the standing
  prometheus-anchored-query pattern.
- **Bundle write only inside the supplied cell directory.**
  No writes outside `<campaign>/cells/<cell>/`.
- **Provenance preserved.** Every PromQL invocation is recorded in the
  output's `queries` array. The `dcgm_correlation.json` carries
  `schema_version`, `sweep_start_utc`, `sweep_end_utc`, `n_gpus`,
  `dcgm_group_level`. It also carries the
  re-query provenance `nodes` (distinct host(s) the DCGM series carried),
  `namespace`, and `pod_label_selector`. `DCGM_FI_DEV_POWER_USAGE` is
  **per-node**, so without the node a window-only capture cannot be
  re-queried later for `tokens_per_watt` - the live `correlate()` now
  auto-captures the node from the series labels (`Hostname` / `node` /
  `exported_node` / `instance`), and a frozen YAML SHOULD record
  `nodes:` (+ optional `namespace:` / `pod_label_selector:`) so the
  byte-grounding stays re-queryable after the time-series ages out.

## Pairs with

- [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)
 - per-kernel arithmetic intensity. Run BOTH for a complete SoL
  picture: ncu surfaces the per-kernel %SoL, this skill surfaces the
  workload-level %SoL aggregated over the same window.
- [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) - the
  drive_load.py sweep that produces the window this skill correlates
  against.
- [`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md)
 - the general anchored-PromQL primitive. This skill is
  the DCGM-specific specialisation.
- [`analyze-zymtrace-workload`](/plugins/profile-and-optimize/skills/analyze-zymtrace-workload/SKILL.md)
 - the time-share proxy view (level 1 of SoL hierarchy). This skill
  is its level-3 upgrade.

## Source-of-truth references

- Tool: [`server/tools/perf_tune_report/dcgm_correlate.py`](/plugins/profile-and-optimize/server/tools/perf_tune_report/dcgm_correlate.py).
- Tests: [`server/tools/perf_tune_report/test_dcgm_correlate.py`](/plugins/profile-and-optimize/server/tools/perf_tune_report/test_dcgm_correlate.py)
 - fake-Prometheus-client coverage of ratio/byte-rate aggregation,
  PROF/counter/absent fallback, and short-sweep warning.
- Renderer page 6: `server/tools/perf_tune_report/renderer/dcgm_sol.py`
  (consumes the emitted `dcgm_correlation.json`).
- `docs/METHODOLOGY.md` "Speed-of-light framing" - the standing
  three-level rigor hierarchy this skill operationalises.

## Contact

Open an issue on this repository.
