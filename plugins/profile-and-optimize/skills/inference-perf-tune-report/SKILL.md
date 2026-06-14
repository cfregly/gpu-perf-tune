---
name: inference-perf-tune-report
last_validated: 2026-05-24
description: >-
  Build a multi-page benchmark report PDF (scatter panels + per-concurrency
  heatmap tables) from `vllm bench sweep serve_workload` and/or AIPerf outputs,
  in the GLM-5.1 reference layout (5x2 scatter facet by max_num_batched_tokens +
  3x2 heatmap tables for C={8,16,32} x {tok/s/GPU, TTFT}. First-class
  failed/partial cells). Also renders the cross-engine (vLLM + SGLang) CHAMPION
  page 8: baseline vs top-X variants + the overlaid roofline + a
  RECOMMENDED-FOR-PRODUCTION pick via `champion_select`. Backed by the
  `perf_tune_report` CLI library. Triggers on "build a perf report", "render the atlas
  PDF", "perf-report campaign", "glm5p1-style report", "pareto report",
  "perftunereport smoke", "champion selection", "which variant to ship", "production
  champion", "baseline vs top variants", or any combination of "perf / atlas /
  report / champion" with "inference / vllm / sglang / aiperf / b200 / gb300 /
  nvfp4 / fp8".
allowed-tools:
  - mcp__profile_and_optimize__perf_tune_report_campaign_init
  - mcp__profile_and_optimize__perf_tune_report_cell_run
  - mcp__profile_and_optimize__perf_tune_report_import_variant_ab
  - mcp__profile_and_optimize__perf_tune_report_atlas_aggregate
  - mcp__profile_and_optimize__perf_tune_report_dcgm_correlate
  - mcp__profile_and_optimize__perf_tune_report_import_roofline_sweep
  - mcp__profile_and_optimize__perf_tune_report_champion_select
  - mcp__profile_and_optimize__perf_tune_report_report_render
  - mcp__profile_and_optimize__perf_tune_report_report_smoke
  - mcp__profile_and_optimize__perf_tune_report_publish_to_lake
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Bash(perftunereport:*)
  - Bash(kubectl:*)
  - Bash(vllm:*)
  - Bash(open:*)
  - Read
  - Write
---

# inference-perf-tune-report

## Purpose

Build a multi-page benchmark report PDF -- scatter panels + per-concurrency
heatmap tables -- from inference benchmark sweep output. Mirrors the
GLM-5.1 reference layout:

> **Steady-state window:** the report only renders what the sweep measured. A throughput
> sweep cell needs `num_prompts >= 2*c` or its high-c throughput is undercounted
> ~1.6-1.8x (ramp/drain-dominated window, `docs/METHODOLOGY.md` trap 4). The
> `import_roofline_sweep` importer flags `steady_window_undercount` + WARNs on any cell
> `< 2c` - heed it before publishing the report.

- **Page 1**: 5 rows (one per `max_num_batched_tokens` in {1024, 2048, 4096,
  8192, 16384}) x 2 columns of scatter panes -- left=TTFT vs request
  throughput, right=tok/s/user vs tok/s/GPU. Inline-labels selected
  profiling concurrencies (default {1, 8, 32}). Color encodes (hardware,
  TP). Circle marker = EP, X marker = TP. Hollow markers indicate MTP.
- **Page 2**: 3 rows (C in {8, 16, 32}) x 2 columns of heatmap tables
  showing output tok/s/GPU and TTFT avg (ms) for every (variant, mbt)
  combo, with gray cells for missing / failed / partial measurements.

The renderer is workload-agnostic. It ingests any
[`AtlasCell`](/plugins/profile-and-optimize/server/tools/perf_tune_report/schema.py)-shaped JSONL.
Backends (vllm-sweep + AIPerf) normalize into the same schema.

## When to use

- Comparing inference performance across multiple hardware generations
  (e.g. H100 FP8 vs B200 NVFP4 vs GB300 NVFP4) for the same model.
- Comparing parallelism strategies (TP vs EP) and speculative-decoding
  variants (MTP on / off) on a single deployment.
- Re-rendering a stakeholder-grade report from previously captured
  benchmark data without re-running the cluster sweep.
- Producing a release-day PDF for a new model variant before promoting it
  to staging / prod.

Do **not** use this skill for:

- Single-deployment, single-concurrency snapshot benchmarks -- the
  upstream [`perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) skill is the
  right tool. Use this skill once you have multiple atlas cells to
  compare side-by-side.
- Training-benchmark reports (step-time / MFU) -- out of scope for this
  plugin.
- Quality / accuracy evaluation -- use
  [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md).

## Example prompts

- "Build a perf report comparing Kimi K2.6 on B200 NVFP4 vs GB300 NVFP4."
- "Render the atlas PDF for the glm51-multihw campaign."
- "Run a glm5p1-style report across H100/B200/GB300 with TP and EP for
  each."
- "Smoke-render the perf-report PDF from the synthetic fixture so I can
  see the layout before committing cluster time."
- "perftunereport campaign_init configs/campaigns/kimi-k26.yaml"
- `/inference-perf-tune-report --campaign glm51-multihw`

## Prerequisites

1. **profile-and-optimize bundled MCP server installed** (the standard `bash
   ~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/<ver>/server/install.sh`
   or, from a repo checkout, `bash plugins/profile-and-optimize/server/install.sh`).
   The `perftunereport` console script is installed into the same venv as
   `profile-and-optimize-mcp`.
2. **`matplotlib>=3.8` + `pandas>=2.0`** in the profile-and-optimize venv. Install
   the optional extra: `.venv/bin/pip install -e '.[perf_tune_report]'`.
3. **For `cell_run` only**: live cluster access (`kubectl` context +
   namespace + bench pod) AND the same prerequisites as the upstream
   [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) skill
   (HF_TOKEN, replay-playback dataset access).
4. **No cluster needed** for `campaign_init`, `atlas_aggregate`,
   `report_render`, or `report_smoke` -- those operate on already-captured
   data or the bundled synthetic fixture.

## Workflow

### Phase A: smoke-render the layout (no cluster, ~2 seconds)

Always do this first when introducing a new operator to the skill. It
proves the renderer is healthy and lets the operator see the output
shape without committing benchmark time.

```text
/inference-perf-tune-report smoke --out /tmp/perftunereport-smoke.pdf
```

or via the MCP tool:

```text
perf_tune_report_report_smoke --out /tmp/perftunereport-smoke.pdf
```

The bundled synthetic fixture mirrors the GLM-5.1 PDF coverage exactly:
40 atlas cells, 38 full sweeps, 1 partial, 1 failed, 232 plot-ready
points, 20 evicted MTP context cells.

### Phase B: scaffold the campaign

Author a campaign YAML under
`configs/campaigns/<slug>.yaml` (or any
operator-chosen location). Each cell entry must include `cell_id`,
`model`, `hardware`, `quant`, `tensor_parallel`, `parallel_strategy`,
`mtp`, `max_num_batched_tokens`, `concurrencies`, plus backend-specific
extras under `vllm_sweep:` and / or `aiperf:`.

```text
perf_tune_report_campaign_init --config configs/campaigns/<slug>.yaml
```

This scaffolds `./campaigns/<UTC-ts>-<slug>/`
with `SOURCE.md`, `summary.md`, frozen `config.yaml`, and empty
`cells/` + `commands/` subdirs. Returns the campaign dir in the JSON
envelope.

### Phase C: run each cell

For each cell in the campaign config, drive `cell_run` with the
operator-chosen backend. **This verb is ack-gated (`safety=submits_jobs`)**
-- the cockpit requires the explicit
`i_understand_this_submits_jobs=true` parameter (or the CLI flag
`--i-understand-this-submits-jobs`) before any actual benchmark runs.
`--dry-run` lets you see the generated shell commands without
consuming cluster time.

```text
# vllm-sweep backend
perf_tune_report_cell_run \
  --campaign <slug> \
  --cell h100-fp8-tp16-ep-mbt4096 \
  --backend vllm-sweep \
  --i-understand-this-submits-jobs

# aiperf backend (same campaign, different cell)
perf_tune_report_cell_run \
  --campaign <slug> \
  --cell b200-nvfp4-tp8-ep-mbt4096 \
  --backend aiperf \
  --namespace <namespace> \
  --bench-pod my-perf-bench \
  --kube-context <cluster> \
  --endpoint-url http://kimi-k26-svc.<namespace>.svc.cluster.local:8000 \
  --served-model moonshotai/Kimi-K2.6 \
  --i-understand-this-submits-jobs
```

Each cell writes its raw backend output under
`<campaign>/cells/<cell-id>/raw/` and a normalized
`<campaign>/cells/<cell-id>/normalized.json` consumed by the next phase.
Cell `status.txt` is set to one of `full` / `partial` / `failed` based
on whether all requested concurrencies completed.

For multi-cell campaigns, loop the operator through each cell -- the
sweeps are long-running and benefit from explicit human consent per
cell (e.g. checking cluster capacity before each launch).

### Phase D: aggregate into the atlas

```text
perf_tune_report_atlas_aggregate --campaign <slug>
```

Unions every cell's `normalized.json` into one
`<campaign>/atlas.jsonl` and returns the Coverage summary block that
will appear on page 1 of the rendered PDF (e.g.
`"Coverage: 40 atlas cells | 38 full sweeps | 1 partial sweeps |
1 failed cells | 232 plot-ready concurrency points"`).

### Phase D1: TPM supported across hardware types (pricing rollup)

After aggregation, roll the atlas into a per-hardware **tokens-per-minute (TPM)**
capacity table for pricing / capacity discussions. This is a pure
post-processing step on already-measured data (no cluster runs).

```text
# peak-capacity only:
perftunereport tpm_summary --campaign <slug>

# add a latency-SLA-bounded operating point (the customer-commitment number):
perftunereport tpm_summary --campaign <slug> --ttft-sla-ms 500 --tpot-sla-ms 50
```

Writes `tpm_summary.{json,csv,md}` to the campaign dir. For each
`(model, hardware, quant, TP, strategy, MTP)` group it reports a **peak** point
(max output tok/s/GPU -- warm sweep best-case) and, when `--ttft-sla-ms` /
`--tpot-sla-ms` are supplied, an **sla** point (highest tok/s/GPU meeting the
thresholds), each at **per-GPU / per-replica (=TP GPUs) / per-node**
(`--gpus-per-node`, default 8) bases, for both **output-only** and **total**
(input+output) TPM. Total-TPM is `n/a` for backends that emit no total-token
line. `report_render` also draws a peak-only "TPM supported across hardware
types" PDF page, and `publish_to_lake` lands a `tpm_v1` table.

Report every TPM number with the warm/cold + ISL/OSL caveat the summary header
carries (per `server/AGENTS.md` "Benchmark methodology hygiene"): peak is a
warm best-case, the SLA point is the customer-commitment number.

**Defaults:** the SLA point + cost columns populate for EVERY
campaign without a config block -- `discover_tpm_config` falls back to a default
SLA (TTFT<=2000ms / TPOT<=50ms / `gpus_per_node=8`) and a default cost table
(representative public on-demand list rates per GPU: H100 $6.16 / H200 $6.31 /
B200 $8.60. GB300 unset). Declare a `tpm:`/`cost:`
block in the campaign `config.yaml` only to override (per field. Cost is overlaid).
`report_render` and `publish_to_lake` read it via `discover_tpm_config`, so all
three surfaces emit the same peak + sla points. The SLA point is still left unset
(not invented) for a group whose rows lack a decode metric (TPOT/ITL) at/under
the SLA -- a throughput-only bench has no decode metric to derive an SLA point
from, so re-run the bench with decode capture if SLA-TPM is needed.

**Cost + analysis carry-through.** The `cost:` block (`usd_per_gpu_hour: { B200: 4.50, ... }`)
overrides/extends the default cost table behind the `$/1M tok` PDF column and the
`cost_v1` economics/TCO lake table (`$/1M output|total`, plus `tokens_per_watt` +
`power_watts_per_gpu` when DCGM power is captured -- a `cost:` block cannot
synthesize tokens-per-watt). Default rates: H100/H200/B200 = representative public
list rates, **GB300 = $12.00/GPU-hr ESTIMATE (no public list rate)**. To
derive `tokens_per_watt` the cell's DCGM power capture must record BOTH the bench
window AND the node (per-node `DCGM_FI_DEV_POWER_USAGE`). A window-only artifact
cannot be re-queried later. The atlas
also carries per-cell `mean_input_tokens`/`mean_output_tokens` (ISL/OSL, derived
from the bench `Total input/generated tokens` lines -> `tpm_v1.mean_isl/mean_osl`),
`cache_mode` (declared via `import_perf_bench --cache-mode warm|cold`), and
`prefix_cache_hit_rate` (from the bundle metadata). For `tokens_per_watt`, add
`power_watts_per_gpu` to the cell's `dcgm_frozen_v1` YAML before `dcgm_correlate`.

### Phase D2: byte-ground every cell to raise sol_rigor (recommended, not a gate)

DCGM byte-grounding raises a campaign's `sol_rigor` from `L1` (zymtrace proxy)
to `L3` (byte-grounded) - it is **recorded, not gated** (the
always-publish policy: a `dcgm_grounded=false` campaign still publishes, with
the gap visible on the `campaign_v1` row). For each plot-ready cell, produce a
`dcgm_correlation.json` (renderer pages 6 + 6b) via the
[`inference-dcgm-correlate`](/plugins/profile-and-optimize/skills/inference-dcgm-correlate/SKILL.md)
skill (the live Prometheus MCP path) OR the offline CLI verb against a frozen
DCGM snapshot:

```text
perf_tune_report_dcgm_correlate --campaign <slug> --cell-id <cell> \
  --frozen-yaml <dcgm-frozen>.yaml
```

Capture DCGM (SM/DRAM/tensor/GR + NVLINK bytes) concurrently with each cell's
bench window so the means are real (see the experiment-isolation rule). A run
that skips it publishes at `sol_rigor=L1` (`dcgm_grounded=false`) - valid and
comparable, just less tight. Prefer L3/L4 when DCGM/ncu are available.

### Phase D3: prefill/decode roofline (page 7) - always-on

The phase-separated roofline + per-(c,ISL) DCGM utilization (the "what
concurrency maxes the TFLOPs" + "is decode >=75% HBM bandwidth" questions, always
wanted). Capture with the gated `*-deploy/profiling/roofline-sweep.sh` (a
decode-concurrency sweep + a prefill-ISL sweep, each with per-cell in-pod
`dcgmi dmon` PROF - SM/tensor/DRAM/NVLINK), then ingest:

```text
roofline-sweep.sh <ns> <pod> <out> <model> <tokenizer> \
  "1 2 4 8 16 32 64 128 192" "512 1024 2048 4096 8192" <container>
perf_tune_report_import_roofline_sweep --campaign <slug> --bundle <out> \
  --hardware GB300 --tensor-parallel <tp> --quant NVFP4 --cache-mode cold
```

This writes `cells/<id>-decode` + `<id>-prefill` rows (per-(c,ISL) DCGM +
analytical roofline coords in `extra` -> `atlas_v1.extra_json` + the `roofline_v1`
lake table) + a `cells/<id>-decode/roofline_sweep.json` (carrying the embedded
analytical `ModelShape`) the renderer turns into **page 7**. Page 7 sets
`sol_complete` + `dcgm_grounded` + `sol_rigor=L3`. The per-(c,ISL) in-pod `dcgmi`
data is what the workload-level `dcgm_correlate` window-mean (Phase D2) cannot
produce. Import multiple configs (TP2/TP4/TP8, fp8-KV vs NVFP4-KV, baseline vs
optimized, vLLM vs SGLang, ...) into one campaign and page 7 overlays them.

**Methodology (`server/tools/perf_tune_report/ROOFLINE-METHODOLOGY.md`):** page-7
panel A plots `x = analytical arithmetic intensity (FLOP/byte)` (from the model's
`config.json` via `roofline_math.py`) and `y = flop_per_token x measured tok/s /
n_gpus` against PER-GPU ceilings, so a decode point's vertical gap below the
memory diagonal IS its HBM utilization. Panel B (Q2) plots BOTH the DCGM
`DRAM_ACTIVE` duty-cycle proxy AND the byte-grounded delivered-HBM-BW % (honestly
labeled) + the 75% line. For a model not in the built-in registry, pass
`--model-config <config.json>` (or let `roofline-sweep.sh` capture the in-pod
`model_config.json`) so the analytical axis engages instead of the DCGM-proxy
fallback. **Page 7 is now a strict-publish gate:** a throughput/mixed serving
campaign with plot-ready points but no page 7 is REFUSED under `publish_to_lake
--strict` (the default, `--no-strict` records the gap). The rendered PDF also
carries a **"Source under test"** page (vLLM/SGLang commit + delivery + infr patch
+ GitHub URL) from the bundle's `experiment_provenance_v1` block + `source-registry.yaml`.
The campaign's recorded `delivery` is the code-under-test identity: a number
from this campaign may be cited as evidence only for THAT delivery, never cross-tier (an
`overlay`/offline-prepped run is not evidence for an `infr-patch`, even if the kernels match).

### Phase D4: champion selection (page 8) - the production pick

The "what do we ship" synthesis. When the campaign holds a baseline + variant
arms (e.g. a cross-engine A/B imported via `perf_tune_report_import_variant_ab` from a
`run-variant-ab.sh` bundle: one engine-tagged `vllm-sweep` / `sglang-sweep` cell
per arm), `champion_select` ranks the baseline + top-X CROSS-ENGINE under the
focus metric + a TPOT SLO, summarizes each across the 4-layer SoL ladder, overlays
their rooflines, and emits the production recommendation:

```text
perf_tune_report_champion_select --campaign <slug> --top 3 \
  --focus <throughput|latency> --workloads-present aa,sonnet,sharegpt,random,code \
  --accuracy-gate pass --same-node --trials 3
```

It writes `CHAMPION.md` + `champion_select.json` (the artifact **page 8**
consumes -- a baseline-vs-top-X table with per-variant sol_rigor + HBM/tensor/SM%,
the overlaid roofline, and a RECOMMENDED-FOR-PRODUCTION banner) and, on publish,
the `champion_v1` per-variant lake table + `campaign_v1.recommended_cell`. The
recommendation is **DRAFT** unless the champion passes the variance (`--same-node`
+ `--trials>=3`), multi-workload (`--workloads-present` covers the canonical
suite), and accuracy (`--accuracy-gate pass`) gates AND is L3 byte-grounded --
then it is a **VERDICT**. This is the deliverable that makes the prod choice
obvious. Run it before Phase E so page 8 lands in the PDF.

### Phase E: render the PDF

```text
perf_tune_report_report_render \
  --campaign <slug> \
  --variants-line "Variants: GLM-5.1-FP8 (FP8, H100) | GLM-5.1-NVFP4 (NVFP4, B200) | GLM-5.1-NVFP4 (NVFP4, GB300)" \
  --data-source-line "Data source: replay-playback 2025_07 | prompt 32k input | OSL 4k"
```

`report_render` always renders + records `sol_complete` / `focus` / `sol_rigor`
(it never refuses), `publish_to_lake --strict` is the opt-in gate if you want
publish to refuse a `dcgm_grounded=false` or otherwise incomplete campaign.

Default output path is `<campaign>/report.pdf`. Override with `--out`.

The renderer embeds UTC provenance into the PDF (`CreationDate`/`ModDate`
metadata + `Keywords` + a page-1 footer carrying `campaign=<run-id>`,
`rendered=<...Z>`, and the bench-capture window). All of these are UTC.
The PDF's OS file **mtime is local wall-clock and is NOT authoritative** --
on a UTC-7 workstation a render at `22:41` local reads a calendar day
behind its `YYYYMMDDTHHMMSSZ` run-id. Cite the run-id or the embedded
`rendered=<...Z>` stamp, never the mtime.

### Phase F: share or archive

A campaign directory is fully self-contained
(`SOURCE.md` + frozen `config.yaml` + `cells/` + `atlas.jsonl` +
`report.pdf` + per-shell `commands/` capture). Share via
`tar czf <slug>.tar.gz <campaign>/` to any teammate who has profile-and-optimize
installed.

## Ack-gating behavior

`cell_run` is the only ack-gated verb (`safety=submits_jobs`). The
cockpit will:

1. Refuse the call if `i_understand_this_submits_jobs` is not set to
   `true`.
2. Print a FATAL message naming the missing ack field and suggesting
   `--dry-run` as the safe alternative.
3. Exit with code 2 (non-zero) so the cockpit treats it as a refusal,
   not a silent skip.

All other verbs (`campaign_init`, `atlas_aggregate`, `report_render`,
`report_smoke`) write only to local disk under
`./campaigns/<slug>/` (operator-relocatable via
`PERFREPORT_CAMPAIGNS_DIR`) and are not ack-gated.

## Campaign-storage location

The default campaigns root is
`./campaigns/` -- a local-only directory that holds campaign
artifacts. It is
gitignored so per-campaign output never accidentally
escapes the workstation. Source code for the renderer / runners /
schema lives in **this** plugin (`server/tools/perf_tune_report/`),
preserving profile_and_optimize's
single-clone-self-contained invariant.

Override the campaigns root via the `PERFREPORT_CAMPAIGNS_DIR` env var
or the `--campaigns-dir` CLI flag if you want campaigns to land
elsewhere (e.g. on a shared filesystem).

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

The rendered PDF carries up to **four pages**, the last of which is the
Speed-of-Light Roofline page. It draws automatically when:

1. The campaign has at least one valid `cells/*/kernels.json` (page 3
   precondition. Zymtrace per-category data). A zymtrace L1 empty-now right after
   the window is ClickHouse ingest lag, not absence - poll and requery
   after the flush (see
   [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md)).
2. The renderer finds `configs/sol-ceilings.yaml` on the
   ancestor path (or `SOL_CEILINGS_YAML=/path` env var override).
3. The atlas's `hardware` field maps to a key in the YAML (`B200` →
   `b200_sm100`, `GB300` → `gb300_nvl72`, `H100` → `h100_sxm`).

When ANY of the three is absent the page is **not** silently skipped.
The omission is recorded in `report_status.json`, surfaced on a
loud "Report completeness" page in the PDF (with the exact reason + how to
populate it), printed as a `WARNING:` on stderr, and returned in the
`report_render` JSON `render_status` block (`sol_complete`, `focus`,
`sol_rigor`, `omitted_pages`). `sol_complete=true` when ANY SoL page (4 zymtrace
/ 5 ncu / 6 DCGM) renders, `sol_rigor` records the highest level present.
Publish lands the row regardless (always-publish policy). Pass
`publish_to_lake --strict` only when you want a missing roofline or 0
throughput-scatter points (non-`latency` focus) to be a hard refusal. When the
YAML is found but malformed the renderer still raises `SoLCeilingsMalformed`
and aborts - same no-silent-degradation contract as `KernelsJsonMalformed`.

Per `server/AGENTS.md` "Speed-of-light framing", every campaign
SHOULD also carry a `<campaign>/sol-summary.md` doc with the
workload-level HBM-roofline calc and a link to the relevant grounding
doc. The SoL summary
is operator-written today. The page-4 visualisation is the auto-rendered
companion.

## Asset validation (review + FAIL LOUD)

The report PDF this skill renders (`report_render` -> report.pdf: scatter / heatmap / roofline /
champion pages) is a DELIVERABLE held to `server/AGENTS.md` "Validate every generated asset"
(`docs/METHODOLOGY.md`): the renderer **FAILS LOUDLY** on missing/bad
data (`SoLCeilingsMalformed` / `KernelsJsonMalformed` raises, the degenerate-PDF guard that raises on
a <10KB report, and the `methodology_problems` `--strict` gate that refuses an incomplete/`unknown`
descriptor), and the agent **REVIEWS** the rendered PDF -- opens it, confirms every panel is accurate
(numbers + identities + matched comparisons trace to the campaign. Curves physically plausible,
failed/partial cells shown as such. Nothing mislabeled) -- and **rebuilds** it if wrong. Never ship a
broken/confusing report with a caveat.

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

Per `server/AGENTS.md` "Verdict rigor: DRAFT vs VERDICT", set the campaign's
`verdict_tier` honestly. Default **draft**. Set **verdict** only for a decision-grade
campaign that is variance-controlled (same-node, >=3 trials, mean +/- std),
metric-isolated (median TPOT/ITL for decode-latency claims), against a
production-representative baseline, and (for which-kernel claims) backed by nsys/ncu
per-kernel data. Under the always-publish policy a `verdict`-tier campaign
missing this provenance is **auto-downgraded to `draft` and still lands** (the
honest tier is recorded on `campaign_v1`). Pass `publish_to_lake --strict` if you
want the unsupported verdict claim to refuse instead.

## Experiment isolation & traceability (mandatory)

This pipeline is the canonical "results -> rooflines -> perf-lake" path the
`server/AGENTS.md` "Experiment Isolation & Traceability" rule
(and `docs/METHODOLOGY.md`) requires every measurement
experiment to use:

- **Campaign id = experiment-id** (the evidence-bundle run-id), so the
  published `campaign=<id>` rows in `atlas_v1` / `campaign_v1` join back to the
  cluster objects (label `experiment=<id-slug>`) and the evidence bundle.
- **`cell_run` cells that submit cluster workloads MUST use experiment-unique
  serve names + the `experiment=<id-slug>` label** and MUST NOT reuse
  standing/platform/migration names (forbidden list in the `server/AGENTS.md` rule).
  Cluster-scoped PV names are global. A collision silently breaks another
  owner's PVC binding.
- **`publish_to_lake` is mandatory**, not optional - a campaign is "done"
  once the atlas + campaign rows are written AND `report.pdf` contains a
  Speed-of-Light roofline page. Capturing DCGM + zymtrace during the cells
  raises the roofline rigor (see the Speed-of-light reporting section + Phase D2).
- **Always-publish with focus + sol_rigor.** EVERY run publishes a
  `sol_complete` roofline with a recorded `focus` (latency|throughput|mixed) +
  `sol_rigor` (`L4` ncu | `L3` DCGM | `L1` zymtrace-proxy | `none`).
  `publish_to_lake` **never refuses** by default - a `dcgm_grounded=false` /
  latency-bound / proxy / no-SoL / 0-plot-ready run lands with the gap RECORDED
  on `campaign_v1` + warned. An unsupported `verdict_tier=verdict`
  auto-downgrades to `draft`. The one hard requirement is that `report_render`
  ran first. Pass `--strict` only when you want publish to refuse an incomplete
  campaign. Run `perf_tune_report_dcgm_correlate` (or the `inference-dcgm-correlate`
  skill) per cell (Phase D2) to raise `sol_rigor` to L3 - it raises rigor, it is
  not a gate.
- Tear down cell workloads by label and verify standing/migration objects are
  untouched. Record the campaign-id + `s3://perf-lake/...` paths in the parent
  evidence bundle.

## Cross-references

- [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) -- the
  per-deployment AIPerf runbook the `aiperf` backend wraps. Use it when
  you only need to benchmark one deployment with one config. Come here
  when you have an atlas to compare.
- [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md) /
  [`perf-baseline-diff`](/plugins/profile-and-optimize/skills/perf-baseline-diff/SKILL.md) -- record a
  campaign's atlas.jsonl as a baseline to regression-check future
  campaigns against.
- [`evidence-bundle-init`](/plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md) -- scaffold
  a parent evidence bundle if this report is part of a wider
  investigation (e.g. a deploy-regression post-mortem).
