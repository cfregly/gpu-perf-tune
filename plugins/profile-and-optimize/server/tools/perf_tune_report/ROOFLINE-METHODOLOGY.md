# Prefill + Decode Roofline Methodology (perf_tune_report page 7)

Canonical, model-agnostic methodology for the prefill/decode roofline the
`perf_tune_report` renderer emits as **page 7** and that the Superset
`fact_perf_tune_report_roofline` dashboard renders. This is the single source of
truth. The renderer (`renderer/prefill_decode_roofline.py`), the analytical math
(`roofline_math.py`), the importer (`importers/roofline_sweep.py`), and the
serving-side `roofline-sweep.sh` capture all implement it.

It answers, per (model, config), the four questions an inference perf reviewer
always asks:

- **Q1 "what concurrency C maxes the TFLOPs?"** -> Panel C (tensor + SM active
  vs C). Finding pattern: decode never saturates compute (it is memory-bound),
  prefill is the phase that uses the tensor cores.
- **Q2 "decode should hit ~75% HBM-BW utilization"** -> Panel B (HBM util vs C,
  with the 75% reference line). Finding pattern: dense decoders can approach it,
  sparse-MoE + MLA decoders plateau lower, and at low C the GPU is host-gap
  idle, so HBM is far below 75%.
- **Q3 "are we using the optimal sharding degree? (the first lever)"** -> overlay
  TP2/TP4/TP8 (and DP) on one per-GPU roofline. The per-GPU normalization makes
  the comparison fair.
- **Q4 "roofline for prefill AND decode for every model and config"** -> this page,
  always generated on every measurement bench.

## The two axes (measured operating point vs analytical intensity)

A Williams roofline plots a **measured** operating point against an
**analytically-derived** arithmetic intensity and a **datasheet** ceiling. The
perf NUMBER is measured. The intensity coefficient is a property of the
algorithm + shapes (analytical), exactly like every published roofline. This is
consistent with the `docs/METHODOLOGY.md` rule that all performance numbers
must come from actual performance runs: the y-coordinate (tok/s) is a real bench number,
the x-coordinate (FLOP/byte) and the FLOP/token coefficient are derived from the
model's `config.json`, not measured, and are cross-checked against the measured
DCGM tensor/dram active ratio.

- **x = arithmetic intensity (FLOP/byte)** from `roofline_math.ModelShape`:
  - decode: `flop_per_token / (active_weight_bytes(expert_union)/C + kv_bytes(ctx))`
    -- weight bytes amortize across the batch (the expert union
    `min(n_routed, n_experts_per_tok*C)` is loaded once, reused by every token),
    KV bytes scale per token with context. AI grows ~linearly with C but stays
    far LEFT of the ridge for any feasible serving batch -> memory-bound.
  - prefill: `flop_per_token / (active_weight_bytes(all_experts)/T)` -- a chunk
    of T tokens reuses each weight load T times, so AI ~ T -> climbs toward the
    ceiling (compute-bound).
- **y = achieved compute per GPU (TFLOP/s)** = `flop_per_token * measured_tok/s / n_gpus`
  (decode: output tok/s. Prefill: input tok/s). Per-GPU so TP2/TP4/TP8 share one
  ceiling. `flop_per_token = 2 * active_params(experts_per_tok)` (weight-GEMM
  FLOPs. Attention FLOPs excluded -> a conservative lower bound on prefill
  achieved-compute, by construction).
- **ceiling (per GPU)** from `configs/sol-ceilings.yaml` by quant:
  compute roof = `nvfp4|fp8|bf16_dense_pflops`. HBM slope = `hbm3e_tbps`. Ridge =
  `compute_peak / hbm_peak` (GB300 NVFP4 = 1875 FLOP/byte). NEVER the aggregate
  `peak * n_gpus` -- per-GPU only.

## Panel B -- HBM-BW utilization (Q2), honestly labeled

Two estimates are plotted. Neither is silently passed off as the other:

1. **DCGM `DRAM_ACTIVE` duty-cycle** (`dcgmi dmon` field 1005, 0..1): the fraction
   of cycles the HBM interface was transferring. The canonical hardware proxy for
   memory-pipe utilization and an upper bound on delivered-BW/peak. Labeled
   "DRAM-active % (DCGM)".
2. **Byte-grounded delivered-BW %** = `(weight_bytes/token + kv_bytes) * tok/s / n_gpus / hbm_peak`.
   The analytical bytes/token times the measured tok/s, over the datasheet peak.
   Labeled "delivered HBM-BW % (analytical bytes x measured tok/s)".

The 75% reference line is drawn. Where the Prometheus MCP DCGM window is still
retained, the workload-level byte-grounded L3 `dcgm_correlate` (page 6) is the
third cross-check. Calling DRAM_ACTIVE "HBM-BW util" without the byte-grounded
overlay is the classic defect this methodology prevents.

## Panel C -- compute utilization (Q1)

DCGM `PIPE_TENSOR_ACTIVE` (field 1004) and `SM_ACTIVE` (1002) vs C. Decode tensor%
stays low + flat (memory-bound). SM% climbs as concurrency amortizes the host
gap. Prefill tensor% is 5-10x decode's -- prefill is the compute phase.

## Rigor ladder (which tier page 7 is)

Page 7 is **L3** (DCGM byte/FLOP-grounded, per-(C, ISL)). It complements:
`L1` zymtrace per-category time-share proxy (page 4) < `L2` zymtrace x DCGM
(page 6b) < `L3` DCGM workload byte-traffic (page 6) + per-point sweep (page 7)
< `L4` ncu per-kernel measured arithmetic intensity (page 5). Page 5 is the only
tier with a MEASURED (not analytical) per-kernel AI. Page 7's workload AI is
analytical-x / measured-y. Cite the tier. Never withhold a lower tier.

## Observations vs mechanisms (mandatory companion)

Every roofline ships with a findings pair (`evidence-bundle-init`
scaffolds it):

- `01-observations.md` -- measured tables ONLY (DCGM %, tok/s, TPOT). No "why".
- `02-mechanisms.md` -- each item `OBSERVATION -> MECHANISM (causal) -> CONFIDENCE`.

A mechanism claim (e.g. "decode plateaus at 41% HBM because the sparse-MoE+MLA
kernel mix has low DRAM efficiency") needs a profile (DCGM/zymtrace/nsys/ncu) per
the "attribution needs profiles" rule -- the rooflines + DCGM panels are the
observation. The mechanism is the separately-evidenced interpretation.

## Source-code attribution

Each rendered roofline carries the source under test (vLLM commit/branch/image +
delivery + infra/SGLang patch) from the bundle's `experiment_provenance_v1` block,
resolved to a source URL via the campaign's source-registry mapping
(consumed by `provenance.py`). A roofline with no source link is not
reproducible -- the renderer prints the link panel and `publish_to_lake`
carries the flat provenance columns.

## Model shapes + ceilings (sources)

- Shapes: each family's published `config.json` (`roofline_math.from_hf_config`),
  or the in-pod `/work/model/config.json` captured by `roofline-sweep.sh`. The
  `_REGISTRY` in `roofline_math.py` carries the exemplar (GLM-5.1) verbatim so
  pre-embedded campaigns still render.
- Ceilings: `configs/sol-ceilings.yaml` (datasheet peaks, never
  inlined). GB300: NVFP4 15 / FP8 7.5 / BF16 3.75 PFLOPS/GPU. HBM3e 8 TB/s/GPU.
