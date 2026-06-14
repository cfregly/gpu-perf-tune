---
name: inference-quantize-calibrate
last_validated: 2026-06-07
description: >-
  Produce quantized inference weights from a BF16/FP8 base checkpoint via a
  post-training-quantization (PTQ) pipeline -- instead of only ever pulling
  NVFP4 weights pre-quantized. A calibration prep Job then
  a quantize Job: NVIDIA ModelOpt -> NVFP4 (default), or llm-compressor -> FP8
  (alt). Writes weights + per-layer scales to an experiment-prefixed PVC, gated on
  inference-model-eval accuracy + a perf A/B. When a vendor nvidia/*-NVFP4 exists,
  runs a 3-way comparison (base vs ours vs NVIDIA) on accuracy + perf + ncu.
  Ack-gated. The calibrate+quantize phase of inference-model-optimize.
  Triggers on "quantize a model", "calibrate NVFP4", "run modelopt PTQ", "produce
  NVFP4 weights", "llm-compressor FP8", "PTQ pipeline", "compare our PTQ to NVIDIA
  NVFP4", "our nvfp4 vs nvidia nvfp4", "calibration dataset for quantization",
  "quantize <model> to NVFP4/FP8", or any combination of "quantize / calibrate /
  PTQ / modelopt / llm-compressor" with "nvfp4 / fp8 / weights / model /
  inference".
allowed-tools:
  - mcp__profile_and_optimize__evidence_init
  - mcp__profile_and_optimize__perf_baseline_record
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__perf_tune_report_campaign_init
  - mcp__profile_and_optimize__perf_tune_report_cell_run
  - mcp__profile_and_optimize__perf_tune_report_report_render
  - mcp__profile_and_optimize__perf_tune_report_publish_to_lake
  - mcp__profile_and_optimize__search_runbooks
  - Bash(kubectl:*)
  - Bash(sinfo:*)
  - Bash(squeue:*)
  - Read
  - Write
---

# inference-quantize-calibrate

> **Fast model loading (always-applied):** when staging a base checkpoint or standing up a vLLM deploy, never load 100s-of-GB single-stream via s3fs FUSE (a GLM-5.1-sized model takes ~50 min that way. See `docs/METHODOLOGY.md`). Prefer, in order: a fast model-loading endpoint when reachable -> parallel multipart to local NVMe (`server/tools/stage-model-parallel.py`) -> `runai_model_streamer` (`--load-format runai_streamer`) -> tensorizer. Flag a slow load loudly: if effective rate < ~500 MB/s on a large model, STOP and switch. Details: `server/docs/inference-fast-model-loading.md`.

## Purpose

Produce a quantized checkpoint **from a base model** -- rather than relying on
NVFP4 weights *pulled pre-quantized* from `nvidia/*-NVFP4` HF repos. This skill
provides the calibrate + quantize recipe for a fresh model as a
two-Job PTQ pipeline:

1. **Calibration-dataset prep Job** -- tokenizes a small calibration corpus
   (default: a few hundred samples of C4 / the model's chat template) into the
   format the quantizer's PTQ calibration loop expects.
2. **Quantize Job** -- runs the quantizer over the base weights with the
   calibration set, emitting quantized weights + per-layer activation/KV scales.

Two pluggable backends:

| Backend | Target | Use when |
| --- | --- | --- |
| **NVIDIA ModelOpt** (`modelopt`, default) | **NVFP4** | Matches the `nvidia/*-NVFP4` serving checkpoints (modelopt 0.45, `quant_algo=NVFP4`, group_size 16). |
| **llm-compressor** | **FP8** (compressed-tensors) | FP8 weight quant for an FP8-tier deploy or an FP8-vs-NVFP4 A/B. |

The output PVC is experiment-prefixed and the result feeds straight into a deploy
+ accuracy/perf gate.

## When to use

- A new model has no published NVFP4/FP8 variant and you need quantized weights
  to serve it on Blackwell efficiently.
- You want to reproduce / improve on a vendor NVFP4 checkpoint with your own
  calibration set (e.g. a domain-specific calib corpus).
- The `inference-model-optimize` orchestrator reached Phase 4.

Do **not** use this skill for:

- Models that already have a good vendor NVFP4/FP8 checkpoint you trust -- just
  pull the vendor checkpoint (cheaper, byte-for-byte vendor recipe).
- KV-cache dtype selection at serve time (`--kv-cache-dtype=fp8_e4m3`) -- that is
  a `vllm.extraArgs` lever, not weight PTQ. Tune it with
  [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md).
- Runtime activation-quant kernel work --
  that is custom-kernel territory, not this PTQ pipeline.

## Example prompts

- "Quantize MiniMax-M2.7 to NVFP4 with modelopt and validate accuracy."
- "Run an FP8 llm-compressor PTQ on this model and A/B it against BF16."
- "Produce NVFP4 weights for `zai-org/GLM-5.2` with a C4 calibration set."
- `/inference-quantize-calibrate --base /models/base --backend modelopt --target nvfp4`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. The **base** (BF16 or FP8) checkpoint reachable (HF id / S3) -- you quantize
   *from* the base, not the NVFP4 serving copy. On the current GB300 (4 GPU/node)
   a large MoE base may need TP across 2 NVLink-domain nodes. B200 fits 8/node.
2. Output persistence (cluster-profile aware): on plain-K8s (no storageclass)
   write to emptyDir + upload to S3-compatible object storage from inside the
   pod. On legacy Slurm-on-K8s, an experiment-prefixed PVC
   (never reuse a standing `*-cache` PVC).
3. Cluster access + a free GPU node: plain-K8s GB300 (default) via
   `kubectl` GPU requests. Legacy Slurm-on-K8s via `sinfo`/`squeue`. The quantize Job is
   GPU-resident for calibration forward passes.
4. The quantizer container image (modelopt or llm-compressor) -- the `quantize/`
   templates pin one. Air-gapped clusters mirror it first.
5. The explicit ack flag for the quantize Job (it consumes a GPU node).

## Workflow

### Phase 0: confirm intent + scaffold the quant bundle

State back: `base` path, `backend` (`modelopt` | `llm-compressor`), `target`
(`nvfp4` | `fp8`), calibration corpus + sample count, output PVC name
(experiment-prefixed), and the experiment-id (== the parent run-id). Read the
base `config.json` for arch + chat template so calibration uses the right
formatting. The scaffolder lays down `quantize/`:

- `quantize/00-calib-pvc.yaml` -- experiment-prefixed output PVC.
- `quantize/01-calib-dataset-job.yaml` -- calibration-dataset prep Job.
- `quantize/02-quantize-job.yaml` -- the PTQ quantize Job (backend-parameterized).
- `quantize/quantize.py` -- the backend dispatcher (modelopt / llm-compressor).
- `quantize/README.md` -- the per-bundle runbook.

### Phase 1: prepare the calibration set

Apply `01-calib-dataset-job.yaml`. It tokenizes the calibration corpus with the
model's tokenizer + chat template into the quantizer's expected JSONL/arrow and
writes it to the output PVC. Gate: Job Complete + the calib file exists and has
the requested sample count.

### Phase 2: run the quantize Job (ack-gated)

```text
# gate the manifest first:
verify-experiment-labels.sh <id-slug> quantize/02-quantize-job.yaml
# then apply (the Job carries the ack/intent in an env flag the entrypoint checks):
kubectl -n <slurm-namespace> apply -f quantize/02-quantize-job.yaml
```

The Job runs `quantize.py`:

- **modelopt path**: `mtq.quantize(model, NVFP4_DEFAULT_CFG, calib_loop)` then
  export to a HF-layout NVFP4 checkpoint (`quant_algo=NVFP4`, group_size 16) +
  `hf_quant_config.json`, so vLLM auto-detects it exactly like a
  `nvidia/*-NVFP4` repo.
- **llm-compressor path**: a `oneshot` FP8 recipe (compressed-tensors), emitting
  per-layer scales.

Outputs land at `/models/quantized/target` on the experiment PVC. Gate: Job
Complete + the output `config.json`'s quant method is the expected one
(`modelopt`/`modelopt_fp4`/`NVFP4` for NVFP4, `compressed-tensors`/`fp8` for FP8)
-- the same assertion the `03b` puller makes on a vendor checkpoint.

### Phase 3: validate the quantized weights (the gate that matters)

Deploy the quantized weights (the parent orchestrator's Phase 5, or a standalone
deploy off the experiment PVC) and run:

- [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md) (GPQA + MMLU-Pro)
  vs the **base** model's scores.
- A perf A/B (throughput + median TPOT) vs the base via
  [`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md) /
  [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md).

Gate: accuracy delta within tolerance (default: no metric drops > 1.0 absolute
point) **AND** a perf win. A quant that wins perf but fails accuracy is a
**no-ship** -- report the regression and keep the base/vendor checkpoint. Record
the accuracy + perf result as a baseline so future re-quants diff against it.

### Phase 3b: 3-way comparison vs the vendor NVFP4 (our-PTQ vs NVIDIA-PTQ)

When a vendor `nvidia/*-NVFP4` (or `*-FP8`) checkpoint exists for the same model,
"is our PTQ good enough" is the wrong question -- the decision is **which of the
two NVFP4 checkpoints to ship**. Run a head-to-head **3-way comparison** across
three variants on one pinned node:

| Variant | Source | Role |
| --- | --- | --- |
| `base` | BF16/FP8 base | accuracy ceiling reference |
| `ours` | our ModelOpt NVFP4 (Phase 2 output) | candidate |
| `nvidia` | `nvidia/*-NVFP4` (pulled from HF) | incumbent baseline |

Compare on three axes, each artifact-cited:

1. **Accuracy** -- [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md)
   (GPQA + MMLU-Pro) for all three. Report `ours` and `nvidia` as deltas vs
   `base`.
2. **Perf** -- median TPOT + throughput, same-node + >=3 trials
   ([`inference-tune-sweep`](/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md) controlled A/B).
3. **Per-kernel** -- [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)
   on the dominant decode kernel of each, to explain *why* one wins (tensor-core
   engagement, achieved roofline %) rather than asserting it.

Implement as a `perftunereport` campaign with **one cell per variant** (`base` /
`ours` / `nvidia`) so the three land side-by-side in the atlas + PDF + perf-lake
automatically, and the verdict ("ship ours" / "ship vendor" / "tie -> keep
vendor") is a published, data-backed row -- not a claim. A common honest outcome
is parity, in which case the vendor checkpoint wins on provenance. Record that.

### Phase 3c: NVFP4-MoE serving gotchas

A quantized MoE checkpoint can fail to *serve* (or serve garbage) for reasons
unrelated to PTQ quality. Three gotchas, each with the fix:

1. **Activation-capability gap.** The native FP4 MoE kernels (CUTLASS,
   FlashInfer trtllm-gen) only advertise a fixed activation set. A model whose
   expert FFN uses an unlisted activation (gemma4 = `gelu_pytorch_tanh` ->
   `MoEActivation.GELU_TANH`) is rejected at engine init even though the weights
   are fine. Fix: add the activation to the kernel class `_supports_activation`
   (CUTLASS routes non-SiLU through the generic `apply_moe_activation`, so it is
   often a Python-only add) + the FlashInfer `ACTIVATION_TO_FI_ACTIVATION` map
   (note: FlashInfer has no tanh-gelu variant -> trtllm maps GELU_TANH -> Geglu,
   an *approximation*. The exact path is CUTLASS).
2. **Intermediate alignment is TP-dependent.** The FP4 MoE kernels need the
   per-rank expert intermediate `%64` (and the fused gate+up `2N %128` for
   trtllm). `moe_intermediate_size` that is fine at TP=1 can break at TP=4
   (gemma4: 704 -> per-rank 176, not %64) with a `NotImplementedError` (CUTLASS
   gated-pad) or `assert M%128==0` (FlashInfer). Two fixes: (a) **offline**
   zero-pad `moe_intermediate_size` up to a multiple of `64*TP` on the
   checkpoint (pure zero-insertion on the stored pre-swizzle tensors -
   numerically exact, no modelopt. The production answer, no runtime cost), or
   (b) a **runtime** gated per-shard pad in `prepare_nvfp4_moe_layer_for_fi_or_cutlass`.
3. **The numerical gate MUST use the chat template for `-it` models.** A bare
   `/v1/completions` prompt on an instruction-tuned model produces degenerate
   repetition by design ("is is is...") - this is NOT a quant bug. Gate with
   `/v1/chat/completions`. (On gemma4 the raw gate falsely reported 0/4 garbage
   while the chat gate returned correct answers 4/4.) Always include a known-good
   control (TP=1, or the BF16 base) when a gate fails, to attribute model-vs-gate.

### Phase 3d: fused-3D experts + the model-family registry

The single biggest PTQ hazard for MoE is **fused 3D expert tensors**: some HF
models store experts as one `nn.Parameter [num_experts, dim, dim]` instead of an
`nn.ModuleList` of `nn.Linear`. A naive `mtq.quantize(model, NVFP4_DEFAULT_CFG,
calib_loop)` walks `nn.Linear` modules and **SILENTLY SKIPS** the fused experts
(~90%+ of the model) -> a fabricated partial checkpoint (NVIDIA/Model-Optimizer
#1173). The generated `quantize/quantize.py` now carries a **model-family
registry** + a post-quant **coverage guard** that handles this. Use it instead of
hand-rolling:

- **`MODEL_FAMILY` registry** (keyed by `config.json` `model_type`) auto-resolves
  the **recipe** + **exclude scope** per family. Match the `nvidia/*-NVFP4`
  sibling's `hf_quant_config.json` `exclude_modules` exactly (pull it from HF) so
  the A/B is apples-to-apples. Worked entries: `gemma4_text` (native-plugin,
  6-pattern experts-only excludes, `needs_mm_assets`), `qwen3_moe` (generic,
  exclude `lm_head` + router `mlp.gate`), `nemotron_h` (generic, exclude the
  Mamba/attn `mixer.*` projections). Add a new family with one dict entry or pass
  `--exclude-pattern`.
- **Recipes**: `generic` (ModuleList experts, OR fused experts a modern modelopt
  auto-registers a QuantModule for) | `native-plugin` (this script registers
  Gemma4TextExperts #1219 for the fused Gemma-4 MoE) | `fail-closed` (refuse a
  fused MoE). `generic` no longer *refuses* a fused model upfront -- it attempts
  and the **coverage guard** (experts quantized > 0) catches a true silent-skip,
  so it works for archs modelopt natively supports without a hand-written plugin.
- **transformers version determines fused-vs-ModuleList.** The SAME model can load
  as ModuleList on transformers 4.x and as a fused-3D module on 5.x (Qwen3-30B-A3B
  is `Qwen3MoeExperts` fused-3D on transformers 5.5.4). If `generic` hits a
  coverage-0 FATAL on a fused load, either pin a transformers that exposes
  ModuleList experts, or use a modelopt build that registers that arch's
  QuantModule. ALWAYS read the MoE STRUCTURE DUMP the script prints first.
- **Hybrid Mamba-MoE (Nemotron-H) needs `mamba-ssm` + `causal-conv1d` in the quant
  image** -- its transformers modeling code hard-imports `mamba_ssm` at load
  (vLLM serves it with its own kernels, but the transformers calibration load
  does not). The stock vLLM serving image lacks them AND no prebuilt Blackwell
  (sm_100) wheels exist -> **build them from source CPU-side** (CUDA 12.9 + nvcc
  cross-compile, `TORCH_CUDA_ARCH_LIST="9.0;10.0"`, `--no-build-isolation`,
  `*_FORCE_BUILD=TRUE`, ~10 min for both), cache the wheels in object storage, and have the
  quant Job `pip install` them. Worked: causal_conv1d 1.6.2 + mamba_ssm 2.3.2 on
  CUDA 12.9 / torch 2.11 / cp312. PROVEN: Nemotron-H quantized via the generic path
  (5934 experts) and reached vendor accuracy parity (gsm8k chat 0.69).
- **Export `AttributeError: 'list' object has no attribute 'keys'`** -- some remote
  modeling code (Nemotron-H) declares `_tied_weights_keys` as a LIST, but transformers
  5.x's `_get_tied_weight_keys` calls `.keys()` (expects a dict). The generated
  `quantize.py` normalizes list -> `{k: k}` for every module before
  `export_hf_checkpoint` (generic. Fixes any list-tied-weights model on newer transformers).

### Phase 3e: serve marlin-first. Calibrate with the natural-routing loop

- **MARLIN first for ROBUSTNESS, but ALWAYS run the serve-backend MATRIX per-model --
  do NOT conclude "marlin-only".** W4A16 marlin (dequants FP4->FP16) is the robust
  default that serves any NVFP4 export. But it is NOT the perf winner: on a SAME-NODE
  3-way sweep, NVFP4-**cutlass** (W4A4 native FP4) ties FP8 and beats NVFP4-marlin by
  ~+21% at high throughput (Qwen3 c256), while marlin wins only the c16 memory-bound
  knee and FP8 wins c1 latency -- there is NO universal winner (see
  `docs/METHODOLOGY.md`). Cutlass degeneracy is
  **MODEL-SPECIFIC** (Gemma's `gelu_pytorch_tanh` + 704-pad), NOT a property of "our
  export" -- Qwen3 serves cutlass fine. So: test EVERY backend per model with a raw
  "2+2" coherence gate, then run the backend x concurrency matrix sweep
  (same-node, 3-trial). Never carry a backend verdict from one arch to another.
- **A serve "FATAL"/"Engine core initialization failed" is a per-cell DEBUG target,
  not "blocked".** The real error is in the `(EngineCore pid=...)` worker subprocess
  (upstream of the APIServer "see root cause above" wrapper) -- read that
  traceback directly. One-line root causes that have hidden behind it:
  `ModuleNotFoundError: pytest` (marlin_fp4 -> cupy.testing lazy import, `pip install
  pytest`), `AttributeError: 'list'...keys()` (Nemotron-H `_tied_weights_keys` vs
  transformers 5.x. Normalize list->dict before export), `NotImplementedError ...
  Qwen3MoeExperts ... export` (pin transformers 4.x for ModuleList experts).
- **Calibrate with `create_forward_loop(cnn_dailymail)` + `HF_HUB_OFFLINE=0`**
  (the quant Job must set it). The natural-routing loop on plain web text scales
  the experts correctly. A chat-control-token-wrapped calib corpus SKEWS the
  per-expert amax -> degenerate output even with a correct plugin + serve path.
  Keep a plain-text JSONL only as the offline fallback.
- **Pad the MoE intermediate ONLY if a backend rejects the alignment.** Most
  models are already 64-aligned (Qwen3 768). Gemma-4's 704 needed the offline
  704->768 pad. `PAD_TARGET=0` (skip) is the common case -- do not pad reflexively.
- **`-it` accuracy gate = 0-shot raw + chat** (e.g. a gsm8k probe), n>=400 for a
  tight A/B. Few-shot raw transcripts make an -it model emit end-of-turn (empty).

### Teardown

Tear down the quantize Job + calib Job by label
(`kubectl delete job -l experiment=<id-slug>`). Keep the output PVC (it holds the
champion weights) until the parent run promotes or discards them, then delete it
PV-last per the experiment-isolation rule.

## Calibration notes

- **Sample count**: NVFP4 PTQ converges with a few hundred calibration samples,
  more rarely helps and costs GPU time. Start at 256-512.
- **Domain match**: use a calibration corpus representative of the serving
  workload when accuracy is marginal -- the calib set shapes the per-layer scales.
- **KV scales**: NVFP4 KV is blocked for MLA models in the current serving vLLM
  (see the GLM/Kimi notes). This skill quantizes *weights* + activation scales.
  Set `--kv-cache-dtype` at serve time as a separate lever.

## Verdict rigor (DRAFT vs VERDICT)

Accuracy deltas are node-independent (a quant either changes the logits or it
does not), so a single eval run is a valid accuracy VERDICT. The **perf** half of
the gate follows the standard rule -- DRAFT unless same-node + >=3 trials +
metric-isolated. Report the accuracy and perf verdicts separately. Never let a
perf win paper over an accuracy regression.

## Safety

- **Ack-gated** -- the quantize Job consumes a GPU node. It fails closed without
  the explicit intent flag (`safety=submits_jobs`), per
  [`server/docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).
- **Experiment isolation** -- the output PVC + both Jobs are experiment-prefixed
  and carry `experiment=<id-slug>`. Never reuse a standing `*-cache` PVC. Tear
  down by label, PV last.
- **Scheduling (cluster-profile aware)** -- plain-K8s GB300: `default-scheduler` +
  `kubectl` GPU preflight + emptyDir/hf-pull + object-storage output. Legacy Slurm-on-K8s: the
  quantize Job uses the Slurm scheduler + the lock toleration + `sinfo`/`squeue`.
- **No accuracy bypass** -- the accuracy gate is mandatory. A quant is not "done"
  until it has been eval'd against the base. No silent fallback to "ship it
  anyway".

## Source-of-truth references

- The vendor-checkpoint assertion (`quant_method in {modelopt, modelopt_fp4,
  NVFP4}`) this skill's output must satisfy.
- [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md) -- the accuracy gate.
- NVIDIA TensorRT Model Optimizer (`modelopt`) PTQ + llm-compressor `oneshot`
  upstream docs (pinned in `quantize/README.md` of the generated bundle).
- `server/AGENTS.md` "All performance numbers..." + "Verdict rigor" +
  "Experiment Isolation & Traceability".

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

If this skill produces a measurement (tok/s, latency, %SoL, speedup), follow the
rigor discipline: capture L1 zymtrace + L3 DCGM (L4 ncu where feasible)
Speed-of-Light and publish `--strict`. Canonical map: `server/AGENTS.md`
"Rigor discipline index" / `docs/METHODOLOGY.md`. Skills that
do not produce measurements are exempt (`server/AGENTS.md` "Speed-of-light framing").

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
