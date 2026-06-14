---
name: inference-spec-decode-train
last_validated: 2026-06-01
description: >-
  Train + validate a speculative-decoding draft head (EAGLE3 or DFlash) for an
  ARBITRARY target LLM, generalizing the GLM-5.1-only SpecForge recipe so any new
  model gets one. Parameterizes what that offline
  recipe hard-codes: target path, aux-hidden-states layers (derived [1, L/2-1, L-4] from
  config.json num_layers), chat template, draft-head config, and method. Wires
  the SpecForge prepare_hidden_states -> train_eagle3 / train_dflash ->
  convert-to-vLLM flow on Slurm, then gates on a measured acceptance-length A/B
  (dflash_vllm_eval.py + vLLM spec_decode_* counters) vs the standing
  config. This is the train-spec-decode phase of inference-model-optimize, usable
  standalone. Triggers on "train an EAGLE3 draft", "train a DFlash head", "build a
  speculative decoder for <model>", "spec-decode draft training", "draft model
  acceptance", "SpecForge for <model>", "generalize eagle3-train", or any
  combination of "train / build / validate" with "eagle3 / dflash / draft /
  speculative / spec-decode".
allowed-tools:
  - mcp__profile_and_optimize__evidence_init
  - mcp__profile_and_optimize__slurm_drain
  - mcp__profile_and_optimize__slurm_resume
  - mcp__profile_and_optimize__search_runbooks
  - Bash(kubectl:*)
  - Bash(sinfo:*)
  - Bash(squeue:*)
  - Bash(sbatch:*)
  - Bash(sacct:*)
  - Read
  - Write
---

# inference-spec-decode-train

## Purpose

Give an arbitrary target LLM a speculative-decoding draft head. SpecForge's
worked EAGLE3 + DFlash recipes are hard-wired to one model
(the `run_glm5p1_eagle3_offline.sh` recipe bakes in the target path, the aux
layers `[1,38,74]`, the `glm5p1` chat template, and the GLM draft config). This
skill **generalizes** that stack so a new model gets the same train ->
validate -> promote loop without copy-pasting the GLM script.

It reuses SpecForge verbatim
(`prepare_hidden_states.py`, `train_eagle3.py`, `train_dflash.py`,
`dflash_vllm_eval.py`, `convert_dflash_to_vllm.py`, the `slurm/*.sbatch`
launchers) and only parameterizes the per-target inputs.

## When to use

- A new model has a latency ceiling you want to beat with a trained draft head
  and no public draft exists.
- You want to train a DFlash head as an alternative to a model's built-in MTP.
- The `inference-model-optimize` orchestrator reached Phase 6.

Do **not** use this skill for:

- `deepseek_mtp` / `ngram` -- there is nothing to train (MTP is a built-in head,
  ngram is prompt-driven). Just set `--speculative-config` and validate
  acceptance (Phase 7 of the orchestrator / the validation half here).
- Consuming a pre-trained public draft (e.g. `lightseekorg/kimi-k2.6-eagle3`) --
  pull it and serve it. No training needed.
- The acceptance A/B alone on an already-trained head -- you can run just the
  validation phase below.

## Example prompts

- "Train an EAGLE3 draft head for MiniMax-M2.7 and A/B its acceptance."
- "Build a DFlash speculator for this 70-layer model."
- "Generalize the GLM eagle3-train recipe to `zai-org/GLM-5.2`."
- `/inference-spec-decode-train --target /models/base --method eagle3 --chat-template <name>`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. The **target** base checkpoint (BF16 or FP8 -- NOT the NVFP4 serving copy)
   staged on a shared FS / PVC the Slurm trainer can read.
2. The SpecForge trainer container built for the target arch -- a new arch may
   need a newer SGLang/transformers (a target arch class missing from the
   trainer's SGLang is a kill-gate. Reconcile versions before any GPU spend).
3. The target `config.json` (for `num_hidden_layers` -> aux layers, and to
   confirm the arch is supported by the trainer's SGLang).
4. Free GPU nodes: plain K8s (e.g. GB300) via `kubectl` GPU requests, or
   Slurm-on-K8s (e.g. B200) via `sinfo`/`squeue`. Training is a
   multi-hour job -- submit-and-queue, never preempt serving.
5. A chat-template name registered in SpecForge's `specforge/data/template.py`
   for the target family (add one if absent, as was done for `glm5p1`).

## Workflow

### Phase 0: derive the per-target inputs + scaffold

State back the parameters and **derive the EAGLE3 aux layers from
`config.json`**: for an `L`-layer target the SpecForge convention is
`[1, L//2 - 1, L - 4]` (the GLM 78-layer target -> `[1, 38, 74]`). Resolve:

- `target-model-path`, `method` (`eagle3` | `dflash`), `chat-template`,
  `draft-config` (a per-target draft-head JSON, seeded from the GLM
  `glm5p1-eagle3.json` with `num_layers`/`hidden_size`/vocab taken from the
  target `config.json`), `num-gpus`/`tp-size`, dataset.

The scaffolder lays down `spec-decode/` in the bundle:

- `spec-decode/configs/<slug>-eagle3.json` (+ `<slug>-dflash.json`) -- draft head
  config rendered from the target `config.json`.
- `spec-decode/run-<slug>-offline.sh` -- the generalized 2-step recipe
  (parameterized clone of `run_glm5p1_eagle3_offline.sh`: env-driven
  `TARGET_MODEL_PATH`, `AUX_LAYERS`, `CHAT_TEMPLATE`, `DRAFT_CONFIG`, `METHOD`).
- `spec-decode/slurm/` -- copies of the SpecForge `*.sbatch` launchers
  (prepare-data, capture, train, eval) with the target paths parameterized.
- `spec-decode/dflash_vllm_eval.py` + `convert_dflash_to_vllm.py` -- the
  in-engine acceptance eval + checkpoint converter (verbatim from SpecForge).
- `spec-decode/deploy/<slug>-spec-canary.yaml` -- the `--speculative-config`
  canary deploy for the acceptance A/B.

### Phase 1: prepare the offline dataset

Submit the data-prep sbatch (CPU): build the mixed UltraChat + ShareGPT JSONL
through the target's chat template. Gate: the JSONL exists with the expected
conversation count and renders cleanly through the chat template (a bad
loss-mask / template is the GLM "training loss stays zero" failure mode).

### Phase 2: capture target hidden states

Submit the capture sbatch: `prepare_hidden_states.py` holds the target resident
at `--tp-size` and dumps aux hidden states at the derived `AUX_LAYERS`. Offline
(capture-then-train) is mandatory for huge MoE targets -- hold the target only
during capture, then iterate draft training cheaply. Gate: hidden-state shards
written for the full dataset.

### Phase 3: train the draft head (submit-and-queue)

Submit the train sbatch (`train_eagle3.py` or `train_dflash.py`). This is the
multi-hour job. Submit it as a self-driving Slurm job and do NOT babysit (it
re-queues through preemption). Gate: training `acc_0` / loss is non-degenerate
(a healthy run lands `acc_0` around ~0.6-0.7. A head stuck near zero acceptance
is a train/serve mismatch to debug, not a result to ship).

### Phase 4: convert to a vLLM-loadable draft

Run `convert_dflash_to_vllm.py` (DFlash) / the EAGLE3 export to produce a
draft the serving vLLM can load via `--speculative-config.model=<path>`. Stage
it on the serving PVC (e.g. `/models/target/speculator/<slug>/<sha>/`).

### Phase 5: validate acceptance (the gate that matters)

Deploy the `--speculative-config` canary and measure **in-engine acceptance**:

- `spec-decode/dflash_vllm_eval.py` -- the authoritative in-engine acceptance
  (acceptance length per position) via vLLM metrics.
- vLLM Prometheus `spec_decode_num_accepted_tokens` /
  `spec_decode_num_draft_tokens` / `spec_decode_num_accepted_tokens_per_pos`
  deltas over a driven bench window.

Run it as a same-node A/B: trained-draft arm vs the standing config (e.g. MTP
K=3, or no-spec). Gate: a measured **acceptance-length win** AND an end-to-end
TPOT win under cudagraph (NOT eager -- an eager spec-decode "win" is host
overhead, not GPU work). A draft head that loses -- even one whose training
acceptance looked healthy -- is **do-not-ship**: keep the standing config and
report why.

### Teardown

Cancel any running Slurm jobs you own. Tear down the canary deploy by label
(`kubectl delete deploy,pod -l experiment=<id-slug>`). Keep the trained draft
artifact on the PVC until the parent run promotes or discards it.

## Generalization map (GLM-hardcoded -> parameter)

| GLM-5.1 hardcoded value | Generalized parameter | Derivation |
| --- | --- | --- |
| `TARGET_MODEL_PATH=/mnt/data/models/GLM-5.1-FP8` | `--target-model-path` | operator input (base, not NVFP4) |
| `--aux-hidden-states-layers 1,38,74` | `AUX_LAYERS` | `[1, L//2-1, L-4]` from `config.json` `num_hidden_layers` |
| `--chat-template glm5p1` | `--chat-template` | target family. Add to SpecForge `template.py` if absent |
| `configs/glm5p1-eagle3.json` | `spec-decode/configs/<slug>-eagle3.json` | rendered from target `config.json` (hidden_size/vocab/layers) |
| EAGLE3 only | `--method eagle3 | dflash` | operator input |
| `TP_SIZE=8` (single B200 node) | `--tp-size` / `--num-gpus` | from the target's fit math |

## Verdict rigor (DRAFT vs VERDICT)

Training `acc_0` is a training-time signal, not a serve verdict. The
ship/no-ship claim is the **serve acceptance + TPOT A/B** in Phase 5, which
follows the standard rule: DRAFT unless same-node + >=3 trials + metric-isolated
(median TPOT/ITL) + both arms under matching `cudagraph_mode`. Acceptance length
must be measured in-engine (the vLLM counters), never inferred from end-to-end
latency alone.

## Safety

- **Never preempt serving** -- training uses idle Slurm nodes only. Submit-and-
  queue with a self-driving Job, do not babysit or drain a serving node. A
  drained quiet window is for bench windows only, never for training.
- **Standing config stays until a measured win** -- the standing
  `--speculative-config` (e.g. MTP K=3) is not changed until the Phase-5 A/B
  shows a real acceptance + TPOT win under cudagraph.
- **Experiment isolation** -- the canary deploy + any draft-staging objects are
  experiment-prefixed + `experiment=<id-slug>` labeled. Teardown by label.
- **Local fork only** -- keep any SpecForge changes on a local fork. No upstream
  PR / external outreach without explicit per-turn operator approval.

## Source-of-truth references

- SpecForge `examples/run_glm5p1_eagle3_offline.sh`
  -- the 2-step offline recipe the generalized `run-<slug>-offline.sh` clones.
- vLLM `vllm/config/speculative.py` -- the `--speculative-config` schema
  (`method`, `model`, `num_speculative_tokens`, `draft_tensor_parallel_size`).
- `docs/METHODOLOGY.md` -- benchmark hygiene (eager vs cudagraph for
  spec-decode), kernel-work classification, and verdict rigor.

## Contact

Open an issue in this repository.

## Full-context reporting (no bare numbers)

Per the methodology canon "Every performance number carries its full context (no bare
numbers)" (`docs/METHODOLOGY.md`, "Full-context reporting"): every number this
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
Speed-of-Light and publish `--strict`. Canonical map: `docs/METHODOLOGY.md`. Skills that
do not produce measurements are exempt (`docs/METHODOLOGY.md` "Speed-of-light framing").

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per the Grind Mandate (`docs/METHODOLOGY.md`, "Always be grinding"). A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.
