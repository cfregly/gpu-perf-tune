---
name: inference-tune-sweep
last_validated: 2026-06-01
description: >-
  Search a vLLM serving model's config space for its best inference performance:
  a `perftunereport campaign_run` matrix sweep over concurrency x
  max_num_batched_tokens x kv-cache-dtype x cudagraph_mode, plus a same-node
  controlled A/B (`run-controlled-ab.sh`) for any decode-latency claim, gated by
  `perf-baseline-diff`. Picks a champion config that beats the baseline on the
  run's focus metric (throughput tok/s OR median TPOT/ITL), honoring warm/cold +
  eager/cudagraph methodology rules. This is the tuning phase of
  `inference-model-optimize`, usable standalone. Not a Bayesian optimizer -- it
  sweeps an operator-named grid (an --optimizer hook is stubbed). Triggers on
  "tune the vllm config", "sweep
  max_num_batched_tokens", "find the best serving config", "vllmArgs sweep",
  "cudagraph_mode A/B", "kv-cache-dtype sweep", "tune concurrency", "config
  search for <model>", or any combination of "tune / sweep / optimize / search /
  A-B" with "vllm / config / vllmArgs / concurrency / batched-tokens / cudagraph".
allowed-tools:
  - mcp__profile_and_optimize__perf_tune_report_campaign_init
  - mcp__profile_and_optimize__perf_tune_report_campaign_run
  - mcp__profile_and_optimize__perf_tune_report_atlas_aggregate
  - mcp__profile_and_optimize__perf_baseline_record
  - mcp__profile_and_optimize__perf_baseline_diff
  - mcp__profile_and_optimize__search_runbooks
  - Bash(perftunereport:*)
  - Bash(kubectl:*)
  - Bash(sinfo:*)
  - Bash(squeue:*)
  - Bash(run-controlled-ab.sh:*)
  - Bash(pin-node.sh:*)
  - Read
  - Write
---

# inference-tune-sweep

> **Fast model loading (always-applied):** when standing up a vLLM deploy, never load 100s-of-GB single-stream via s3fs FUSE (a several-hundred-GB model can take ~50 min that way). Prefer, in order: a fast-model-loading endpoint when reachable -> parallel multipart to local NVMe (`plugins/profile-and-optimize/server/tools/stage-model-parallel.py`) -> `runai_model_streamer` (`--load-format runai_streamer`) -> tensorizer. Flag a slow load loudly: if effective rate < ~500 MB/s on a large model, STOP and switch. Details: `plugins/profile-and-optimize/server/docs/inference-fast-model-loading.md`.

## Purpose

Find the best serving config for a vLLM deployment by sweeping a config grid and
A/B-ing the survivors, instead of hand-editing `vllm.extraArgs` and eyeballing
one bench. It is the "tune / optimize" phase of
[`inference-model-optimize`](/plugins/profile-and-optimize/skills/inference-model-optimize/SKILL.md) factored into
its own discoverable skill, and is fully usable on its own against any already-
deployed model.

> **Steady-state window (throughput trap):** every throughput/concurrency cell MUST send
> `num_prompts >= 2*c` (one full batch beyond ramp+drain). `output_throughput` is
> tokens/full-duration, so too few prompts (e.g. `num=c+4`) makes the window ramp/drain-
> dominated and **undercounts high-c throughput ~1.6-1.8x**. Also set `max_num_seqs >= max c`
> so the decode batch is not capped below the sweep's top concurrency.
> `import_roofline_sweep` WARNs on any cell `< 2c`.

There is deliberately **no optimizer engine** here -- the search is an
operator-named matrix (the levers that actually move inference perf:
concurrency, `max_num_batched_tokens`, `kv-cache-dtype`, `cudagraph_mode`,
`enable-chunked-prefill`, `enable-prefix-caching`). A propose/validate optimizer
loop (borrowing the `ai_tuning_*` engines)
is a documented `--optimizer` hook left as a stub for a later phase.

## When to use

- You have a stable baseline deploy and want its throughput / latency champion
  config before publishing a roofline.
- You changed a quality-risk lever (`kv-cache-dtype`, spec-decode) and need a
  fair, methodology-clean A/B vs the prior config.
- The `inference-model-optimize` orchestrator reached Phase 3.

Do **not** use this skill for:

- The first stability pass on a brand-new model -- get it Ready + QA-clean
  (deploy, smoke-test, accuracy QA) first. Tuning a hanging config chases ghosts.
- A which-kernel "why is it slow" question -- that is the profiling skills
  ([`inference-kernel-profile`](/plugins/profile-and-optimize/skills/inference-kernel-profile/SKILL.md) et al.).
- Rendering the final PDF -- that is
  [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md).

## Example prompts

- "Sweep max_num_batched_tokens and concurrency for the GLM-5.1 deploy and pick
  the throughput champion."
- "A/B cudagraph_mode FULL vs PIECEWISE for this model, same node, 3 trials."
- "Find the best kv-cache-dtype for Kimi on B200 without regressing TPOT."
- `/inference-tune-sweep --release glm-inference --focus throughput`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. A stable, Ready baseline deploy of the target model (pod Healthy, smoke
   passes). Record it as a baseline first (`perf_baseline_record`).
2. profile-and-optimize bundled MCP server installed (`perftunereport campaign_run` reachable).
3. The deploy bundle's `my-values-<slug>.yaml` (the `base_values` the sweep
   overlays per cell) and the chart fork.
4. Cluster access + a free GPU node confirmed before any GPU cell: plain K8s
   (e.g. GB300) via a `kubectl` free-GPU preflight (k8s GPU requests are
   authoritative). Slurm-on-K8s (e.g. B200) via `sinfo`/`squeue`.
5. For a decode-latency **verdict**: the `pin-node.sh` + `run-controlled-ab.sh`
   helpers so both arms land on the SAME node.

## Workflow

### Phase 0: state the focus + grid

Resolve and state back:

- `focus`: `throughput` | `latency` | `mixed` (sets the champion metric and the
  campaign's `focus:` field -- throughput tok/s/GPU for throughput, median
  TPOT/ITL for latency).
- The grid: which of `{concurrency, max_num_batched_tokens, kv-cache-dtype,
  cudagraph_mode, enable-chunked-prefill, enable-prefix-caching}` to vary and
  their values. Keep it small -- a 2x3 grid is usually enough to find the knee.
- The baseline to beat (the current `my-values` config, recorded as a baseline).

### Phase 1: author the campaign matrix YAML

Write a `campaign_run` matrix YAML: `target_release`, `target_namespace`,
`chart_dir`, `base_values: my-values-<slug>.yaml`, and one `cell` per grid point
with `helm_overrides` (the per-cell `vllm.extraArgs` deltas) + `concurrencies` +
`backend: vllm-sweep`. Set `campaign.focus` to the Phase-0 focus. Pin the bench
client `backend` explicitly (the methodology-hygiene rule -- `--backend vllm` is
~3% faster dispatch than the `openai` default).

### Phase 2: run the sweep (ack-gated)

```text
perf_tune_report_campaign_run --config <matrix>.yaml --i-understand-this-submits-jobs
```

`campaign_run` is `submits_jobs`-tier: it fails closed without the ack flag, and
each cell helm-upgrades the release, warms up, runs a `vllm bench sweep`, writes
a per-cell verdict to `commands/`, and (fail-fast) aborts subsequent cells on a
red unless `--continue-on-red`. Use `--dry-run` first to print every helm
override + bench command with no cluster spend.

**Methodology gates (mandatory, per `docs/METHODOLOGY.md`):**

- **Warm vs cold** -- a sequential prefix-cached sweep's tail point is a *warm*
  best-case. Label every throughput number warm (sweep-tail) or cold
  (fresh/single-shot) and never compare across the two.
- **Eager vs cudagraph** -- both arms of any A/B MUST match `cudagraph_mode`. An
  eager "win" is host-overhead, not GPU work. Record `enforce_eager` /
  `cudagraph_mode` per cell.

### Phase 3: pick the champion + confirm with a controlled A/B

`atlas_aggregate` the cells, read the per-cell focus metric, and pick the
champion grid point. For any decode-latency claim OR a ship/no-ship verdict,
confirm with a same-node repeated-trial A/B (champion vs baseline):

```text
N=$(pin-node.sh pick)                                   # verified-idle node, fail-closed
# deploy baseline arm pinned to N, then:
run-controlled-ab.sh <baseline-deploy> <out>/baseline   # records the node
# deploy champion arm on the SAME N, then:
PIN_VERDICT_NODE=$N run-controlled-ab.sh <champion-deploy> <out>/champion
```

`run-controlled-ab.sh` reports per-trial + mean +/- std median TPOT/ITL per
concurrency and fails closed if the arms are cross-node (cross-node => DRAFT).

### Phase 4: gate with perf-baseline-diff + write the champion config

```text
perf_baseline_diff --baseline <recorded-baseline> --current <champion> \
  --tolerance <focus-appropriate>
```

Gate: GREEN/YELLOW with a real win on the focus metric. On pass, update the
bundle's `my-values-<slug>.yaml` with the champion `vllm.extraArgs` and record
the champion as the new baseline. On a red or a within-noise tie, keep the prior
config and report "no improvement found" -- do not ship a config that only wins
warm or only wins in eager.

### Phase 5: cross-engine (SGLang) tuning + champion selection

The vLLM `campaign_run`/helm grid above tunes ONE engine. To tune **across the
stack (vLLM AND SGLang)** -- the production-relevant comparison -- drive the
levers through the engine-agnostic variant A/B harness
(`run-variant-ab.sh`),
which benches every arm from ONE shared client (`vllm bench serve --backend
openai`) so vLLM and SGLang numbers are comparable by construction. Fill
`arms.tsv` with both engines' levers (one arm per lever. First row per engine is
that engine's baseline):

- **vLLM levers:** `max_num_batched_tokens`, `kv-cache-dtype`, `cudagraph_mode`,
  `--enable-expert-parallel`, MoE backend (FlashInfer / cutlass / trtllm).
- **SGLang levers (engine=sglang col):** `--moe-runner-backend`
  (`flashinfer_cutlass` / `triton`), `--attention-backend` (`flashinfer` /
  `trtllm`), `--enable-torch-compile` / cuda-graph (`--disable-cuda-graph` to A/B
  it off), `--mem-fraction-static`, `--chunked-prefill-size`,
  `--disable-radix-cache` (match caching across engines before crowning a
  throughput champion -- a radix-cache replay can manufacture a phantom win).

Then `perftunereport import_variant_ab --bundle <out> --model <id>` (engine-tagged
`vllm-sweep` / `sglang-sweep` cells) -> `atlas_aggregate` -> **SoL (REQUIRED for a
throughput/mixed champion): `import_roofline_sweep` (page 7 - `publish_to_lake
--strict` refuses a serving campaign that omits it) + `dcgm_correlate`** ->
**`perftunereport champion_select --campaign <id> --top 3`** to pick the baseline +
top-X cross-engine champion with the production recommendation. Import the
baseline + each kept variant's roofline sweep into the ONE campaign so page 7
overlays baseline-vs-optimized on a single per-GPU roofline (see
`plugins/profile-and-optimize/server/tools/perf_tune_report/ROOFLINE-METHODOLOGY.md`). The champion VERDICT requires the multi-workload + accuracy gates
(`--workloads-present`, `--accuracy-gate pass`) and L3 byte-grounding. See
[`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md).

### Optional: the --optimizer hook (stub, later phase)

A future `--optimizer bayesian|hyperband` flag would replace the fixed grid with
an adaptive propose/validate loop borrowing the `ai_tuning_*`
report -> space -> proposal -> validate machinery. Out of scope for v1. The grid
sweep is the shipped behavior. The hook is documented here so the search surface
is forward-compatible.

## Verdict rigor (DRAFT vs VERDICT)

Every sweep number defaults to **DRAFT**. Promote the champion to a **VERDICT**
only when the confirming A/B is same-node + >=3 trials (mean +/- std),
metric-isolated (median TPOT/ITL for latency -- NOT output tok/s at small
`num_prompts`), against the production-representative baseline, and the
warm/cold + eager/cudagraph arms match. A cross-node or single-trial champion is
a DRAFT.

## Safety

- **Scheduling (cluster-profile aware)** -- on plain K8s (e.g. GB300):
  `default-scheduler` + a `kubectl` free-GPU preflight (k8s requests are
  authoritative). On Slurm-on-K8s (e.g. B200): the Slurm `schedulerName` + the lock
  toleration + `sinfo`/`squeue`, where a `default-scheduler` + hard `nodeSelector`
  GPU grab is forbidden (double-books Slurm). NOTE: if the chart cannot stage
  model weights per cell, run the config sweep via the variant A/B harness
  (`run-variant-ab.sh` arms) instead of `campaign_run`/helm.
- **Ack-gated** -- `campaign_run` is `submits_jobs`-tier. Fails closed without
  `--i-understand-this-submits-jobs`. `--dry-run` is the safe preview.
- **Experiment isolation** -- sweep cells use experiment-prefixed serve names +
  `experiment=<id-slug>` labels. Teardown by label. Never reuse standing names.
- **No silent fallbacks** -- a within-noise result is reported as "no
  improvement", never rounded up to a win.

## Source-of-truth references

- `run-controlled-ab.sh` + `pin-node.sh` -- the same-node controlled-A/B and
  node-pinning helpers (Phase 3).
- [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md) -- where the
  `campaign_run` matrix + the final render live.
- `docs/METHODOLOGY.md` -- benchmark hygiene (warm/cold, eager/cudagraph, pin
  the bench backend) + verdict rigor.

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

## Champion close: capture the known-good config (CONFIG half)

A champion is not closed until its **known-good config** is captured (the performance
ratchet's CONFIG half). When this sweep picks a
champion, **`known_good_config record`** its REQUIRED serve flags (the boot-blocker /
crash-at-high-c workarounds it needs -- e.g. a model that needs `gdn_prefill_backend=triton`)
into `perf-tune-report/configs/known-good-configs.yaml`
via the [`inference-known-good-config`](/plugins/profile-and-optimize/skills/inference-known-good-config/SKILL.md) skill, so the won
config is never re-discovered the hard way. The champion-close gate
fails closed unless BOTH the next lever (grind-ledger) AND a registered known-good config are present.
