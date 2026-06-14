---
name: inference-aa-workload
last_validated: 2026-05-29
description: >-
  Reproduce the Artificial Analysis (AA) language-model performance workload
  shapes against an OpenAI-compatible chat endpoint using NVIDIA AIPerf.
  Drives the three AA text shapes (1k input / >=1k answer, 10k / >=1.5k, 100k
  / >=2k) with temperature 0, top_p 1, and the vLLM-style min_tokens +
  ignore_eos "at least N answer tokens" guarantee. Two modes: synthetic
  (AIPerf generates the prompt at the token mean) and dataset-replay (a
  generated o200k_base-counted JSONL replayed identically). Ships a
  self-contained script and an `aa` perf_tune_report cell_run backend. Use when
  comparing a hosted inference endpoint to AA leaderboard numbers or
  reproducing AA's methodology. Triggers on "artificial analysis workload",
  "AA benchmark", "AA 1k/10k/100k shapes", "reproduce artificialanalysis.ai",
  "AA methodology", "aa-10k", "compare to AA leaderboard", or any combination
  of "artificial analysis / AA" with "workload / shape / benchmark / dataset".
allowed-tools:
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - mcp__profile_and_optimize__perf_tune_report_campaign_init
  - mcp__profile_and_optimize__perf_tune_report_cell_run
  - mcp__profile_and_optimize__perf_tune_report_atlas_aggregate
  - mcp__profile_and_optimize__perf_tune_report_import_roofline_sweep
  - mcp__profile_and_optimize__perf_tune_report_report_render
  - mcp__profile_and_optimize__perf_tune_report_publish_to_lake
  - Bash(python3:*)
  - Bash(uv:*)
  - Bash(aiperf:*)
  - Bash(date:*)
  - Read
  - Write
---

# inference-aa-workload

## Purpose

Reproduce the [Artificial Analysis (AA) language-model performance
benchmarking methodology](https://artificialanalysis.ai/methodology/performance-benchmarking)
against an OpenAI-compatible chat endpoint, so your deployment
can be compared apples-to-apples with AA's published leaderboard
numbers. AA tests three text workload shapes with fixed sampling parameters
and an "at least N answer tokens" guarantee. This skill drives NVIDIA
[AIPerf](https://github.com/ai-dynamo/aiperf) to emit the same shapes and
captures TTFT / output-speed / throughput.

The AA shapes (the single source of truth lives in
[`aa_workload.py`](/plugins/profile-and-optimize/server/tools/perf_tune_report/runners/aa_workload.py)
`AA_SHAPES` and the bundled script's `AA_SHAPES`, kept in lock-step by a
drift-guard test):

| Shape | Input tokens | Min answer tokens |
| --- | ---: | ---: |
| `aa-1k` | ~1,000 | 1,000 |
| `aa-10k` | ~10,000 | 1,500 (AA site default) |
| `aa-100k` | ~100,000 | 2,000 |

AA parameters reproduced: `temperature: 0`, `top_p: 1` (non-reasoning
default), tokens counted as `o200k_base` (tiktoken), and the "at least N
answer tokens" guarantee via vLLM-style `min_tokens` + `ignore_eos` extra
inputs.

Two run modes:

- **`synthetic`** (default) - AIPerf synthesizes the prompt at the target
  token mean (`--synthetic-input-tokens-mean`). Faithful to AA's
  "generate a fresh prompt to fill the token budget" approach and to the
  original reproducer script.
- **`dataset-replay`** - a generated JSONL of real-text prompts (sized to
  the o200k_base token budget) is replayed identically
  (`--input-file ... --custom-dataset-type mooncake_trace` with the
  `text_input` field), giving deterministic cross-endpoint comparability.

## Reasoning models (TTFT vs TTFO. Thinking often cannot be disabled)

AA's synthetic-filler + `min_tokens` + `ignore_eos` shape is **ill-posed for a reasoning
model**. With filler input the model reasons about nonsense and fills the entire answer
budget with `<think>` tokens (a reasoning model on aa-1k can emit ~1k reasoning
tokens and ZERO answer tokens), so the AA "answer length" guarantee becomes a
"reasoning length" guarantee and the latency metric you must report changes:

- **AIPerf TTFT** (`time_to_first_token`) counts the first token of ANY type, including
  the reasoning stream. On a reasoning model this is time-to-START-thinking (tens of ms),
  NOT the answer latency.
- **AIPerf TTFO** (`time_to_first_output_token`) counts the first non-reasoning (answer)
  token. This is the AA-faithful answer latency. For a non-reasoning model TTFT == TTFO.

Rules for reasoning models:

1. **Report TTFO with its COVERAGE, not TTFT.** AIPerf averages TTFO over only the
   requests that emitted an answer token. On synthetic filler a reasoning model can
   exhaust the whole budget thinking, so always pair `ttfo_avg_ms` with `ttfo_coverage`
   (= ttfo.count/ttft.count) - a low-coverage TTFO is an answered-subset stat. The
   perf-report runners persist `ttfo_avg_ms` + `ttfo_coverage` +
   `reasoning_token_count` (schema `atlas_v1`). A TTFT-only number under-reports answer
   latency by the whole think phase (a 0.10 s "TTFT" can hide a ~4 s answer latency).
2. **TTFT/TTFO are NOT comparable across stacks unless normalized.** AIPerf can only split
   them when the stack surfaces a reasoning field it recognizes, and the field name/exposure
   varies (vLLM 0.20 `reasoning_content`, vLLM 0.22 `reasoning`, some front-ends expose
   neither -> TTFT==TTFO "tied"). The portable, stack-agnostic measurement is the bundled
   [`aa_ttfo_probe.py`](assets/aa_ttfo_probe.py): it times the first `delta.content` (answer)
   token regardless of the reasoning field name and counts reasoning tokens. Run it from
   inside the serving pod (urllib only, no PyPI/aiperf).
3. **Thinking may NOT be disable-able.** Some reasoning models enforce thinking at the
   parser level: `chat_template_kwargs: {enable_thinking: false}` is IGNORED and the
   vendor ships no non-thinking mode. When no-think is impossible, the only
   apples-to-apples latency is **TTFO on a real-content prompt** (`--mode dataset-replay`
   with a real-content `--input-file`), because a real question elicits a bounded think then
   an answer. Synthetic filler does not. Pass `--reasoning` to the reproducer to surface
   this guidance.

4. **Record the reasoning-parser state PER STACK, and treat a tied TTFT==TTFO column on a
   reasoning model as the fingerprint of an UNCONFIGURED parser until falsified** (one
   streaming request: look for `reasoning_content`/`reasoning` delta fields). Measured on a
   large reasoning MoE in a single-stack A/B: with the parser off, TTFO collapses onto TTFT
   exactly (think text streams inline as `content`, reasoning uncounted). With it on, TTFO
   lands seconds later (the think phase) while the decode rate is parser-invariant. Reasoning
   splitting is OPT-IN on the common stacks (vLLM `--reasoning-parser`, NVIDIA Dynamo
   `--dyn-reasoning-parser`), so two endpoints under the same column header can be reporting
   different events. The parser state belongs in the full-context descriptor of every AA
   number on a reasoning model.
5. **Unified-parser landmine:** some engines register UNIFIED parsers where the TOOL-CALL
   parser alone activates reasoning splitting (e.g. vLLM 0.22 `minimax_m2`: "Using unified
   parser ... for reasoning and tool parsing"). You cannot keep tool parsing and disable
   reasoning exposure. A true parser-OFF arm must drop the reasoning parser, the tool-call
   parser, AND auto tool choice together.

## When to use

- "Reproduce the Artificial Analysis 10k workload on the Kimi-K2.6 deploy."
- "How does my W&B Inference endpoint compare to AA's leaderboard at 1k/10k/100k?"
- "Generate the AA replay dataset once so we can run it across three providers."
- "Run the AA single-prompt and 10-parallel load scenarios."

Do **not** use this skill for:

- An agentic-coding replay dataset against an **in-cluster** vLLM endpoint -
  that is
  [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)
  (mooncake-trace multi-turn replay, not AA synthetic shapes).
- Quality / accuracy evaluation (GPQA, MMLU-Pro, Terminal-Bench) - use
  [`inference-model-eval`](/plugins/profile-and-optimize/skills/inference-model-eval/SKILL.md).
- Training performance benchmarking - out of scope for this plugin.

This skill is distinct from `inference-perf-bench` precisely because AA's
synthetic fixed-shape methodology is a different workload than an agentic
multi-turn replay dataset.

## Example prompts

- "Run the AA 1k/10k/100k workload against api.inference.wandb.ai for Kimi-K2.6."
- "AA benchmark, dataset-replay mode, concurrency 10."
- `/inference-aa-workload --mode synthetic --shapes aa-10k`
- "Generate the AA-shaped replay dataset for all three shapes."

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. **AIPerf reachable** - `aiperf` on PATH, or `uv` (the script falls back
   to `uv run --with aiperf --python 3.13 aiperf`, which fetches aiperf
   ephemerally), or `AIPERF_BIN` set. Canonical install:
   `uv tool install aiperf` - this lands the `aiperf` shim in `~/.local/bin`
   (already on PATH. Run `uv tool update-shell` once if that dir isn't on
   PATH yet). Inside a venv you can instead `pip install aiperf` (PyPI
   package `aiperf`, Python >=3.10. MacOS and Linux x86_64 install from
   prebuilt wheels - Linux aarch64 needs a C toolchain, e.g.
   `build-essential`, for the `crick` dependency). The reproducer prefers a
   real `aiperf` on PATH over the uv fallback.
2. **Endpoint URL + auth** - `URL` (default `https://api.inference.wandb.ai`)
   and, for the W&B gateway, `WANDB_INFERENCE_API_KEY` (or `API_KEY`). The
   key is read from the environment and never written to the campaign config
   or evidence bundle.
3. **`tiktoken`** (dataset-replay only, recommended) - for exact
   `o200k_base` token-budget sizing. Without it the generator falls back to
   a ~0.75 tokens/word heuristic and warns loudly.
4. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** (perf_tune_report integration only) - set by the
   bundled MCP server. Campaigns land under `./campaigns/` by default.

## Interaction style

Iterative - confirm parameters, optionally generate the dataset, run the
shapes one at a time, report TTFT / output-speed / throughput per shape,
then (optionally) bridge to the perf_tune_report campaign pipeline. Pause at each
gate. Never auto-advance past a failed shape.

## Workflow

### Phase 0: confirm intent

Resolve the operator request to concrete parameters and state them back:
model, endpoint URL, shapes (`aa-1k,aa-10k,aa-100k`), load scenario
(concurrency 1 = AA "single prompt". Concurrency 10 = AA "parallel
prompts"), mode (`synthetic` vs `dataset-replay`), and request count. Get
confirmation before running.

### Phase 1: standalone path (quick, portable)

Run the bundled self-contained script (no cluster, no MCP). From any shell,
the repo-root `bin/aa-workload` wrapper runs it under the server venv's python
(which has `tiktoken`), so no venv activation is needed:

```bash
# From any directory: the wrapper forwards all args to the reproducer.
WANDB_INFERENCE_API_KEY=... bin/aa-workload --mode synthetic --shapes aa-1k,aa-10k,aa-100k --dry-run
```

Or invoke the script directly with your own interpreter:

```bash
# Dry-run first to inspect the exact aiperf commands.
WANDB_INFERENCE_API_KEY=... python3 \
  plugins/profile-and-optimize/skills/inference-aa-workload/assets/repro_artificialanalysis.py \
  --mode synthetic --shapes aa-1k,aa-10k,aa-100k --dry-run

# Generate the reproducible replay dataset once (dataset-replay mode):
python3 .../repro_artificialanalysis.py --generate-dataset-only --shapes aa-10k

# Then execute.
WANDB_INFERENCE_API_KEY=... python3 .../repro_artificialanalysis.py --mode synthetic
```

Report the per-shape AIPerf artifact dirs under
`artifacts/artificial-analysis/<shape>/`. Ask whether to land the result in
the perf-lake (Phase 2).

### Phase 2: perf_tune_report campaign path (traceable, roofline-ready)

One AA shape maps to one perf-report cell (so the `(cell_id, concurrency)`
atlas key stays unique. A campaign that wants all three shapes uses three
cells: `aa-1k`, `aa-10k`, `aa-100k`). Each cell's YAML carries an `aa:`
block (`model`, `url`, `shape`, `mode`, `api_key_env`, optional
`tokenizer` / `custom_dataset_type` / `namespace`+`bench_pod`+`kube_context`
for in-cluster parity).

In kube mode (`bench_pod` set), each per-concurrency run is automatically
bracketed with `/metrics` scrapes of the endpoint's `vllm:spec_decode_*`
counters: raw scrapes land under `<cell>/spec_metrics/`, and the computed
acceptance length (`AL = 1 + accepted/drafts`) + accept rate
(`accepted/draft_tokens`) land on the cell's normalized/atlas rows as
`acceptance_length` / `spec_accept_rate` at the `(cell, concurrency)` grain.
On a spec-OFF deploy the window sees no drafts and the fields stay null.
Opt out with `aa: {spec_scrape: false}`.

**Settle discipline (sweep vs sleep. Opt-in, kube mode).** Deploy-first
measurements run 6-37% low without shape-matched warmup (measured in a GB300
settle audit), and prewarm alone is insufficient. Enable via the cell's
`aa:` block: `prewarm_shapes: [aa-1k, aa-10k]` (one completion per shape at
the shape's dims before the sweep), `burn_in: true` (one run-and-discard pass
of the first concurrency point), `settle_s: 30` (pause after prewarm/burn-in
and between recorded points). Steps are logged to `<cell>/commands/settle.log`,
never quote a first-after-deploy number without this discipline.

```text
mcp__profile_and_optimize__perf_tune_report_campaign_init with:
  args: ["--config", "<aa-campaign>.yaml", "--json"]

mcp__profile_and_optimize__perf_tune_report_cell_run with:
  args: ["--campaign", "<id>", "--cell", "aa-10k", "--backend", "aa",
         "--aa-shape", "aa-10k", "--aa-mode", "synthetic",
         "--i-understand-this-submits-jobs", "--json"]

mcp__profile_and_optimize__perf_tune_report_atlas_aggregate with: args: ["--campaign", "<id>", "--json"]
mcp__profile_and_optimize__perf_tune_report_report_render   with: args: ["--campaign", "<id>", "--json"]
mcp__profile_and_optimize__perf_tune_report_publish_to_lake with: args: ["--campaign", "<id>", "--json"]
```

`cell_run --backend aa` is **ack-gated** (`safety=submits_jobs`). Use
`--dry-run` to print the per-concurrency aiperf commands without executing.
AA cells flow through `atlas_aggregate -> report_render -> publish_to_lake`
unchanged because the runner records `backend=aiperf` provenance and stashes
the AA shape/mode in each row's `extra`.

### Phase 2.5: prefill/decode roofline (page 7) - always-on

AA campaigns are serving-throughput campaigns, so they MUST carry the
prefill/decode roofline (the "what C maxes the TFLOPs / is decode >=75% HBM /
which sharding degree" answers - `publish_to_lake --strict` refuses a
throughput/mixed serving campaign that omits page 7). Against the same AA pod,
before teardown, capture + import the sweep with your deploy's roofline-sweep
script (same step as `inference-perf-tune-report` Phase D3):

```text
roofline-sweep.sh <ns> <pod> <out> <model> <tokenizer> \
  "1 2 4 8 16 32 64 128 192" "512 1024 2048 4096 8192" <container>
mcp__profile_and_optimize__perf_tune_report_import_roofline_sweep with:
  args: ["--campaign", "<id>", "--bundle", "<out>", "--hardware", "GB300",
         "--tensor-parallel", "<tp>", "--quant", "<NVFP4|FP8|BF16>", "--cache-mode", "cold"]
```

Then re-render (`report_render`) so page 7 + `roofline_v1` land. See
`server/tools/perf_tune_report/ROOFLINE-METHODOLOGY.md`.

## Safety

- **`cell_run --backend aa` is ack-gated** (`safety=submits_jobs`) per
  [`server/docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).
  Pass `--i-understand-this-submits-jobs` to execute, `--dry-run` to preview.
- **No credentials in artifacts.** The API key is read from the env var
  named by `cell.aa.api_key_env` (default `WANDB_INFERENCE_API_KEY`). It is
  never written into the campaign config, `SOURCE.md`, or commit history.
- **Public-gateway caveat.** AA's methodology benchmarks a provider's public
  endpoint, so the standalone script defaults to the W&B public gateway. The
  documented dev-vs-prod ~3x throughput skew (per
  [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) Safety) means an
  in-cluster service URL and the public gateway are NOT interchangeable -
  report which one was measured.
- **Provider field rejection.** If the endpoint rejects `min_tokens` /
  `ignore_eos`, set `EXTRA_OUTPUT_CONTROLS=0` (script) /
  `cell.aa.extra_output_controls: false` (campaign). Output length then
  becomes a cap, not an "at least N answer tokens" guarantee - note this in
  the result.

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

Every result table this skill produces MUST carry a `%SoL` column alongside
absolute throughput / latency, per `docs/METHODOLOGY.md` "Speed-of-light
framing". Peak numbers come from the shared
`configs/sol-ceilings.yaml` cited by key path
(`b200_sm100.hbm3e_tbps`, etc.). The perf_tune_report renderer auto-emits the SoL
pages when DCGM + zymtrace capture is present for the bench window.

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

## Source-of-truth references

- AA methodology: <https://artificialanalysis.ai/methodology/performance-benchmarking>.
- Shared command builder + dataset generator + normalizer:
  [`server/tools/perf_tune_report/runners/aa_workload.py`](/plugins/profile-and-optimize/server/tools/perf_tune_report/runners/aa_workload.py).
- perf_tune_report runner: [`server/tools/perf_tune_report/runners/aa_bench.py`](/plugins/profile-and-optimize/server/tools/perf_tune_report/runners/aa_bench.py).
- Self-contained script: [`assets/repro_artificialanalysis.py`](/plugins/profile-and-optimize/skills/inference-aa-workload/assets/repro_artificialanalysis.py).
- Pair: [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)
  (in-cluster multi-turn replay counterpart),
  [`inference-perf-tune-report`](/plugins/profile-and-optimize/skills/inference-perf-tune-report/SKILL.md)
  (renders the campaign PDF).
- [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) - full-context reporting + Speed-of-light framing.
