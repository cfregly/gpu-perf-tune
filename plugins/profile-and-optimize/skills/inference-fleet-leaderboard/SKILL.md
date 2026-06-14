---
name: inference-fleet-leaderboard
last_validated: 2026-06-07
description: >-
  Render cross-model fleet leaderboards from the local perf-report campaigns in ONE
  command (`perftunereport fleet_leaderboard`): a latency tier (aa-1k/10k/100k tok/s/user
  + TTFT + cost at c=1/c=10), a throughput tier (peak tok/s/GPU per (model,quant,TP)
  with latency + $/1M-tok), and a decision capstone (the perf Pareto
  frontier across decode-latency x first-token x throughput, with the perf-dominated
  set and a hard PERF != quality caveat). Auto-discovers every model's AA + roofline
  cells in campaigns/*/atlas.jsonl, so re-running refreshes as new campaigns publish.
  Use to answer "which model do I pick" / "rank the fleet on latency/throughput/cost"
  / "is model X perf-dominated". Triggers on "fleet leaderboard", "which model should
  I use", "cross-model comparison", "rank the fleet", "model selection guide", "pareto
  frontier of models", "perftunereport fleet_leaderboard", or any combination of "fleet /
  cross-model / which-model / leaderboard / pareto" with "latency / throughput / cost
  / pick / rank / compare".
allowed-tools:
  - mcp__profile_and_optimize__perf_tune_report_fleet_leaderboard
  - mcp__profile_and_optimize__perf_tune_report_experiments_index
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Bash(perftunereport:*)
  - Read
  - Write
---

# inference-fleet-leaderboard

## Purpose

Each perf-report campaign PDF is one model, "which model do I pick?" is a
**cross-campaign** question the per-campaign artifacts cannot answer. This skill
renders three complementary cross-model leaderboards from the local campaigns
dir in one `perftunereport fleet_leaderboard` call:

| output | tier | answers |
| --- | --- | --- |
| `AA-FLEET-LEADERBOARD-<hw>.md` | latency | interactive rank (aa-1k/10k/100k tok/s/user + TTFT, c=1 and c=10) + cost |
| `THROUGHPUT-FLEET-LEADERBOARD-<hw>.md` | throughput | peak tok/s/GPU per (model,quant,TP) + the latency/$ at that peak |
| `FLEET-MODEL-SELECTION-<hw>.md` | decision | perf Pareto frontier (pick-by-priority) vs perf-dominated, PERF-only caveat |

It auto-discovers every model's AA cells (`cell_id` in aa-1k/10k/100k) and
roofline cells (non-AA, `output_tps_per_gpu`) across `campaigns/*/atlas.jsonl`,
normalizing the drifted atlas `model`/`quant` strings (org prefixes + quant
suffixes stripped, quant upper-cased) so one model that appears as
`zai-org/GLM-5.1`, `GLM-5.1-NVFP4`, and `GLM-5.1` collapses to one row.

One verb replaces what would otherwise be a handful of per-leaderboard
generator scripts.

## When to use

- Someone asks "which model is fastest / cheapest / best for
  batch?" or "is model X worth its cost vs model Y?".
- After any new AA or roofline campaign publishes, to refresh the fleet view
  (the leaderboards are PROVISIONAL while a sweep is still filling cells).
- As the cross-model companion to `experiments_index` (which lists experiments,
  this ranks models).

Do **not** use this to *measure* a model (that is `inference-perf-bench` /
`inference-aa-workload`) or to render a single-campaign PDF (that is
`inference-perf-tune-report`). This skill only synthesizes already-published cells.

## Workflow

1. **Run the verb** (defaults: `--hardware GB300`, `--gpu-hr 8.60`, output dir
   = the campaigns dir's parent):

```bash
perftunereport fleet_leaderboard --campaigns-dir ./campaigns --json
```

   or the MCP tool `mcp__profile_and_optimize__perf_tune_report_fleet_leaderboard`. For
   different hardware pass `--hardware B200` (filters atlas rows + names outputs) and
   `--gpu-hr <rate>`.

2. **Read the decision capstone first** (`FLEET-MODEL-SELECTION-<hw>.md`): the
   pick-by-priority table + the non-dominated frontier answer "which model" in one
   look. Then the throughput / AA leaderboards for the per-axis detail.

3. **Heed the caveats the renderer embeds:**
   - **PROVISIONAL** when a roofline sweep is still publishing - re-run to refresh.
   - **PERF-only != quality:** a perf-dominated model (e.g. a larger/smarter MoE,
     or GLM-5.1 whose value is DSA-sparse long-context) is correctly dominated on
     the AA short-context shape yet chosen for capability. The frontier is "the
     perf-efficient choice AMONG quality-equivalent models", never "never use".
   - **Engine-version skew:** per-model serves may use different vLLM builds. Treat
     TTFT as directional across engines, output-speed/cost as the robust metrics.

4. **For the grounded *why*** (e.g. the models are host/KV-bound, not
   memory-bound - DCGM L3 + zymtrace L1), pair the leaderboards with a curated
   DCGM+zymtrace attribution doc built via `inference-dcgm-correlate` +
   `analyze-zymtrace-workload`. This verb does not generate one.

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

The leaderboards carry the $/1M-output-token cost column (latency-optimal at c=1,
throughput-optimal at the knee). They do
not themselves recompute %SoL (that is `inference-dcgm-correlate` /
`inference-perf-tune-report` pages 4-7). A common throughput-tier finding is
that peak tok/s/GPU is set by per-token active-param work + host/KV scheduling,
not by hitting the HBM/compute roofline (HBM <8% at small-active-param knees).

## Asset validation (review + FAIL LOUD)

Every asset this skill emits (leaderboard table / Pareto chart / fleet view) is held to
`docs/METHODOLOGY.md` "Asset validation": the generator **FAILS LOUDLY** on
missing/bad/degenerate data (no campaigns discovered,
zero plot-ready points, `unknown`/null/NaN where a value is required) -> raise / non-zero exit
naming what is missing, never a silent placeholder/empty leaderboard, and the agent **REVIEWS**
the rendered leaderboard for human-sense + 100% accuracy (every model row traces to its
campaign, latency/throughput/cost matched on concurrency, the Pareto frontier
is plausible) and **rebuilds** it if wrong -- never ships a wrong/confusing asset with a caveat.

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

## Provenance

Backed by the `perf_tune_report` CLI `fleet_leaderboard` verb (+ the auto-derived
`perf_tune_report_fleet_leaderboard` MCP tool) in
`plugins/profile-and-optimize/server/tools/perf_tune_report/fleet_leaderboard.py`. Read-only on
campaigns/. Writes only the three `*-FLEET-*.md` files. Unit tests:
`test_fleet_leaderboard.py`.
