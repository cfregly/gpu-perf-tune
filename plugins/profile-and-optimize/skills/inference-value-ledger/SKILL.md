---
name: inference-value-ledger
last_validated: 2026-06-05
description: >-
  Render the leadership value-prop ledger for the inference effort: the deployable
  wins vs the FlashInfer-TRTLLM + best-tuned-vLLM 0.21/0.22 baseline, grouped
  DONE / IN-PROGRESS / NOT-DONE / CLOSED-NEGATIVE, each row data-backed by a perf-lake
  campaign (live sol_rigor + verdict tier), plus the ranked GRIND FRONTIER of next levers
  (the always-be-grinding performance ratchet). Joins the curated
  perf-tune-report/configs/value-findings.yaml registry with the live campaigns via
  `perftunereport value_view`; flags any finding whose backing campaign is missing or
  ungrounded. Use when you need to show value / report to leadership / answer "what have
  we uncovered that we can deploy, revalidate, or pursue". Triggers on "value ledger",
  "value prop", "show value", "show the value prop", "leadership view", "what wins do we
  have", "vs flashinfer", "value-findings", "what can we deploy", or any combination of
  "value / wins / findings / leadership / report" with "inference / vllm / flashinfer / perf".
---

# inference-value-ledger

One command to render the always-current leadership value-prop view.

## Run it

```bash
# print to stdout:
perftunereport value_view
# write a doc:
perftunereport value_view --out /path/to/VALUE-LEDGER.generated.md
# JSON envelope (finding + flag counts, e.g. for CI):
perftunereport value_view --json
```

`perftunereport` ships with the profile_and_optimize server; if it is not on PATH,
put the server on PYTHONPATH and invoke it as a module.

## How it works

- **Curated source-of-truth:** `perf-tune-report/configs/value-findings.yaml`
  — one entry per finding (`id`, `title`, `lifecycle`, `baseline`, `win`, `hardware`,
  `deploy_readiness`, `campaign_ids`, **`next_lever`** [+ optional `next_value`]). Edit this
  when a finding's status changes or a new finding lands. This is the ONLY hand-maintained part.
- **`perftunereport value_view`** joins it with the LIVE perf-lake campaigns: each campaign's
  `report_status.json` (`sol_rigor` / `dcgm_grounded`) + `verdict.json` (`tier` /
  `baseline_named`), and renders the grouped table **plus the ranked GRIND FRONTIER**. Win
  numbers are human-verified in the registry; the live columns + flags are read at render time
  so the ledger never silently drifts from the lake.
- **Flags surface** when a campaign is missing locally (S3-only / unpublished), ungrounded
  (`sol_rigor` none/L1), its baseline is not named, or **a finding has no `next_lever`** — fix
  by publishing/tagging the campaign or naming the next lever.

## Keep it honest

Lifecycle is evidence-state, not aspiration: a finding is `done` only when its campaign is
published + grounded; `in_progress` while revalidating; `not_done` for research;
`closed_negative` for banked dead-ends (value = prevented spend). Never mix B200 (TP=8) and
GB300 (TP=4) without the hardware column. Every row names its baseline (the
DRAFT-vs-VERDICT "fair baseline" rule). **Code-under-test provenance match:** a
finding's `source_refs` `delivery`+`commit` MUST match the cited `campaign_ids`' provenance -- citing
an `overlay`/offline-prepped campaign as the benefit of an `infr-patch` is a cross-tier DRAFT defect
(even when the kernels match). `value_view` flags this via `provenance_match_problems()` (delivery/
commit mismatch between `source_refs` and a cited campaign). Typical failure: citing an
overlay-deploy champion's number as a source patch's benefit -- always re-measure on the
patch's own deploy.

## Full-context reporting (no bare numbers)

Per the methodology canon "Every performance number carries its full context (no bare
numbers)" (`docs/METHODOLOGY.md`, "Full-context reporting"): every number this
skill emits (throughput, latency, TPOT/ITL, BW, %SoL, speedup, efficiency, goodput, acceptance
rate, scaling efficiency, thermal/failure rate — whatever it reports) MUST carry its full
measurement-context descriptor, and every comparison MUST be matched on it. A bare number is a
defect — it cannot set a default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold; latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` — never inline a peak), sol_rigor (L1–L4), trials n (mean±std), same-node, baseline named. (If the metric is not roofline-bound — e.g. accuracy/acceptance — omit `%SoL` but keep the rest of the descriptor.)
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) — never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

## Asset validation (review + FAIL LOUD)

Every asset this skill emits (value-prop ledger / GRIND FRONTIER view / table) is held to
the asset-validation canon (`docs/METHODOLOGY.md`,
"Asset validation"): the generator **FAILS LOUDLY** on missing/bad data (a finding whose backing campaign is
missing or ungrounded, `unknown`/null where a value is required) -> flag it loudly, never silently
drop or fabricate a row; and the agent **REVIEWS** the ledger for human-sense + 100% accuracy
(every win row is campaign-grounded with live sol_rigor + verdict tier, no orphaned/ungrounded
claim presented as a win) and **rebuilds** it if wrong -- never ships a wrong/confusing ledger
with a caveat.

## PR value proposition (WHY + benefit + applicability)

This skill IS the value-prop ledger -- so when one of its findings graduates into a PR (a
downstream patch PR or any upstream/shareable PR), that PR MUST carry the same value framing,
not just the metric: the **WHY** (blocker removed / capability unlocked), the **measured benefit**
(full descriptor + `%SoL`/`sol_rigor` + a resolving perf-lake / evidence link; never a hand-waved
"no perf penalty"), and an **applicability matrix** -- model architecture (sparse/MoE vs dense),
size, reasoning-vs-not, quant, parallelism (incl. any TP>1-specific behavior), workload, and
concurrency regime; state what it does NOT apply to. **Keep it tight (no AI-slop):** lead with
SoL%/roofline% and matched numbers; cut hedging, redundancy, and decorative charts (infra PRs don't embed
images). De-slop checklist: no em-dashes (#1 AI-slop tell), minimal bold, plain punctuation, inline code
out of narrow table cells. Source: `docs/METHODOLOGY.md` "Value proposition" + "De-slop".

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per the Grind Mandate (`docs/METHODOLOGY.md`, "Always be grinding"). A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses;
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.
