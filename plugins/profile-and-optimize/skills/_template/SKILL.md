---
name: _template
last_validated: 2026-05-20
description: >-
  Copy this directory to plugins/profile-and-optimize/skills/<your-skill-name>/ and
  rewrite this description to be third-person + include WHAT the skill does
  and WHEN (specific trigger phrases). Max 1024 chars. Do NOT leave the
  word "template" in the real skill's description.
disable-model-invocation: true
allowed-tools:
  - mcp__profile_and_optimize__search_runbooks
  - Read
  - Write
---

# _template (starter scaffold. Do not invoke)

> **This is a copy-paste starter for new contributors.** It is intentionally
> registered with `disable-model-invocation: true` so Claude never auto-loads
> it. Copy this directory to `plugins/profile-and-optimize/skills/<your-skill-name>/`,
> rename it, and edit every section before opening a PR.

## Where the new contributor starts

1. `cp -r plugins/profile-and-optimize/skills/_template plugins/profile-and-optimize/skills/<your-skill-name>`.
2. Open the new `SKILL.md` and rewrite the YAML frontmatter:
   - `name`: match the new directory name (lowercase + hyphens, max 64 chars).
   - `description`: third-person, includes WHAT + WHEN, max 1024 chars. **Remove `disable-model-invocation: true`** so the skill is discoverable.
   - `allowed-tools`: pin to the specific tools the workflow uses. See [`CONTRIBUTING.md`](/CONTRIBUTING.md#mcp-tool-naming-convention).
3. Rewrite every section below. Anything wrapped in `<...>` is a placeholder.
4. Read [`CONTRIBUTING.md`](/CONTRIBUTING.md) for the validation + version-bump + PR flow.
5. Run `make smoke-test` from the repo root to validate.

## Purpose

<One short paragraph. What does the skill do, and why does it exist?>

## When to use

- <Bulleted trigger scenario 1>
- <Trigger scenario 2>
- <Trigger scenario 3>

Do **not** use this skill for:

- <Overlapping skill or out-of-scope scenario 1>
- <Overlapping skill or out-of-scope scenario 2>

Cross-check the existing skills under `plugins/profile-and-optimize/skills/` before shipping - if your task overlaps with an existing skill, explain why a new one is warranted.

## Example prompts

<Several concrete operator prompts that should trigger this skill, both terse and verbose. These get folded into the description's trigger list.>

- "<terse trigger 1>"
- "<more conversational trigger>"
- `/<skill-name> --arg1 val1`

## Prerequisites

The skill **fails closed** if any of these are not satisfied.

1. <Env var or repo path>.
2. <Cluster reservation or MCP server reachability>.
3. <Operator confirmation flag, if mutating>.

## Interaction style

<Iterative pattern: one step, report, ask before continuing. Borrow language from
an existing skill such as
[`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md).>

## Workflow

### Phase 0: confirm intent

<Resolve the operator's request to specific parameters. State them back. Get confirmation.>

### Phase 1: <name>

<First tool call sequence. Include the exact MCP envelope.>

```text
mcp__profile_and_optimize__<verb> with:
  args: ["--flag", "<value>", "--json"]
```

Report: <what gets reported back>. Ask: <what the next decision is>.

### Phase 2: <name>

<Subsequent phase.>

### Phase N: <name>

<Final phase. Hand off to a sibling skill if applicable.>

## Verdict rigor (DRAFT vs VERDICT)

If this skill emits a performance claim, tier it per the
[`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md)
"Verdict rigor: DRAFT vs VERDICT" rule. Default every number to **DRAFT** (label it
provisional). Promote to a **VERDICT** only for a decision-grade claim AND only when
it is: variance-controlled (same-node, >=3 trials, mean +/- std), metric-isolated
(median TPOT/ITL for decode-latency claims - NOT output tok/s at small num_prompts),
compared to a production-representative baseline, and (for which-kernel claims)
backed by nsys/ncu per-kernel data. Under the always-publish policy
a campaign published as `verdict_tier=verdict` without this provenance is
auto-downgraded to `draft` and still lands (the honest tier is recorded on
`campaign_v1`), `publish_to_lake --strict` refuses instead. Supersede a DRAFT
everywhere it propagated once a VERDICT overturns it.

## Kernel rubric (K/R/H/P/A)

If this skill emits a **custom-kernel** comparison, classify the candidate AND the named
baseline along the five axes (K complexity / R representation / H hardware specialization
/ P perf target / A automation) and record both coordinate tuples in the bundle. A win
over a strictly-lower-H/R baseline
(e.g. beating generic Triton when production runs the `sm100f` tensor-core library) is a
**DRAFT, never a VERDICT** - the H + P proof (tensor-core engagement + roofline) comes
from [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md). Delete
this section if the skill never touches custom kernels.

## Full-context reporting (no bare numbers)

If this skill emits ANY performance number, it MUST carry the number's full
measurement-context descriptor and match every comparison on it, per
[`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md)
"Full-context reporting". A bare `tok/s` / TPOT / BW / %SoL / speedup is a
defect - it cannot set a default, ship a config, or appear in a report. Cite all that apply:
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named.
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").
Mechanically enforced for atlas-emitting paths by `methodology_problems()` (descriptor + per-row
ISL/OSL shape) and `shape_label_problems()` (no shared shape over heterogeneous cells) in the
perf-report `lake_writer.py` (publish/render `--strict` fail-closed). Delete this section ONLY if the
skill produces no measurements (read-only diagnostic).
- **Prefill/decode roofline (page 7) + obs/mechanisms:** a serving throughput/mixed measurement
  MUST carry the page-7 roofline (`import_roofline_sweep`, `publish_to_lake --strict` refuses a
  serving campaign that omits it) per `server/tools/perf_tune_report/ROOFLINE-METHODOLOGY.md`, and SHOULD ship
  the `findings/01-observations.md` (measured) + `findings/02-mechanisms.md`
  (`OBSERVATION -> MECHANISM -> CONFIDENCE`) split + a source-code link (provenance block).

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) "Always be grinding (next-lever framing)". A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
**Highlight NOVEL FRONTIER breakthroughs over config / well-known optimizations.** Two classes: the
CONFIG class (flag sweeps, batching, cudagraph/async toggles, kv-dtype, well-known serving knobs) is
table-stakes that rarely moves the SoL ceiling. The FRONTIER class (a custom or persistent megakernel,
op-chain fusion, a vendor kernel-occupancy/tiling fix, a quant-frontier format, a novel router /
parallelism / spec-decode design, an architecture-coupled kernel) is where the real unlock lives. A
skill's reported next-lever frontier MUST lead with the frontier class and name config levers as the
floor, not the headline -- and a frontier lever that is out-of-repo (vendor/upstream) is still a
first-class breakthrough to surface (with the substantiating profile), not a dead end.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. **Escalate the ladder: config
levers -> quant / spec-decode / parallelism -> NOVEL kernel-level (a megakernel / persistent-kernel
decode, FUSION of the per-step op chain, or a custom vLLM/SGLang kernel patch).** Config-exhaustion is
NOT frontier-exhaustion: a byte-grounded config-bound conclusion (occupancy / host-gap bound, tensor-pipe
far from roofline) is the START of the kernel hunt -- megakernel + fusion attack exactly that regime,
which a faster individual GEMM cannot. Never conclude "exhausted/optimal/done" without an explicit
next-lever frontier that INCLUDES the kernel-level candidates (each pursued, or banked with a K/R/H/P/A
structural-cap / infra-wall reason). An empty frontier requires a documented SoL wall AND the kernel
frontier assessed. Delete this section ONLY if the skill produces no measurements.

## Asset validation (review + FAIL LOUD)

If this skill emits a generated asset (visualization, chart, report, table, PDF, gallery,
exported data), it is a DELIVERABLE held to the
[`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) "Asset validation" rule: (1) the generator
**FAILS LOUDLY** on missing/bad/degenerate data -- source unreachable, zero rows when rows are
expected, a required field `unknown`/null/NaN, a near-empty PNG/PDF, a non-finite series -> raise
/ non-zero exit naming what is missing. NEVER a silent placeholder/fabricated asset at exit 0
(genuinely-optional absent data = an explicit labelled "no data" panel only). (2) the agent
**REVIEWS** the rendered asset -- opens the image / reads the report -- and confirms it is
ACCURATE (numbers + identities + matched comparisons trace to the data) AND MAKES SENSE to a
human (curves physically plausible, nothing mislabeled, empty == genuinely empty). If it is
wrong or confusing, **REBUILD and revisit** -- never ship a wrong/confusing asset with a caveat.
Delete this section ONLY if the skill produces no assets.

## PR value proposition (for shareable artifacts / PRs)

If this skill's output feeds a PR (especially an upstream serving-stack patch PR) or any
shareable/leadership artifact, that PR MUST lead with its **value proposition**: the WHY (blocker
removed / capability unlocked), the **measured benefit** (full descriptor + `%SoL`/`sol_rigor` + a
resolving data link. NEVER a hand-waved "no perf penalty"), an **applicability matrix** (model
architecture sparse/MoE-vs-dense, size, reasoning-vs-not, quant, parallelism incl. any
TP>1-specific behavior, workload, concurrency regime), and the **open gates**. A PR that only
states WHAT it changes is incomplete. Before opening or finalizing such a PR, run a
**claim-by-claim PR-body evidence audit**: every number resolves to a `campaign=<id>` / bundle
whose provenance `delivery`+`commit` MATCHES the code-under-test (a runtime `overlay` /
offline-prepped run is NOT evidence for an `infr-patch`, even if the kernels match -- the
code-under-test provenance must match). Confirm every cited campaign + link resolves,
reviewer-objection pass (impact scope, numerical approximation, honest gaps).
**Keep it tight (no AI-slop):** lead with SoL%/roofline% and matched numbers. Cut hedging,
redundancy, and decorative visuals (infra PRs do NOT embed charts). De-slop checklist: no
em-dashes (the #1 AI-slop tell), minimal bold (no bold-lead bullets), plain punctuation, inline
code out of narrow table cells.
Source: [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) "Value proposition" + "Full-context
reporting" + "De-slop". PR-facing surface of
[`inference-value-ledger`](/plugins/profile-and-optimize/skills/inference-value-ledger/SKILL.md).
Delete this section ONLY if the skill never feeds a PR / shareable artifact.

## Safety

- **<Ack flag mandatory / read-only / fail-closed condition>** per [`server/docs/mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).
- **<No silent fallbacks>** - fail fast rather than silently degrading.
- **<If this skill queries/consumes zymtrace: empty != gap.>** A zymtrace query empty right after a bench is usually ClickHouse INGEST LAG (async flush), not absence - wait + requery for the freshest data before concluding. See [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md). Delete if the skill never touches zymtrace.
- <Other forbidden actions: e.g. "read-only against external systems. No writes".>

## Source-of-truth references

- [`<runbook or doc path inside server/>`](../../server/<path>).
- [`<related skill>`](../<related-skill>/SKILL.md).
- [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) - the measurement-methodology canon.
