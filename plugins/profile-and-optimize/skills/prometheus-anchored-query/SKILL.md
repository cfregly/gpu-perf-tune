---
name: prometheus-anchored-query
last_validated: 2026-05-21
description: >-
  Reusable wrapper for the knowledge-base-first PromQL pattern.
  Operator names the metric / question / time range. Skill calls
  query_observability_knowledge_base first to confirm labels + cardinality,
  derives the safe narrow PromQL, runs it via the Prometheus MCP, and saves the
  raw payload to a provenance-bearing bundle per the perf-lake-contract.
  Workload-agnostic. Replaces the ad-hoc "run PromQL and hope" pattern with
  a disciplined anchor-then-query workflow. Triggers on
  "prometheus query", "promql query", "anchored query", "knowledge-base
  query", "query the Prometheus MCP", "what's the prometheus metric for", "save
  prometheus payload", or any combination of "promql / prometheus / metric /
  query" with "anchored / knowledge-base / safe / provenance".
allowed-tools:
  - mcp__prometheus_mcp__query_observability_knowledge_base
  - mcp__prometheus_mcp__query_prometheus
  - mcp__prometheus_mcp__list_prometheus_label_values
  - mcp__prometheus_mcp__list_prometheus_label_names
  - mcp__prometheus_mcp__list_prometheus_metric_names
  - mcp__prometheus_mcp__list_prometheus_metric_metadata
  - mcp__prometheus_mcp__list_datasources
  - mcp__prometheus_mcp__get_datasource
  - Bash(date:*)
  - Bash(jq:*)
  - Read
  - Write
---

# prometheus-anchored-query

## Purpose

Make ad-hoc PromQL queries **safe by default** by wrapping the "Contention And O11y Pattern" discipline (anchor labels in the knowledge base FIRST, then run the query) as a reusable skill. Saves the raw payload to a provenance-bearing bundle so future queries can replay the same shape, and so the perf-lake-contract is honored without operator effort.

This skill exists because PromQL is easy to write **wrong** at cluster scale: a missing label selector turns a 100-series query into a 10,000-series query that overloads the Prometheus backend. The bundled server's [`mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md) Contention And O11y Pattern (step 3) makes the knowledge-base-first call mandatory. This skill enforces it.

Use it for ad-hoc questions ("what's the IB traffic on `<node>` right now?" / "is the dataloader being starved?" / "what's the median dcgm_xid rate across a partition?").

## When to use

- Operator has a question that maps to a PromQL query but doesn't know the exact labels yet.
- Operator wants to save a Prometheus payload to disk with provenance (for evidence bundles, post-incident write-ups, share-with-team artifacts).
- Operator wants to confirm a metric exists + has the expected cardinality before writing a dashboard panel for it.

Do **not** use this skill for:

- Already-vetted queries - call `mcp__prometheus_mcp__query_prometheus` directly (or run them from whatever dashboarding UI they live in). The anchor pass adds nothing when the query shape is already known.
- Log queries - this skill emits PromQL only. The MCP server also fronts a log datasource via `query_loki_logs`, but anchored log queries are out of scope here.

## Example prompts

- "What's the IB traffic on these 8 nodes for the last hour?"
- "Is dataloader starvation showing up in dcgm_gpu_utilization for the cohort I just ran?"
- "Save the dcgm_xid trajectory across the gb300 partition for the last 24h."
- "What's the median GPU temp on this rack right now?"
- "Anchored PromQL for nccl_perf_bandwidth across the cohort."
- `/prometheus-anchored-query --question "IB tx bytes by node for <partition> last 1h" --time-range "now-1h..now"`
- `/prometheus-anchored-query --metric dcgm_gpu_temp --labels '{partition="gb300"}' --time-range "now-24h..now"`

### Inference-perf example

The [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) workflow scrapes four canonical vLLM Prometheus metrics pre/post each benchmark run:

| Metric | What it answers |
| --- | --- |
| `vllm:prefix_cache_hit_rate` | KV cache hit rate during the run (delta pre vs post) |
| `vllm:gpu_cache_usage_perc` | GPU cache utilization peak |
| `vllm:num_requests_running` | Concurrent requests peak |
| `vllm:avg_generation_throughput_toks_per_s` | Generation throughput |

Operators replaying a perf-bench run via this skill should anchor against these four metric names first (knowledge-base call confirms cardinality + datasource UID), then run the narrow PromQL with `{model_name="<served-name>", namespace="<ns>"}` selectors. See [`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md) Phase 4 / 7 for the in-pod scrape pattern this skill complements at the Prometheus MCP layer.

## Prerequisites

1. **Prometheus MCP reachable** (`PROMETHEUS_MCP_URL` set).
2. **Operator question or metric+labels** - one of:
   - Free-form question (skill derives metric + labels via knowledge base).
   - Explicit `--metric <name> --labels <json>`.
3. **Time range** - `--time-range "now-Nh..now"` or absolute UTC range. Default `now-1h..now`.
4. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for the payload bundle path (falls back to `${PWD}/prometheus-bundles/<run-id>/`).

## Interaction style

Iterative. The whole point of the skill is the "anchor first, then query" discipline. Pause after the knowledge-base call to surface the proposed PromQL before running it.

## Workflow

### Phase 0: resolve the question

If operator provided a free-form question, restate the question + the proposed metric / labels in one sentence. Get confirmation.

If operator provided explicit metric + labels, proceed.

### Phase 1: knowledge-base anchor

```text
mcp__prometheus_mcp__query_observability_knowledge_base with:
  query: "<one-line question or metric+context>"
  user_prompt: "<operator's original prompt>"
```

The knowledge-base response tells us:

- Whether the metric exists in the Prometheus MCP's catalog.
- The available labels for that metric.
- The expected cardinality (so we don't accidentally select 100k series).
- The canonical datasource UID for this metric on this cluster (the MCP can front several datasources - e.g. per-cluster Prometheus instances plus a log backend - so every query tool takes an explicit `datasourceUid`).
- Any noted caveats (sampling rate, retention, legacy renames).

If the knowledge base does NOT cover the metric: stop. Report what's missing. Per [`mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md) "Contention And O11y Pattern" step 3, running uncovered PromQL is unsafe.

### Phase 2: validate labels exist

For each label the proposed query selects on:

```text
mcp__prometheus_mcp__list_prometheus_label_values with:
  datasourceUid: "<from knowledge-base anchor in Phase 1>"
  labelName: "<label>"
  matches: ["<metric>{<other-selectors>}"]
  user_prompt: "<operator's original prompt>"
```

If a label or value doesn't exist, the query would be silently empty. Stop and surface.

### Phase 3: emit the PromQL

Construct the narrow PromQL using only knowledge-base-confirmed labels and the operator's `--time-range`. Print it back:

```text
Proposed PromQL: <full query>
Datasource UID:  <from knowledge base>
Time range:      <start> .. <end>
Expected series count: ~<from knowledge base cardinality estimate>
```

Ask: "Run this query?" Do not auto-advance.

### Phase 4: run + save

After confirmation:

```text
mcp__prometheus_mcp__query_prometheus with:
  datasourceUid: "<from-anchor>"
  expr: "<promql>"
  queryType: "range"
  startTime: "<start>"
  endTime: "<end>"
  stepSeconds: <resolution-seconds>
  user_prompt: "<operator's original prompt>"
```

(Use `queryType: "instant"` + only `endTime` for a single-point query. Drop `startTime` / `stepSeconds` in that case.)

Save the raw payload to:

```
${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/prometheus-bundles/<run-id>/
  SOURCE.md                     # operator + question + cluster + UTC-ts
  query.promql                  # the exact PromQL run
  query.json                    # the exact request JSON
  response.json                 # the raw Prometheus response (saved verbatim)
  summary.md                    # human-readable summary: series count, range, top-5 anomalies
  knowledge-base-anchor.json    # the knowledge-base response (provenance for the label discipline)
```

### Phase 5: report

Print:

- **Series count** + **time-bucket count** (sanity check: did we get the cardinality the knowledge base predicted?).
- **Top 5 anomalies** (highest / lowest values, recent step-deltas).
- **Bundle path** for the saved payload.

## Full-context reporting (no bare numbers)

Per `docs/METHODOLOGY.md` "Full-context reporting": every number this
skill emits (throughput, latency, TPOT/ITL, BW, %SoL, speedup, efficiency, goodput, acceptance
rate, scaling efficiency, thermal/failure rate - whatever it reports) MUST carry its full
measurement-context descriptor, and every comparison MUST be matched on it. A bare number is a
defect - it cannot set a default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named. (If the metric is not roofline-bound - e.g. accuracy/acceptance - omit `%SoL` but keep the rest of the descriptor.)
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per `docs/METHODOLOGY.md` "Always be grinding". A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.

## Safety

- **Knowledge base first is mandatory.** Phase 1 cannot be skipped - fail fast, no silent fallbacks. The whole point of the skill is to enforce that discipline.
- **No high-cardinality queries.** If the knowledge base reports the proposed query would return >10k series, the skill refuses to run it. Operator narrows the selector and retries.
- **Raw payload preservation.** Per [`perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md), every saved query records the datasource UID, the exact PromQL, the time range, the response time, and the source label.
- **Read-only.** The skill never mutates state through the Prometheus MCP. Never creates dashboards / annotations / alerts (those would need `create_*` tools which are not in `allowed-tools`).

## Known limitations

- **The agent's MCP token may lack `datasources:read`.** If `mcp__prometheus_mcp__query_prometheus` returns `403 datasources:read on <uid>` for every Prometheus datasource UID, Phases 2-4 of this skill are blocked end-to-end. **Workaround:** the operator runs the exact PromQL produced in Phase 3 from an authenticated shell or the Prometheus web UI and pastes the response JSON into `${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/prometheus-bundles/<run-id>/response.json`. The skill's Phase 5 summary still works against the operator-pasted payload.

## Source-of-truth references

- [`server/docs/mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md) - "Contention And O11y Pattern" (the workflow this skill generalizes).
- [`server/docs/perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md) - raw-payload provenance contract.
- [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) - measurement canon (full-context reporting, verdict rigor).
