---
name: zymtrace-anchored-query
last_validated: 2026-05-24
description: >-
  Reusable wrapper for the knowledge-base-first SQL pattern,
  adapted to the zymtrace ClickHouse profiling backend. Operator names the
  metric / question / time range. Skill anchors the `zymtrace_profiling.events`
  schema first (DESCRIBE + label-value + cardinality probes), derives the
  safe narrow SQL, runs it via `kubectl port-forward` + `curl -X POST`, and
  saves the raw payload to a provenance-bearing bundle per the
  perf-lake-contract. Workload-agnostic. The ClickHouse cousin of
  `prometheus-anchored-query`. Triggers on "zymtrace anchored query",
  "clickhouse anchored query", "zymtrace query", "save zymtrace payload",
  "anchored zymtrace", "anchored clickhouse", or any combination of
  "zymtrace / clickhouse / profile" with "anchored / safe / provenance /
  query / saved-payload".
allowed-tools:
  - Bash(kubectl:port-forward,*)
  - Bash(kubectl:get,*)
  - Bash(curl:*)
  - Bash(jq:*)
  - Bash(sha256sum:*)
  - Bash(date:*)
  - Read
  - Write
---

# zymtrace-anchored-query

## Purpose

Make ad-hoc ClickHouse queries against the
[zymtrace](https://docs.zymtrace.com) profiling backend **safe by
default** by wrapping the "anchor labels in the schema
FIRST, then run the query" discipline as a reusable skill. Saves the
raw payload to a provenance-bearing bundle so future queries can replay
the same shape, and so the perf-lake-contract is honored without
operator effort.

This skill exists because the `zymtrace_profiling.events` table on a
busy cluster holds tens of millions of rows per day across every
profiler-enabled namespace. An SQL query that forgets to bound
`timestamp` or `pod_name LIKE '...'` returns a full-table scan that
overloads ClickHouse and starves real-time ingest. The anchor-then-query
pattern works for arbitrary profiling questions. This skill makes it
agent-routable.

[`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md) is
the Prometheus version of the same pattern. Use this
when the question targets profiling data (CPU on-cpu samples, GPU cuda
kernel samples, stack-traces, function-resolution lookups). Use the
Prometheus skill when the question targets time-series cluster metrics.

## When to use

- Operator has a profiling question that maps to a ClickHouse query but
  doesn't know the exact `pod_name` pattern, time-window, or
  `event_kind` to use yet.
- Operator wants to save a zymtrace payload to disk with provenance
  (for evidence bundles, profile-findings write-ups, share-with-team
  artifacts, or feeding the optional `kernel_class_gpu_pct` +
  `cpu_spinpoll_pct` fields into the
  [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md)
  schema).
- Operator wants to confirm a `pod_name` LIKE pattern matches the
  expected pod set + that the time window has enough samples to be
  statistically useful before running a full top-N query.

Do **not** use this skill for:

- Already-vetted profile queries - just run the SQL directly via
  `curl -X POST 'http://localhost:9123/' --data-binary @<query.sql>`.
- Live UI exploration - the zymtrace web UI has its own
  flamegraph/timeline views. This skill is for headless reproducible
  queries.
- Cross-tenant queries - every example here scopes to a single
  `pod_name LIKE` pattern. Cross-tenant exploration belongs to the
  zymtrace UI or the zymtrace MCP's pattern-recognition tools.

## Example prompts

- "What are the top 25 hottest CUDA kernels in `basic-inference` pods over the last 5 minutes?"
- "Save the on_cpu hot-Python-callers query for `glm-inference` between 16:00 and 16:30 UTC."
- "Did `shm_broadcast.py` CPU time in the `glm-inference` pods drop after we set VLLM_USE_SHM_BROADCAST_BLOCKING=1? Compare the last 30 minutes against the window before the change."
- "Get me the kernel-class breakdown for the `kimi-inference` pods during the perf-bench window I just ran so I can fill `kernel_class_gpu_pct` in the perf-baseline bundle."
- `/zymtrace-anchored-query --question "top-25 cuda kernels for basic-inference pods, last 5 min" --time-range "now-5m..now"`
- `/zymtrace-anchored-query --pod 'basic-inference%' --event-kind cuda --window '<start-UTC>..<end-UTC>' --top 25`

## Prerequisites

1. **Cluster context active** - `kubectl config current-context` returns
   the cluster running zymtrace.
2. **zymtrace backend deployed** - `kubectl -n zymtrace get pods -l app=zymtrace-clickhouse` shows at least one Running pod.
3. **Operator question or pod+kind+window** - one of:
   - Free-form question (skill derives `pod_name` pattern + `event_kind` + window via schema introspection).
   - Explicit `--pod <LIKE-pattern> --event-kind <cuda|on_cpu> --window <UTC-range>`.
4. **Time window** - `--time-range "now-Nm..now"` or absolute UTC range. Default `now-5m..now`. Queries with no `timestamp` bound are refused.
5. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for the bundle path (falls back to `${PWD}/zymtrace-bundles/<run-id>/`).

## Interaction style

Iterative. The whole point of the skill is the "anchor first, then
query" discipline. Pause after the schema-anchor call to surface the
proposed SQL before running it.

## Workflow

### Phase 0: resolve the question

If operator provided a free-form question, restate the question + the
proposed `pod_name` LIKE pattern, `event_kind`, and time window in one
sentence. Get confirmation.

If operator provided explicit pod + kind + window, proceed.

### Phase 1: schema + label anchor

Start the port-forward if not already running:

```bash
# Idempotent: skip if a port-forward to svc/zymtrace-clickhouse is already up on 9123.
if ! curl -fsS -o /dev/null -m 2 'http://localhost:9123/ping'; then
  nohup kubectl -n zymtrace port-forward svc/zymtrace-clickhouse 9123:8123 \
    > /tmp/zt-ch-pf.log 2>&1 &
  disown
  sleep 3
fi
```

Probe the schema for the table the query will hit (default
`zymtrace_profiling.events`):

```bash
curl -fsS -X POST 'http://${CH_USER}:${CH_PASSWORD}@localhost:9123/' \
  --data-binary 'DESCRIBE TABLE zymtrace_profiling.events FORMAT JSON' \
  > /tmp/zt-schema-anchor.json
```

Then sanity-check cardinality of the proposed selector before running
the full top-N query:

```bash
curl -fsS -X POST 'http://${CH_USER}:${CH_PASSWORD}@localhost:9123/' \
  --data-binary "
SELECT
  count() AS rows,
  uniqExact(pod_name) AS pods,
  min(timestamp) AS first_ts,
  max(timestamp) AS last_ts
FROM zymtrace_profiling.events
WHERE pod_name LIKE '<pattern>'
  AND event_kind = '<kind>'
  AND timestamp >= '<start>'
  AND timestamp <= '<end>'
FORMAT JSON
"
```

The cardinality probe response tells us:

- Whether the selector matches **any** rows (catches typos in
  `pod_name` LIKE patterns immediately).
- The exact `pod_name` set the query will scan (catches accidental
  cross-pod scope).
- The actual `timestamp` range that has data (catches misaligned
  windows - e.g. a window anchored on the bench's scheduled start when
  the run actually began and ended a few minutes later, a common
  first-pass mistake).
- The sample count (so the operator can decide if the window is too
  short to be statistically meaningful - <1000 samples is typically
  not enough).

If `rows = 0`: do NOT immediately conclude "no signal". Zymtrace data is not
instantaneous - rank two causes in this order:

1. **Ingest lag (transient).** If the pod/window are freshly-run (the bench just
   ended), zymtrace may not have flushed to ClickHouse yet - it flushes
   **asynchronously** (~seconds-to-minutes). **Wait for the flush and re-run the
   cardinality probe** for the freshest data before concluding. A simple poll
   (mirrors [`scripts/zymtrace-ingest-wait.sh`](/scripts/zymtrace-ingest-wait.sh)):

   ```bash
   for i in $(seq 1 "${ZYM_INGEST_MAX_ATTEMPTS:-6}"); do
     rows=$(curl -fsS -X POST 'http://${CH_USER}:${CH_PASSWORD}@localhost:9123/' \
       --data-binary "<the cardinality probe SELECT ... FORMAT JSON>" | jq -r '.data[0].rows')
     [ "${rows:-0}" -gt 0 ] && break
     sleep "${ZYM_INGEST_WAIT_SEC:-20}"   # requery for the freshest data
   done
   ```

2. **Wrong selector / window.** If it STAYS 0 past the flush window, the
   `pod_name LIKE` pattern or `[start,end]` doesn't match - re-query by
   `host=<node>`+window (the hash-suffixed pod-name filter often reads empty when
   the host filter returns frames) and confirm the window.

Per the fail-fast rationale in
[`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md),
running the top-N on an empty result would look like "no signal" when it is really
either ingest lag (requery) or a wrong selector (fix it) - never silently proceed.

If `rows > 10_000_000`: refuse. The selector is too broad. Narrow the
time window or `pod_name` pattern.

### Phase 2: emit the SQL

Construct the narrow SQL using only schema-anchor-confirmed labels and
the operator's `--time-range`. The SQL **must** include all three
bounds:

```sql
WHERE pod_name LIKE '<pattern>'
  AND event_kind = '<kind>'
  AND timestamp >= '<start>'
  AND timestamp <= '<end>'
```

Print the proposed SQL back:

```text
Proposed SQL: <full query>
Anchor:       pod_name LIKE '<pattern>', event_kind = '<kind>'
Time range:   <start> .. <end>
Expected rows scanned: ~<from cardinality probe>
LIMIT clause: <N rows> (default 25)
```

Ask: "Run this query?" Do not auto-advance.

### Phase 3: run + save

After confirmation, run the query via curl + save the raw response:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE_DIR="${PROFILE_AND_OPTIMIZE_REPO_ROOT:-${PWD}}/experiments/artifacts/zymtrace-bundles/${RUN_ID}"
mkdir -p "${BUNDLE_DIR}"

cat > "${BUNDLE_DIR}/query.sql" <<'EOF'
<the exact SQL from Phase 2>
EOF

curl -fsS -X POST 'http://${CH_USER}:${CH_PASSWORD}@localhost:9123/' \
  --data-binary "@${BUNDLE_DIR}/query.sql" \
  -H 'X-ClickHouse-Format: JSON' \
  -o "${BUNDLE_DIR}/response.json"

sha256sum "${BUNDLE_DIR}/query.sql" "${BUNDLE_DIR}/response.json" \
  > "${BUNDLE_DIR}/sha256sums.txt"
```

Save the bundle layout (parallels `prometheus-bundles/<run-id>/` for
consistency):

```
${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/zymtrace-bundles/<run-id>/
  SOURCE.md                     # operator + question + cluster + UTC-ts + git SHA
  query.sql                     # the exact SQL run
  query.json                    # the curl request shape (URL, headers, body-ref)
  response.json                 # the raw ClickHouse response (saved verbatim)
  summary.md                    # human-readable: row count, top-5 by pct, anchor recap
  schema-anchor.json            # the DESCRIBE + cardinality-probe responses (provenance)
  sha256sums.txt                # query.sql + response.json sha256
```

### Phase 4: report

Print:

- **Row count** + **scanned-rows count** (sanity check: did we get the
  cardinality the anchor predicted?).
- **Top 5 rows** of the response (highest `pct` or `total_samples`).
- **Bundle path** for the saved payload.
- **Cross-link** to
  [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md)
  if the operator's question maps to a `kernel_class_gpu_pct` or
  `cpu_spinpoll_pct` field (it usually does - that's the canonical
  consumer of this skill's output).

## Kernel-class bucketing (appendix)

The
[`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md)
schema's `kernel_class_gpu_pct` field is derived by bucketing kernel
names from this skill's GPU-side response. The canonical regex set
covers the kernel classes that dominate FP8 MoE serving on B200-class
hardware:

| Bucket key | Regex on `kernel` (func_name) | What it catches |
| --- | --- | --- |
| `fp8_bmm` | `^bmm_(E4m3|Bfloat16)_E4m3E4m3` OR `sm100_fp8_gemm` OR `deep_gemm::sm100_fp8` | FP8 block-quantized batched matmul (the dominant compute on B200 Kimi/DeepSeek) |
| `tp_all_reduce` | `multimem_all_reduce_kernel` OR `trtllm_allreduce` OR `oneshot_lamport` | Tensor-parallel all-reduce kernels (both PyTorch CUDASymmetricMemoryOps + flashinfer paths) |
| `cuda_event_sync` | `cudaEventSynchronize` OR `cuda-sync` | Host-side CUDA event sync (sign of cudagraph coverage gaps) |
| `mla_attention` | `fmhaSm100fKernel` OR `cute_flash_fwd_sm100` OR `mla_attention` | DeepSeek-V3-style MLA attention (Q576/V512 paged-KV) |
| `moe_routing_finalize` | `moe::dev::(routing|finalize|activation)` | DeepSeek MoE routing + finalize kernels |
| `per_token_group_quant` | `per_token_group_quant_8bit` | Block-FP8 activation quantization (group_size=128) |

Apply the regex against `kernel` in the GPU response, sum the `pct` per
bucket, leave classes outside this set under arbitrary `<other-class>`
keys (e.g. `nvjet_trt_llm`, `d2d_memcpy`, `elementwise_misc`).

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

- **Empty != no data (ClickHouse ingest lag).** A `rows = 0` on a freshly-run
  pod/window is often ingest lag, not absence - zymtrace flushes to ClickHouse
  asynchronously. **Wait for the flush and re-probe** (`ZYM_INGEST_WAIT_SEC` /
  `ZYM_INGEST_MAX_ATTEMPTS`) before concluding. Only a persistent 0 past the flush
  is a wrong selector / real gap. See
  [`server/docs/zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md).
- **Schema-anchor first is mandatory.** Phase 1 cannot be skipped. The
  whole point of the skill is to enforce that discipline.
- **All three bounds required.** Every SQL the skill emits has
  `pod_name LIKE`, `event_kind =`, and `timestamp >=`/`<=`. Queries
  missing any of the three are refused.
- **No full-table scans.** If the cardinality probe reports >10M rows
  scanned, the skill refuses to run the top-N. Operator narrows the
  selector and retries.
- **LIMIT 25 default.** The default top-N is 25 rows. Operator can
  override to as many as 100. Anything larger is refused - the bundle
  is for human-readable summary, not bulk export.
- **Raw payload preservation.** Per
  [`server/docs/perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md),
  every saved query records the SQL, the schema anchor, the time
  range, the response time, and the cluster context.
- **Read-only.** The skill only runs `SELECT` / `DESCRIBE` / `SHOW`
  statements. Never `INSERT` / `ALTER` / `DROP` / `TRUNCATE`. The
  ClickHouse credential the port-forward uses
  (`${CH_USER}:${CH_PASSWORD}`) is the default in-cluster read/write
  account, but this skill self-restricts to read-only via its
  query-construction logic - operator-supplied SQL that contains any
  non-`SELECT` verb is refused before the curl.
- **Port-forward hygiene.** The `nohup kubectl port-forward` is left
  running in `/tmp/zt-ch-pf.log`. Cleanup is operator-driven. The
  skill does not kill the port-forward on exit (subsequent queries
  re-use it).
- **No credentials in conversation.** The curl lines read `CH_USER` /
  `CH_PASSWORD` from the environment. Nothing is hard-coded here.
  Never paste a real production password into the conversation or any
  saved bundle.

## Source-of-truth references

- [`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md) - the Prometheus cousin of this skill.
- [`inference-perf-baseline-bridge`](/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/SKILL.md) - primary consumer. Derives `kernel_class_gpu_pct` + `cpu_spinpoll_pct` from this skill's output.
- [`analyze-zymtrace-workload`](/plugins/profile-and-optimize/skills/analyze-zymtrace-workload/SKILL.md) - the zymtrace MCP analytical workflow (different layer. Pattern-recognition over the same data).
- Bundled server [`AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) - fail-fast + provenance rules.
