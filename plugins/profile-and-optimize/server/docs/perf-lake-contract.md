Status: Active
Audience: operators and reviewers validating metrics-backed evidence provenance.

# Perf Lake Contract

ClickHouse is the current analytical source for MLPerf performance and
fabric-derived evidence. Agents and tools should consume it when it materially
improves node selection, launch gating, contention diagnosis, reliability
rollups, or performance attribution.

For Grafana-scoped ClickHouse evidence in Cursor, prefer the observability
MCP tools: call `list_clickhouse_tables` first, then `query_clickhouse`
against your canonical Grafana ClickHouse datasource UID.
The generic ClickHouse plugin remains an operator-side read-only path when a
service ID is known, but repo tools must not depend on that workstation-local
plugin.

The consumer contract is intentionally lake-neutral. Current provenance may
reference ClickHouse, Prometheus/VictoriaMetrics, or Grafana diagnostics. A
future perf lake may be served through ClickHouse, Parquet, Iceberg, Spark,
Trino, StarRocks, or another engine, but agent workflows should keep the same
evidence shape and provenance requirements.

Current contract version: `mlperf-perf-lake.v1`.

## Required Provenance

Every metrics-backed artifact derived from ClickHouse or a future perf lake
must record:

- Source backend and datasource name or URI.
- Query text, dashboard/panel ID, or saved-query identifier.
- Query time range and freshness timestamp.
- Schema or contract version used by the consumer.
- Join keys used to connect rows to MLPerf evidence: `node`, `hostname`, `BMN`,
  Slurm job ID, benchmark ID, run ID, and campaign or workstream ID where
  available.
- Raw payload path under `experiments/artifacts/<family>/<run-id>/`.
- Normalized artifact path and format (`json`, `jsonl`, `csv`, `parquet`, or
  another explicit format).
- Coverage summary: rows read, nodes/BMNs matched, nodes/BMNs missing, and any
  degraded-mode flag.

## Fail-Closed Coverage

Missing metrics are not neutral evidence. If a launch gate, selector score,
contention decision, bad-node handoff, or performance recommendation depends on
perf-lake data, the workflow must fail closed when:

- The backend is unreachable.
- The query is stale, broad, or under-scoped.
- Required labels or join keys are absent.
- Node/BMN coverage is incomplete for the requested cohort.
- The raw payload cannot be saved beside the derived artifact.

An operator may choose an explicit degraded mode only when the runbook permits
it. The resulting artifact must name the missing source, affected nodes/BMNs,
and the reason the degraded result is still useful.

## Migration Boundary

Tools should keep backend-specific details at the ingestion edge. Downstream
agent and validator workflows should consume normalized evidence files under
`experiments/artifacts/`, not ad hoc ClickHouse response shapes.

When adding a new perf-lake consumer:

- Write raw backend payloads first, then write normalized artifacts.
- Preserve stable field names for benchmark, run, node, BMN, metric, value,
  unit, time range, and provenance.
- Make schema changes as strict supersets and update readers in the same
  change.
- Keep query examples narrow and reproducible. Never introduce a broad global
  query as a default.
- Document whether the data is authoritative for launch gating, advisory for
  ranking, or diagnostic-only.

## Backends And Migration

The active source for backend-sweep and NCCL-test evidence is ClickHouse.
Pick one Grafana ClickHouse datasource, treat its UID as canonical for these
queries, and use `perf` as the default database.

A longer-term perf lake typically evolves the same way: ClickHouse snapshots
or live extracts land in object storage, become Parquet/Iceberg objects, are
exposed through Spark / Trino / StarRocks-style query engines, and then feed
saved query components and MCP tools. This repo must not depend on any such
runtime directly. It depends only on the normalized artifact and
`mlperf-perf-lake.v1` provenance shape.

## Worked Example: Backend Sweep Outliers

The query reads the current ClickHouse source table
`perf.backend_sweep_nccl_tests` and filters by time range plus low
`bw_average` values before any downstream join, consistent with ClickHouse
filtering guidance.

Narrow query shape:

```sql
WITH low_rows AS (
  SELECT
    client_node AS bmn,
    bw_average,
    gid_index,
    client_lg AS lg,
    client_nvl_domain AS nvl_domain
  FROM perf.backend_sweep_nccl_tests
  WHERE $__timeFilter(ts_start)
    AND client_nvl_domain LIKE '%<ZONE>'
    AND bw_average > 0
    AND bw_average < 150
)
SELECT bmn, count() AS low_rows, min(bw_average) AS min_bw
FROM low_rows
GROUP BY bmn
ORDER BY low_rows DESC
LIMIT 100
```

Normalize the raw `query_clickhouse` export into an artifact directory under
`experiments/artifacts/perf-lake/<run-id>/`, recording the source table, the
narrow SQL above, the exact time range, and the join key (`bmn`).

The output directory contains:

- `perf_backend_sweep_nccl_tests.raw.json` - raw `query_clickhouse` payload.
- `perf_backend_sweep_nccl_tests.normalized.jsonl` - one typed row per returned
  record, preserving original columns plus normalized join keys.
- `provenance.json` - `mlperf-perf-lake.v1`, source datasource, query text,
  time range, join keys, row coverage, raw path, normalized path, and
  degraded-mode state.
