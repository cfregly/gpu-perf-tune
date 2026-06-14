Status: Active
Audience: operators and contributors using the fleet-wide profiling tooling.

# profiling

Fleet-wide profiling tools for MLPerf Training cohorts.

This package implements the fleet-wide profiling and hang-detection
stack designed in
[`../../docs/profiling-and-perf-discovery.md`](/plugins/profile-and-optimize/server/docs/profiling-and-perf-discovery.md)
section `## v6.1 carry-forward: fleet-wide profiling and hang-detection stack`.

## Subpackages

- [`hang_detector/`](/plugins/profile-and-optimize/server/tools/profiling/hang_detector) - fleet-wide NCCL collective hang
  detector. Consumes GPUSD-published per-rank `(rank, seq_num,
  timestamp)` metadata, buckets ranks by `rank % stride`, and emits
  structured alerts when a bucket's median seq_num lags the leader
  bucket. The default `stride=32` matches the MOD-32 hang signature
  (see [`../../docs/learnings/healthverification-mod32-probe-spec.md`](/plugins/profile-and-optimize/server/docs/learnings/healthverification-mod32-probe-spec.md)).

## Quickstart

Replay the bundled 2048N MOD-32 hang fixture and see the canonical alert:

```bash
python -m tools.profiling.hang_detector \
  --fixture tools/profiling/hang_detector/tests/fixtures/gpusd-snapshot-2048n-mod32-hang.json \
  --stride 32 --json
```

Run against a healthy snapshot (no alerts):

```bash
python -m tools.profiling.hang_detector \
  --fixture tools/profiling/hang_detector/tests/fixtures/gpusd-snapshot-healthy.json \
  --stride 32
```

Live-cluster operator path (requires `requests`):

```bash
# Capture the live nodelist for an active job.
sacct -j ${JOBID} --format=NodeList%2000 -h | tr , '\n' > /tmp/nodes.txt

python -m tools.profiling.hang_detector \
  --live-cluster --nodelist-file /tmp/nodes.txt \
  --stride 32 --jobid ${JOBID} \
  --output experiments/artifacts/profiling/hang-detector/${JOBID}/timeline.jsonl
```

The orchestrator appends one JSONL row per invocation, so a polling
loop (`while true; do ...; sleep 5; done`) produces a complete
timeline.

## Tests

```bash
python -m pytest tools/profiling/hang_detector/tests/ -x -q
```

All tests use synthetic JSON fixtures. No live cluster dependency.

## Cross-references

- Design appendix: [`../../docs/profiling-and-perf-discovery.md`](/plugins/profile-and-optimize/server/docs/profiling-and-perf-discovery.md)
  "fleet-wide profiling and hang-detection stack".
- Failure mode worth knowing: fleet-wide polling can exhaust file
  descriptors on the collector host. That finding drives the per-rack
  connection pool.
