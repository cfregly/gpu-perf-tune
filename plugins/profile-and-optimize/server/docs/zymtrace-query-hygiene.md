# Zymtrace query hygiene: empty-now != gap (ClickHouse ingest lag)

Single source of truth for the **wait + requery** discipline when a zymtrace query
comes back empty. Cited by the zymtrace skills (`zymtrace-anchored-query`,
`analyze-zymtrace-workload`), the per-kernel importer
(`tools/perf_tune_report/importers/zymtrace_kernels.py`), and the skill `_template`.
The capture-side mechanism (a poll-until-rows loop) lives in
`scripts/zymtrace-ingest-wait.sh`. The policy
canon is `docs/METHODOLOGY.md` "Zymtrace data is not instantaneous
(ClickHouse ingest lag)".

## Why a zymtrace query can be empty even though capture worked

Zymtrace profiling data is **not instantaneous**. The per-pod CUDA implant
accumulates samples in-process and flushes them to the backend (ClickHouse)
asynchronously, on an interval that is typically seconds but can be ~minutes
under load. The zymtrace MCP (`topfunctions` / `flamegraph`) and direct
ClickHouse `SELECT`s read whatever has landed **so far**. So a query issued at,
or just after, the end of a bench window can legitimately return **zero rows
even though the workload was profiled correctly** - the frames simply are not
queryable *yet*.

This is the sibling of the nsys "empty != blind spot" gate
(`docs/METHODOLOGY.md`): there an empty `cuda_gpu_kern_sum`
is a capture-hygiene bug. Here an empty query is **ingest lag**, and the fix is
to wait for the flush and requery for the freshest data - NOT to declare a gap.

## The three things an empty result can mean (rank them in this order)

1. **Ingest lag (transient).** You queried inside / right after the window and
   the flush has not landed yet. **Wait for the flush interval and requery.** This
   is the default first hypothesis for a freshly-run workload.
2. **Wrong selector / window (operator error).** The `pod_name LIKE` pattern,
   `event_kind`, or `[start,end]` does not match the data. Re-query by
   `host=<node>` + window (the hash-suffixed pod-name filter often returns empty
   even when the host filter returns frames) and confirm the window matches the
   actual run.
3. **Real telemetry gap (capture failure).** Only after (1) and (2) are ruled
   out: the injection never intercepted the pod, the implant was absent, or the
   pod exited before flushing. Run `capture-run-env.sh` (the `gpu_frames_gate`)
   and the implant/intercept checks in `PROFILING-RUNBOOK.md`.

## The wait + requery recipe

Poll a cheap count probe with backoff until it returns a positive count, then run
the real (heavier) query against the now-present data:

```bash
# Conceptually (capture-sol-window.sh does this automatically via the shared helper):
attempt=1
until [ "$(count_probe)" -gt 0 ] || [ "$attempt" -gt "${ZYM_INGEST_MAX_ATTEMPTS:-6}" ]; do
  sleep "${ZYM_INGEST_WAIT_SEC:-20}"   # default 6 x 20s ~= 120s, outlasts the ~60s flush
  attempt=$((attempt + 1))
done
```

- The canonical implementation is `zym_wait_for_rows` in
  `scripts/zymtrace-ingest-wait.sh` (sourced by
  `capture-sol-window.sh`). Tune with `ZYM_INGEST_WAIT_SEC`,
  `ZYM_INGEST_MAX_ATTEMPTS`, `ZYM_INGEST_BACKOFF`. Set `ZYM_INGEST_DISABLE=1`
  when backfilling an old, already-flushed window from retained telemetry.
- For self-driving capture pods, **hold the pod ~60s past the bench before exit**
  so the implant flushes before the pod (and its frames) go away
  (`PROFILING-RUNBOOK.md`).
- The wait is **advisory**: it delays until data is present or the poll is
  exhausted. It does not fabricate data. After the poll, the existing empty-output
  check is what decides a real gap - now legitimately, because you waited.

## For the importer (consumes already-captured TSVs)

`zymtrace_kernels.py` reads a static `<bundle>/zymtrace/*.tsv` snapshot. It cannot
requery ClickHouse. It therefore stays **fail-fast** (a header-only / empty TSV is
a loud `ZymtraceTSVMalformed` / `ZymtraceTSVMissing`, never a silent pass) but its
error message names ingest lag as a likely cause and points back here: the fix is
to re-capture after the flush (the capture script now polls), not to weaken the
importer.
