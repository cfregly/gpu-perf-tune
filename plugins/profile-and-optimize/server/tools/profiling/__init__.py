"""MLPerf profiling tooling.

Subpackages:

- ``hang_detector`` - fleet-wide NCCL collective hang detector. Consumes
  GPUSD-published per-rank SeqNum metadata and emits structured alerts
  when ranks stagnate in a stride pattern (the MOD-32 hang signature
  from the v6.0 submission-week 2048N debugging cluster).

See ``docs/profiling-and-perf-discovery.md`` for the v6.1 carry-forward
design appendix that drives this surface.
"""
