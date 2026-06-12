"""Fleet-wide NCCL collective hang detector.

This module implements v6.1 carry-forward item 3 Piece (b) from
``docs/v6.1-carryforward.md`` and the design appendix at
``docs/profiling-and-perf-discovery.md`` section
``## v6.1 carry-forward: fleet-wide profiling and hang-detection
stack``.

Public surface:

- :func:`detect_stride_lag` - given a snapshot of per-rank
  ``(rank, seq_num, timestamp)`` rows, bucket ranks by
  ``rank % stride`` and return any bucket that lags the others by more
  than the configured fraction. This is the MOD-32 stride detector.
- :func:`scrape_gpusd_snapshot` - read GPUSD metrics from a local
  fixture JSON (default) or live cluster endpoints (operator opt-in).
- :func:`run_detector` - top-level orchestrator that ties the scraper
  to the stride detector and writes an append-only JSONL timeline.

The CLI is at :mod:`tools.profiling.hang_detector.cli`. Unit tests
live under :mod:`tools.profiling.hang_detector.tests`.
"""

from .detector import run_detector
from .gpusd_scraper import scrape_gpusd_snapshot
from .stride_detector import detect_stride_lag

__all__ = ["detect_stride_lag", "run_detector", "scrape_gpusd_snapshot"]
