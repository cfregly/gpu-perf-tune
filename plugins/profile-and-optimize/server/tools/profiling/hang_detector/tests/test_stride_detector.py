"""Unit tests for the MOD-N stride-lag detector.

Per the design appendix at docs/profiling-and-perf-discovery.md
"Piece (b): Fleet-wide hang detector hooked into GPUSD-style signals",
the canonical detection target is the MOD-32 hang signature where
every 32nd rank stagnates while the other ranks advance. The
:func:`detect_stride_lag` algorithm is tested against the synthetic
fixture under ``tests/fixtures/gpusd-snapshot-2048n-mod32-hang.json``.
"""

from __future__ import annotations

import unittest

from tools.profiling.hang_detector.stride_detector import (
    RankSnapshot,
    detect_stride_lag,
)


class StrideDetectorTest(unittest.TestCase):
    def test_healthy_cohort_emits_zero_alerts(self) -> None:
        """All ranks advance synchronously; no bucket lags."""
        snaps = [
            RankSnapshot(rank=r, seq_num=150, timestamp=1.0)
            for r in (0, 1, 16, 31, 32, 33, 48, 63, 64, 95, 127)
        ]
        alerts = detect_stride_lag(snaps, stride=32)
        self.assertEqual(alerts, [])

    def test_mod32_hang_pattern_is_detected(self) -> None:
        """Ranks 31, 63, 95, 127 lag at seq_num=100; others at 150.

        The lagging bucket is `rank % 32 == 31` (containing 31, 63, 95,
        127). The leader bucket is any other (they all hold seq_num=150).
        Expect exactly one alert with the right ranks and lag=50.
        """
        snaps = [
            # Healthy ranks: spread across multiple buckets, all at 150.
            RankSnapshot(rank=0, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=1, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=16, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=30, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=32, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=33, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=48, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=62, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=64, seq_num=150, timestamp=1.0),
            # Lagging bucket: rank % 32 == 31.
            RankSnapshot(rank=31, seq_num=100, timestamp=1.0),
            RankSnapshot(rank=63, seq_num=100, timestamp=1.0),
            RankSnapshot(rank=95, seq_num=100, timestamp=1.0),
            RankSnapshot(rank=127, seq_num=100, timestamp=1.0),
        ]
        alerts = detect_stride_lag(snaps, stride=32, lag_threshold=1)
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.stride, 32)
        self.assertEqual(alert.lagging_bucket, 31)
        self.assertEqual(alert.leader_median_seq_num, 150)
        self.assertEqual(alert.lagging_median_seq_num, 100)
        self.assertEqual(alert.lag, 50)
        self.assertEqual(alert.ranks, [31, 63, 95, 127])

    def test_lag_threshold_suppresses_small_jitter(self) -> None:
        """One bucket trails by exactly 1; threshold=1 -> no alert (must
        exceed)."""
        snaps = [
            RankSnapshot(rank=0, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=32, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=1, seq_num=149, timestamp=1.0),
            RankSnapshot(rank=33, seq_num=149, timestamp=1.0),
        ]
        alerts = detect_stride_lag(snaps, stride=32, lag_threshold=1)
        # Lag of 1 is NOT > threshold=1, so no alert.
        self.assertEqual(alerts, [])

    def test_lag_threshold_zero_is_strictest(self) -> None:
        """At threshold=0, any non-zero lag is flagged."""
        snaps = [
            RankSnapshot(rank=0, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=32, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=1, seq_num=149, timestamp=1.0),
            RankSnapshot(rank=33, seq_num=149, timestamp=1.0),
        ]
        alerts = detect_stride_lag(snaps, stride=32, lag_threshold=0)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].lag, 1)
        self.assertEqual(alerts[0].lagging_bucket, 1)

    def test_single_bucket_returns_no_alerts(self) -> None:
        """Fewer than 2 buckets -> nothing to compare."""
        snaps = [
            RankSnapshot(rank=0, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=32, seq_num=150, timestamp=1.0),
            RankSnapshot(rank=64, seq_num=150, timestamp=1.0),
        ]
        # All in bucket 0 at stride=32.
        alerts = detect_stride_lag(snaps, stride=32)
        self.assertEqual(alerts, [])

    def test_invalid_stride_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_stride_lag(
                [RankSnapshot(rank=0, seq_num=1, timestamp=1.0)], stride=0
            )

    def test_invalid_lag_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_stride_lag(
                [RankSnapshot(rank=0, seq_num=1, timestamp=1.0)],
                lag_threshold=-1,
            )

    def test_empty_snapshots_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_stride_lag([], stride=32)


if __name__ == "__main__":
    unittest.main()
