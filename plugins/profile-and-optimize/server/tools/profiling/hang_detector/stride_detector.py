"""MOD-N stride-pattern hang detection.

The v6.0 2048N MOD-32 hang exhibited a structural signature: every
32nd rank (rank 31, 63, 95, 127, ...) stagnated while the other ranks
in each bucket advanced. The :func:`detect_stride_lag` function below
reproduces the same bucket-and-compare analysis on a snapshot of
per-rank ``(rank, seq_num, timestamp)`` data.

Algorithm:

1. Bucket ranks by ``rank % stride`` (default ``stride=32``).
2. For each bucket, compute the median ``seq_num``.
3. The "leader" bucket is the one with the highest median seq_num.
4. Any bucket whose median trails the leader's by more than
   ``lag_threshold`` seq_nums is flagged as lagging.

The default ``lag_threshold=1`` is intentionally tight: at MLPerf
training-cluster scale, healthy ranks advance synchronously and any
visible per-bucket median gap is suspicious. Operators tuning this for
specific cohorts can raise the threshold via the CLI.

Per `docs/learnings/slack/<team-channel>/2026-05-13-2048n-optimized-node-list-rerun.md`,
the canonical detection target is "rank 31, 63, 95, 127, ... not
advancing while other ranks do" - which this stride-bucket median
comparison exactly reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class RankSnapshot:
    """One per-rank reading from the GPUSD scraper.

    Fields mirror the GPUSD plugin's published metadata. ``op_type`` is
    the NCCL collective name (e.g. ``ALLREDUCE``, ``ALLGATHER``,
    ``REDUCE_SCATTER``) and is preserved here so per-collective lag
    analysis is possible in future versions; the default detector
    operates on the per-rank latest seq_num regardless of op_type.
    """

    rank: int
    seq_num: int
    timestamp: float
    op_type: str = ""


@dataclass(frozen=True)
class StrideLagAlert:
    """One flagged bucket from the stride detector.

    ``lagging_bucket`` is ``rank % stride``; ``leader_bucket`` is the
    bucket the lag is measured against. ``ranks`` is the list of
    physical ranks in the lagging bucket.
    """

    stride: int
    lagging_bucket: int
    leader_bucket: int
    leader_median_seq_num: int
    lagging_median_seq_num: int
    lag: int
    ranks: list[int]


def detect_stride_lag(
    snapshots: list[RankSnapshot],
    *,
    stride: int = 32,
    lag_threshold: int = 1,
) -> list[StrideLagAlert]:
    """Detect rank buckets that lag the rest by more than ``lag_threshold``.

    Args:
        snapshots: per-rank latest seq_num readings. If multiple
            readings exist for the same rank, the caller should
            deduplicate (the CLI orchestrator keeps the newest by
            timestamp).
        stride: bucket modulus. Default 32 matches the MOD-32 hang
            signature.
        lag_threshold: minimum median-seq_num gap to flag a bucket.
            Default 1 is tight; healthy ranks advance synchronously.

    Returns:
        A list of :class:`StrideLagAlert` instances, one per lagging
        bucket. Empty list when no bucket lags.

    Raises:
        ValueError: if ``stride < 1``, ``lag_threshold < 0``, or
            ``snapshots`` is empty.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if lag_threshold < 0:
        raise ValueError(f"lag_threshold must be >= 0, got {lag_threshold}")
    if not snapshots:
        raise ValueError("snapshots must be non-empty")

    buckets: dict[int, list[RankSnapshot]] = {}
    for snap in snapshots:
        bucket = snap.rank % stride
        buckets.setdefault(bucket, []).append(snap)

    if len(buckets) < 2:
        # Need at least two buckets to compare leader vs lagging.
        return []

    bucket_medians: dict[int, int] = {
        b: int(median(s.seq_num for s in group)) for b, group in buckets.items()
    }
    leader_bucket = max(bucket_medians, key=lambda b: bucket_medians[b])
    leader_median = bucket_medians[leader_bucket]

    alerts: list[StrideLagAlert] = []
    for bucket, lagging_median in bucket_medians.items():
        if bucket == leader_bucket:
            continue
        lag = leader_median - lagging_median
        if lag > lag_threshold:
            alerts.append(
                StrideLagAlert(
                    stride=stride,
                    lagging_bucket=bucket,
                    leader_bucket=leader_bucket,
                    leader_median_seq_num=leader_median,
                    lagging_median_seq_num=lagging_median,
                    lag=lag,
                    ranks=sorted(s.rank for s in buckets[bucket]),
                )
            )
    return alerts


def alert_to_dict(alert: StrideLagAlert) -> dict[str, Any]:
    """Serialize :class:`StrideLagAlert` to a JSON-friendly dict."""
    return {
        "stride": alert.stride,
        "lagging_bucket": alert.lagging_bucket,
        "leader_bucket": alert.leader_bucket,
        "leader_median_seq_num": alert.leader_median_seq_num,
        "lagging_median_seq_num": alert.lagging_median_seq_num,
        "lag": alert.lag,
        "ranks": list(alert.ranks),
    }
