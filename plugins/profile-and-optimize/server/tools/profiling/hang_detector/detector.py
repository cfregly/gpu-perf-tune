"""Top-level orchestrator for the fleet-wide hang detector.

``run_detector`` ties the GPUSD scraper to the stride detector and
writes append-only JSONL output for downstream operator triage.

Per ``mlperf-6.0-training/CLAUDE.md`` "Fail Fast, No Silent Fallbacks",
the orchestrator refuses to start if the output directory's parent
does not exist, and refuses to silently swallow stride-detector
errors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpusd_scraper import scrape_gpusd_snapshot
from .stride_detector import alert_to_dict, detect_stride_lag


def run_detector(
    *,
    fixture_path: Path | None = None,
    live_cluster: bool = False,
    nodelist: list[str] | None = None,
    port: int = 9420,
    stride: int = 32,
    lag_threshold: int = 1,
    output_path: Path | None = None,
    jobid: str = "unknown",
) -> dict[str, Any]:
    """Run one detection pass and return the structured result.

    Args:
        fixture_path: fixture JSON for the offline path.
        live_cluster: True to scrape live cluster endpoints.
        nodelist: hostnames (required when live_cluster=True).
        port: GPUSD metrics endpoint port.
        stride: bucket modulus passed to the stride detector.
        lag_threshold: minimum bucket-median lag to flag.
        output_path: when set, append the result as one JSONL row.
            Parent directory must exist; the orchestrator does NOT
            create it (operator artifact directories are managed
            explicitly per CLAUDE.md).
        jobid: identifier carried through to the JSONL row; useful
            for the operator's downstream join against sacct.

    Returns:
        A dict with shape:
            {
                "schema_version": 1,
                "jobid": str,
                "captured_at": ISO-8601 UTC,
                "stride": int,
                "lag_threshold": int,
                "rank_count": int,
                "alerts": [<alert dict>, ...],
            }
    """
    snapshots = scrape_gpusd_snapshot(
        fixture_path=fixture_path,
        live_cluster=live_cluster,
        nodelist=nodelist,
        port=port,
    )
    alerts = detect_stride_lag(
        snapshots, stride=stride, lag_threshold=lag_threshold
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "jobid": jobid,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stride": stride,
        "lag_threshold": lag_threshold,
        "rank_count": len(snapshots),
        "alerts": [alert_to_dict(a) for a in alerts],
    }
    if output_path is not None:
        parent = output_path.parent
        if not parent.is_dir():
            raise FileNotFoundError(
                f"output_path parent does not exist: {parent}; "
                f"create it explicitly (orchestrator does not auto-mkdir)."
            )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    return result
