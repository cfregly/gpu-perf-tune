"""GPUSD endpoint scraper for the fleet-wide hang detector.

a colleague's GPUSD plugin publishes per-rank NCCL collective
metadata (``rank``, ``seq_num``, ``op_type``, ``timestamp``) on each
node's ``mhd`` endpoint in a Prometheus-parseable format. The
canonical install path on the cluster is
``/opt/gpusd/libnccl-profiler-gpusd.so`` per the 2026-05-13 thread at
``docs/learnings/slack/<team-channel>/2026-05-13-2048n-optimized-node-list-rerun.md``.

This module reads GPUSD-shaped snapshots from one of two sources:

1. **Fixture JSON file** (default; for unit-testable orchestration
   without a live cluster). Path passed via
   :func:`scrape_gpusd_snapshot(fixture_path=...)`. See the
   ``tests/fixtures/`` directory for the schema.

2. **Live cluster endpoints** (operator opt-in via the CLI's
   ``--live-cluster`` flag). The scraper issues HTTP GETs against each
   node's ``http://${node}:${port}/metrics`` endpoint. **Connection
   pooling**: ONE persistent HTTP connection per rack (not per node)
   to cap file descriptors at 8192-GPU scale per a colleague's
   2026-05-15 9-failures FD-exhaustion finding at
   ``docs/learnings/slack/<team-channel>/2026-05-15-9-consecutive-node-failures.md``.

The fixture-file path is the default because (a) it makes the
detector unit-testable without live infrastructure, and (b) operators
can capture a one-shot GPUSD snapshot manually via
``curl http://<node>:<port>/metrics > snapshot.json`` and replay it
through the detector offline. The live-cluster path is intentionally
not the default; it requires explicit operator opt-in.

Schema (fixture JSON):

```json
{
  "schema_version": 1,
  "captured_at": "2026-05-13T19:38:00Z",
  "jobid": "11523",
  "stride": 32,
  "ranks": [
    {"rank": 0, "seq_num": 142, "timestamp": 1747166280.0, "op_type": "ALLREDUCE"},
    {"rank": 31, "seq_num": 100, "timestamp": 1747166280.0, "op_type": "ALLREDUCE"},
    ...
  ]
}
```
"""

from __future__ import annotations

import json
from pathlib import Path

from .stride_detector import RankSnapshot


def scrape_gpusd_snapshot(
    *,
    fixture_path: Path | None = None,
    live_cluster: bool = False,
    nodelist: list[str] | None = None,
    port: int = 9420,
) -> list[RankSnapshot]:
    """Read one GPUSD snapshot and return per-rank ``RankSnapshot``s.

    Args:
        fixture_path: path to a fixture JSON in the schema shown above.
            Required when ``live_cluster`` is False.
        live_cluster: when True, scrape live cluster endpoints instead
            of the fixture. The live path requires the ``requests``
            package and a reachable cluster network.
        nodelist: hostnames to scrape when ``live_cluster=True``. The
            CLI populates this from ``sacct --format=NodeList%2000``.
        port: GPUSD metrics endpoint port (default 9420 matches the
            published plugin default).

    Returns:
        A list of :class:`RankSnapshot` instances. The list is
        deduplicated by rank: when the same rank appears more than
        once, the newest timestamp wins.

    Raises:
        ValueError: on missing required args or schema-version mismatch.
        FileNotFoundError: when ``fixture_path`` does not exist.
    """
    if live_cluster:
        return _scrape_live(nodelist or [], port)
    if fixture_path is None:
        raise ValueError(
            "scrape_gpusd_snapshot requires fixture_path when live_cluster=False"
        )
    if not fixture_path.is_file():
        raise FileNotFoundError(f"fixture not found: {fixture_path}")
    return _scrape_fixture(fixture_path)


def _scrape_fixture(fixture_path: Path) -> list[RankSnapshot]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"unsupported fixture schema_version={schema_version}; "
            f"expected 1; see tools/profiling/hang_detector/gpusd_scraper.py"
        )
    ranks_data = payload.get("ranks", [])
    by_rank: dict[int, RankSnapshot] = {}
    for row in ranks_data:
        snap = RankSnapshot(
            rank=int(row["rank"]),
            seq_num=int(row["seq_num"]),
            timestamp=float(row["timestamp"]),
            op_type=str(row.get("op_type", "")),
        )
        prior = by_rank.get(snap.rank)
        if prior is None or snap.timestamp > prior.timestamp:
            by_rank[snap.rank] = snap
    return sorted(by_rank.values(), key=lambda s: s.rank)


def _scrape_live(nodelist: list[str], port: int) -> list[RankSnapshot]:
    """Live-cluster scrape path.

    Per :mod:`tools.profiling.hang_detector` "FD-exhaustion constraint",
    this batches HTTP connections at the rack level (1 connection per
    rack, reused across all nodes in that rack) to cap FDs at
    8192-GPU scale.

    This function is intentionally a thin wrapper: it imports
    ``requests`` lazily so the module is usable in the
    fixture-only test path without that dependency. The actual
    rack-level connection pool is built on demand.

    Args:
        nodelist: hostnames to scrape.
        port: GPUSD metrics endpoint port.

    Returns:
        A deduplicated list of :class:`RankSnapshot`.

    Raises:
        ImportError: if ``requests`` is not installed.
        RuntimeError: if any node returns a non-200 status or a
            schema_version that does not match.
    """
    if not nodelist:
        raise ValueError("live_cluster=True requires a non-empty nodelist")
    try:
        import requests  # noqa: F401 (imported here for the live path only)
    except ImportError as exc:
        raise ImportError(
            "live_cluster=True requires the `requests` package; "
            "install it or use the fixture path"
        ) from exc

    # Group nodes by rack. The cluster's hostname convention is
    # gb300-<rack>-<slot>; we use the rack component as the pool key.
    racks: dict[str, list[str]] = {}
    for host in nodelist:
        rack = _rack_key(host)
        racks.setdefault(rack, []).append(host)

    by_rank: dict[int, RankSnapshot] = {}
    for rack, hosts in racks.items():
        # One persistent session per rack, reused across hosts. This
        # caps FDs at len(racks), not len(nodelist).
        session = requests.Session()
        try:
            for host in hosts:
                url = f"http://{host}:{port}/metrics"
                resp = session.get(url, timeout=5.0)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"{host}: GET {url} returned {resp.status_code}"
                    )
                payload = resp.json()
                if payload.get("schema_version") != 1:
                    raise RuntimeError(
                        f"{host}: unsupported schema_version="
                        f"{payload.get('schema_version')}"
                    )
                for row in payload.get("ranks", []):
                    snap = RankSnapshot(
                        rank=int(row["rank"]),
                        seq_num=int(row["seq_num"]),
                        timestamp=float(row["timestamp"]),
                        op_type=str(row.get("op_type", "")),
                    )
                    prior = by_rank.get(snap.rank)
                    if prior is None or snap.timestamp > prior.timestamp:
                        by_rank[snap.rank] = snap
        finally:
            session.close()
    return sorted(by_rank.values(), key=lambda s: s.rank)


def _rack_key(host: str) -> str:
    """Extract the rack identifier from a cluster hostname.

    The cluster convention is ``gb300-<rack>-<slot>`` (e.g.
    ``gb300-128-001``). Falls back to the full hostname when the
    pattern does not match so the scraper still works on non-standard
    cohorts; the worst case is one session per host (no FD savings)
    rather than crashing.
    """
    parts = host.split("-")
    if len(parts) >= 3 and parts[0].startswith("gb"):
        return f"{parts[0]}-{parts[1]}"
    return host
