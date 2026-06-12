"""Aggregate per-cell normalized.json files into a single atlas.jsonl.

A campaign directory looks like::

    campaigns/<UTC>-<slug>/
      cells/
        <cell-id>/
          normalized.json     # list[dict] of AtlasCell-shaped objects
          status.txt          # one of full | partial | failed | evicted
          backend.txt         # vllm-sweep | aiperf
      atlas.jsonl             # aggregator output (this module writes it)

The aggregator:

- walks ``cells/*/normalized.json``,
- validates each row against the AtlasCell schema (raising on schema drift),
- writes one combined ``atlas.jsonl`` to the campaign root,
- returns a ``CoverageSummary`` (from ``coverage.py``) for the header block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tools.perf_tune_report.coverage import CoverageSummary, summarize
from tools.perf_tune_report.schema import AtlasCell, write_jsonl


@dataclass(frozen=True)
class AggregateResult:
    atlas_path: Path
    row_count: int
    cell_count: int
    coverage: CoverageSummary


def _iter_cell_jsons(campaign_dir: Path) -> Iterable[tuple[Path, list[dict]]]:
    cells_dir = campaign_dir / "cells"
    if not cells_dir.is_dir():
        return
    for cell_dir in sorted(cells_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        normalized = cell_dir / "normalized.json"
        if not normalized.is_file():
            continue
        try:
            data = json.loads(normalized.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"FATAL: malformed JSON in {normalized}: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(
                f"FATAL: {normalized} must contain a JSON list of AtlasCell rows; got {type(data).__name__}"
            )
        yield normalized, data


def aggregate(campaign_dir: Path) -> AggregateResult:
    """Aggregate per-cell normalized.json into ``<campaign>/atlas.jsonl``.

    Raises ``ValueError`` on any schema drift (caller is expected to surface
    the message via stderr).
    """
    campaign_dir = campaign_dir.resolve()
    if not campaign_dir.is_dir():
        raise ValueError(f"FATAL: campaign dir does not exist: {campaign_dir}")

    rows: list[AtlasCell] = []
    for path, items in _iter_cell_jsons(campaign_dir):
        for idx, item in enumerate(items):
            try:
                rows.append(AtlasCell(**item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"FATAL: row {idx} in {path} fails AtlasCell schema: {exc}"
                ) from exc

    atlas_path = campaign_dir / "atlas.jsonl"
    write_jsonl(rows, atlas_path)

    coverage = summarize(rows)
    cell_count = len({r.cell_id for r in rows})
    return AggregateResult(
        atlas_path=atlas_path,
        row_count=len(rows),
        cell_count=cell_count,
        coverage=coverage,
    )
