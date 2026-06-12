"""Shared runner helpers: cell-config validation + normalized.json IO + UTC."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import AtlasCell


@dataclass(frozen=True)
class CellConfig:
    """The minimal per-cell config required by both runners.

    Loaded from a campaign YAML's ``cells:`` list. Each cell defines the
    hardware/parallelism axes (which become AtlasCell identity fields) plus
    the concurrency sweep to run.
    """

    cell_id: str
    model: str
    hardware: str
    quant: str
    tensor_parallel: int
    parallel_strategy: str
    mtp: bool
    max_num_batched_tokens: int
    concurrencies: tuple[int, ...]

    # Optional vLLM-sweep / AIPerf-specific knobs the runner may consume.
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cell_config_from_dict(d: dict) -> CellConfig:
    return CellConfig(
        cell_id=d["cell_id"],
        model=d["model"],
        hardware=d["hardware"],
        quant=d["quant"],
        tensor_parallel=int(d["tensor_parallel"]),
        parallel_strategy=d["parallel_strategy"],
        mtp=bool(d.get("mtp", False)),
        max_num_batched_tokens=int(d["max_num_batched_tokens"]),
        concurrencies=tuple(int(c) for c in d["concurrencies"]),
        extras=dict(d.get("extras", {})),
    )


def write_normalized_json(cell_dir: Path, rows: list[AtlasCell]) -> Path:
    """Write rows to ``<cell_dir>/normalized.json`` (list-of-objects shape)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in rows]
    out = cell_dir / "normalized.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def write_status_file(cell_dir: Path, status: str) -> Path:
    cell_dir.mkdir(parents=True, exist_ok=True)
    out = cell_dir / "status.txt"
    out.write_text(status + "\n")
    return out


def write_backend_file(cell_dir: Path, backend: str) -> Path:
    cell_dir.mkdir(parents=True, exist_ok=True)
    out = cell_dir / "backend.txt"
    out.write_text(backend + "\n")
    return out
