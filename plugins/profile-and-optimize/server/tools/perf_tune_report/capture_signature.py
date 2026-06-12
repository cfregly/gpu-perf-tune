"""Exact-variant capture signatures and reuse planning for perf-report.

This module is intentionally conservative: a capture artifact can be reused only
when the source and target cells produce the same canonical serving-variant
signature. It is pure local filesystem post-processing; it never talks to a
cluster and never weakens render/publish strictness.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.perf_tune_report.schema import AtlasCell, read_jsonl

SCHEMA_VERSION = "capture_signature_v1"
PLAN_SCHEMA_VERSION = "capture_plan_v1"
REUSE_SCHEMA_VERSION = "capture_reuse_v1"

CAPTURE_ARTIFACTS = (
    "kernels.json",
    "dcgm_correlation.json",
    "roofline_sweep.json",
    "ncu_kernels.json",
)

_EXTRA_KEYS = (
    "arm",
    "attention_backend",
    "cuda_graph_max_bs",
    "dcgm_util",
    "engine",
    "env",
    "extra_args",
    "extra_env",
    "fimoe",
    "flags",
    "gpu_memory_utilization",
    "gqa_backend",
    "mem_fraction_static",
    "router",
    "sglang_mem_fraction_static",
    "topology",
)
# NOTE: ``max_num_seqs`` / ``prefix_cache`` / ``spec_decode_k`` are intentionally NOT here --
# they were promoted to first-class AtlasCell fields (max_num_seqs / enable_prefix_caching /
# num_speculative_tokens) in 2026-06-07, and ``_variant_fields`` reads those (with an ``extra``
# fallback). Keeping them in ``_EXTRA_KEYS`` too would double-count them in the signature.

_ROOFLINE_SUFFIXES = ("-decode", "-prefill")


@dataclass(frozen=True)
class CaptureSignature:
    """Canonical serving-variant signature for one perf-report cell."""

    schema_version: str
    hash: str
    fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CellCaptureState:
    """Local capture-artifact state for one rankable cell."""

    campaign_id: str
    campaign_dir: str
    cell_id: str
    artifact_cell_id: str
    signature: CaptureSignature
    artifacts: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signature"] = self.signature.to_dict()
        return d


@dataclass(frozen=True)
class ReuseCandidate:
    artifact: str
    source_campaign: str
    source_campaign_dir: str
    source_cell: str
    source_path: str
    target_campaign: str
    target_campaign_dir: str
    target_cell: str
    target_path: str
    signature_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissingCaptureGroup:
    artifact: str
    signature_hash: str
    signature_fields: dict[str, Any]
    cells: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapturePlan:
    schema_version: str
    generated_at_utc: str
    target_campaigns: list[str]
    source_campaigns: list[str]
    cells: list[CellCaptureState]
    reuse_candidates: list[ReuseCandidate]
    missing_groups: list[MissingCaptureGroup]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "target_campaigns": self.target_campaigns,
            "source_campaigns": self.source_campaigns,
            "cells": [c.to_dict() for c in self.cells],
            "reuse_candidates": [c.to_dict() for c in self.reuse_candidates],
            "missing_groups": [g.to_dict() for g in self.missing_groups],
        }


@dataclass(frozen=True)
class MaterializedReuse:
    schema_version: str
    generated_at_utc: str
    copied: list[dict[str, Any]]
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_hash(fields: dict[str, Any]) -> str:
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _normalize_value(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_normalize_value(x) for x in v]
    return v


def _pick_extra(extra: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _normalize_value(extra[k])
        for k in _EXTRA_KEYS
        if k in extra and extra[k] not in (None, "", "unknown")
    }


def _variant_fields(row: AtlasCell) -> dict[str, Any]:
    extra = row.extra or {}
    fields: dict[str, Any] = {
        "model": row.model,
        "hardware": row.hardware,
        "quant": row.quant,
        "kv_cache_dtype": row.kv_cache_dtype,
        "serving_engine": row.serving_engine or row.backend or "",
        "backend": row.backend,
        "image": row.image,
        "tensor_parallel": int(row.tensor_parallel),
        "data_parallel": int(getattr(row, "data_parallel", 1) or 1),
        "pipeline_parallel": int(getattr(row, "pipeline_parallel", 1) or 1),
        "parallel_strategy": row.parallel_strategy,
        "mtp": bool(row.mtp),
        # Serving-variant knobs: prefer the first-class AtlasCell fields (added 2026-06-07),
        # falling back to the legacy ``extra`` keys so signatures of older rows are stable.
        "num_speculative_tokens": (
            getattr(row, "num_speculative_tokens", None)
            if getattr(row, "num_speculative_tokens", None) is not None
            else extra.get("spec_decode_k")
        ),
        "async_scheduling": getattr(row, "async_scheduling", None),
        "max_num_seqs": (
            getattr(row, "max_num_seqs", None)
            if getattr(row, "max_num_seqs", None) is not None
            else extra.get("max_num_seqs")
        ),
        "enable_prefix_caching": (
            getattr(row, "enable_prefix_caching", None)
            if getattr(row, "enable_prefix_caching", None) is not None
            else extra.get("prefix_cache")
        ),
        "bench_backend": getattr(row, "bench_backend", "") or "",
        "max_num_batched_tokens": (
            int(row.max_num_batched_tokens) if row.max_num_batched_tokens is not None else None
        ),
        "cudagraph_mode": row.cudagraph_mode,
        "gpu_memory_utilization": row.gpu_memory_utilization,
        "cache_mode": row.cache_mode,
        "variant_extra": _pick_extra(extra),
    }
    return {k: _normalize_value(v) for k, v in fields.items()}


def signature_for_row(row: AtlasCell) -> CaptureSignature:
    """Return the exact serving-variant signature for an atlas row.

    Concurrency and workload are deliberately excluded: page-7/pipeline captures
    contain their own concurrency/phase grids, while this signature answers
    whether two cells are the same serving configuration. ``image`` IS included --
    a captured profile cannot be reused across serving images.
    """

    fields = _variant_fields(row)
    return CaptureSignature(
        schema_version=SCHEMA_VERSION,
        hash=_json_hash(fields),
        fields=fields,
    )


def variant_key_for(row: AtlasCell) -> str:
    """Image-INDEPENDENT serving-variant key (the lake ``atlas_v1.variant_key`` + the
    longitudinal trend grouping).

    Distinct from ``signature_for_row().hash``: that INCLUDES ``image`` (a captured profile
    can't be reused across images), but a time-series of "the same config across engine-image
    versions" must SHARE a key -- the image is the version axis you trend OVER. So this drops
    ``image`` and hashes the rest of the variant descriptor."""
    fields = dict(_variant_fields(row))
    fields.pop("image", None)
    return _json_hash(fields)


def _rankable_rows(campaign_dir: Path) -> dict[str, AtlasCell]:
    atlas = campaign_dir / "atlas.jsonl"
    if not atlas.is_file():
        return {}
    rows = read_jsonl(atlas)
    by_cell: dict[str, AtlasCell] = {}
    for row in rows:
        if row.cell_id.endswith(_ROOFLINE_SUFFIXES):
            continue
        if row.status not in ("full", "partial"):
            continue
        by_cell.setdefault(row.cell_id, row)
    return by_cell


def _artifact_cell_id(campaign_dir: Path, row: AtlasCell) -> str:
    """Return the cells/<id>/ directory that should carry artifacts for ``row``.

    Most campaigns use ``row.cell_id`` directly. A few legacy variant-A/B imports
    accidentally suffixed logical atlas rows with ``-Kengine`` while the actual
    cell directory stayed at ``extra["arm"]``; prefer that physical arm directory
    when it exists so planning does not report false misses.
    """

    arm = (row.extra or {}).get("arm")
    if isinstance(arm, str) and arm and (
        (campaign_dir / "cells" / arm).is_dir()
        or (campaign_dir / "cells" / f"{arm}-decode").is_dir()
        or (campaign_dir / "cells" / f"{arm}-prefill").is_dir()
    ):
        return arm
    return row.cell_id


def _artifact_path(campaign_dir: Path, cell_id: str, artifact: str) -> Path:
    base = campaign_dir / "cells" / cell_id / artifact
    if base.is_file():
        return base
    if artifact == "roofline_sweep.json":
        decode = campaign_dir / "cells" / f"{cell_id}-decode" / artifact
        if decode.is_file():
            return decode
    return base


def _artifact_target_path(campaign_dir: Path, cell_id: str, artifact: str) -> Path:
    return campaign_dir / "cells" / cell_id / artifact


def _cell_artifacts(campaign_dir: Path, cell_id: str) -> dict[str, bool]:
    return {artifact: _artifact_path(campaign_dir, cell_id, artifact).is_file()
            for artifact in CAPTURE_ARTIFACTS}


def collect_cells(campaign_dirs: Iterable[Path]) -> list[CellCaptureState]:
    out: list[CellCaptureState] = []
    for campaign_dir in campaign_dirs:
        campaign_dir = campaign_dir.resolve()
        for cell_id, row in _rankable_rows(campaign_dir).items():
            artifact_cell_id = _artifact_cell_id(campaign_dir, row)
            out.append(
                CellCaptureState(
                    campaign_id=campaign_dir.name,
                    campaign_dir=str(campaign_dir),
                    cell_id=cell_id,
                    artifact_cell_id=artifact_cell_id,
                    signature=signature_for_row(row),
                    artifacts=_cell_artifacts(campaign_dir, artifact_cell_id),
                )
            )
    return out


def build_plan(target_campaign_dirs: list[Path], source_campaign_dirs: list[Path]) -> CapturePlan:
    """Build a grouped exact-match capture plan for local campaigns."""

    targets = collect_cells(target_campaign_dirs)
    sources = collect_cells(source_campaign_dirs)
    source_index: dict[tuple[str, str], CellCaptureState] = {}
    for source in sources:
        for artifact, present in source.artifacts.items():
            if present:
                source_index.setdefault((artifact, source.signature.hash), source)

    candidates: list[ReuseCandidate] = []
    missing: dict[tuple[str, str], MissingCaptureGroup] = {}
    for target in targets:
        target_dir = Path(target.campaign_dir)
        for artifact, present in target.artifacts.items():
            if present:
                continue
            source = source_index.get((artifact, target.signature.hash))
            if source is not None:
                source_path = _artifact_path(Path(source.campaign_dir), source.artifact_cell_id, artifact)
                target_path = _artifact_target_path(target_dir, target.artifact_cell_id, artifact)
                candidates.append(
                    ReuseCandidate(
                        artifact=artifact,
                        source_campaign=source.campaign_id,
                        source_campaign_dir=source.campaign_dir,
                        source_cell=source.cell_id,
                        source_path=str(source_path),
                        target_campaign=target.campaign_id,
                        target_campaign_dir=target.campaign_dir,
                        target_cell=target.cell_id,
                        target_path=str(target_path),
                        signature_hash=target.signature.hash,
                    )
                )
            else:
                key = (artifact, target.signature.hash)
                group = missing.get(key)
                cell_ref = {
                    "campaign": target.campaign_id,
                    "campaign_dir": target.campaign_dir,
                    "cell_id": target.cell_id,
                    "artifact_cell_id": target.artifact_cell_id,
                }
                if group is None:
                    missing[key] = MissingCaptureGroup(
                        artifact=artifact,
                        signature_hash=target.signature.hash,
                        signature_fields=target.signature.fields,
                        cells=[cell_ref],
                    )
                else:
                    group.cells.append(cell_ref)

    return CapturePlan(
        schema_version=PLAN_SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_campaigns=[str(p.resolve()) for p in target_campaign_dirs],
        source_campaigns=[str(p.resolve()) for p in source_campaign_dirs],
        cells=targets,
        reuse_candidates=candidates,
        missing_groups=list(missing.values()),
    )


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"capture reuse plan is not valid JSON: {path}: {exc}") from exc
    if data.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"capture reuse plan schema mismatch: expected {PLAN_SCHEMA_VERSION}, "
            f"got {data.get('schema_version')!r}"
        )
    return data


def materialize_reuse(plan_path: Path, *, dry_run: bool = False) -> MaterializedReuse:
    """Copy exact-match artifacts from a capture plan into target cell dirs."""

    plan = _load_plan(plan_path.expanduser().resolve())
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in plan.get("reuse_candidates", []):
        artifact = str(item["artifact"])
        src = Path(item["source_path"]).expanduser()
        dst = Path(item["target_path"]).expanduser()
        if not src.is_file():
            skipped.append({**item, "reason": "source artifact missing"})
            continue
        if dst.is_file():
            skipped.append({**item, "reason": "target artifact already exists"})
            continue
        record = {
            **item,
            "source_sha256": _sha256_file(src),
            "dry_run": dry_run,
        }
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            record["target_sha256"] = _sha256_file(dst)
            reuse_path = dst.parent / "capture_reuse.json"
            existing: dict[str, Any]
            if reuse_path.is_file():
                try:
                    existing = json.loads(reuse_path.read_text())
                except json.JSONDecodeError:
                    existing = {}
            else:
                existing = {}
            if existing.get("schema_version") != REUSE_SCHEMA_VERSION:
                existing = {
                    "schema_version": REUSE_SCHEMA_VERSION,
                    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "artifacts": [],
                }
            existing.setdefault("artifacts", []).append(record)
            reuse_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        copied.append(record)
    return MaterializedReuse(
        schema_version=REUSE_SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        copied=copied,
        skipped=skipped,
    )
