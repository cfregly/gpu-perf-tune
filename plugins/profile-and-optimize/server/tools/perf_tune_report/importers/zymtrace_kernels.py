"""Zymtrace per-kernel breakdown importer for ``inference-perf-bench`` bundles.

Reads the 5 TSVs that ``perf-tune-glm51/03f-variant-runner.sh`` (and any
sibling capture script) emits under ``<bundle>/zymtrace/`` and normalizes
them into a single ``<cell-dir>/kernels.json`` for the renderer to pick up.

Declared-coverage contract
--------------------------

The capture script writes ``<bundle>/capture_sources.json`` as its binding
declaration that this bundle DID capture zymtrace data. The importer's
contract:

- Manifest absent OR ``"zymtrace"`` not in ``captured_sources`` -> return
  ``None`` (skip silently). The bundle never claimed coverage; this is the
  correct no-op for bundles produced by non-zymtrace workflows (other
  clusters, MLPerf training campaigns, future sources).
- Manifest declares zymtrace -> the 5 TSVs are REQUIRED. Any missing /
  empty / malformed file raises a loud exception. This is the silent-
  degradation pattern Phase 5 SOURCE.md fell into; loud failure is the
  only way to prevent it from happening again.

Ingest-lag note
---------------

An empty / header-only TSV is often ClickHouse INGEST LAG at capture time, not
true absence: zymtrace flushes to ClickHouse asynchronously, so the capture
query can have run before the frames landed. This importer reads a STATIC TSV
snapshot and cannot requery, so it stays loud (fail-fast) -- but the fix is to
RE-CAPTURE after the flush (``capture-sol-window.sh`` now polls + requeries via
``zymtrace-ingest-wait.sh``), NOT to weaken this check. See
``server/docs/zymtrace-query-hygiene.md``.

This deliberately splits "tolerance" from "bug detection". Generic
perf-report tool stays usable on non-zymtrace clusters; the bug-detection
contract stays loud for any bundle that ever declared zymtrace coverage.

Schema
------

The emitted ``kernels.json`` shape::

    {
        "schema_version": 1,
        "captured_sources": ["zymtrace"],
        "top_kernels": [
            {"name": "multimem_all_reduce_kernel...", "samples": 558,
             "category": "NCCL"}
        ],
        "per_gpu": [
            {"gpu_name": "NVIDIA B200", "gpu_uuid": "<uuid>", "samples": 35961}
        ],
        "per_category": {"NCCL": 16510, "MoE": 37496, "FMHA": 9226, ...},
        "top_python_during_cuda": [
            {"frame": "vllm.engine.async_llm_engine...", "samples": 12345}
        ]
    }

The renderer's ``kernel_breakdown.py`` module consumes this file directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Expected TSV filenames. Order matches the 5 SQL queries in
# perf-tune-glm51/03f-variant-runner.sh phase 5.
_TSV_FILES = [
    "kernel-class.tsv",
    "top-gpu-frames.tsv",
    "per-gpu.tsv",
    "per-category.tsv",
    "top-python-during-cuda.tsv",
]

_MANIFEST = "capture_sources.json"
_KERNELS_OUTPUT = "kernels.json"

# Appended to the message of an empty / header-only TSV failure: that shape is
# usually ClickHouse INGEST LAG at capture time (zymtrace flushes asynchronously),
# not true absence. The importer reads a static snapshot and cannot requery, so it
# stays loud -- the fix is to RE-CAPTURE after the flush, not to weaken this check.
_INGEST_LAG_HINT = (
    "An empty / header-only zymtrace TSV is often ClickHouse INGEST LAG at capture "
    "time (zymtrace flushes asynchronously), not absence. This importer reads a "
    "static TSV and cannot requery -- RE-CAPTURE after the flush "
    "(capture-sol-window.sh polls + requeries via zymtrace-ingest-wait.sh); do NOT "
    "weaken this loud check. See server/docs/zymtrace-query-hygiene.md."
)


def _is_ingest_lag_shape(reason: str) -> bool:
    """True for the empty/header-only/no-rows shapes that ingest lag produces."""
    return reason == "empty" or reason.startswith("header-only") or "no data rows" in reason


class ZymtraceTSVMissing(Exception):
    """Raised when a declared zymtrace bundle is missing one of the 5 TSVs.

    The capture_sources.json manifest declared ``"zymtrace"`` so all 5 TSVs
    are required. Either a file is absent or its content is 0 bytes. The
    bundle is invalid; do not pretend it has data. A 0-byte (``reason=empty``)
    file is often ClickHouse ingest lag at capture time, not absence -- the
    message says so and points at the recapture path.
    """

    def __init__(self, path: Path, reason: str = "absent"):
        msg = f"zymtrace TSV missing: {path} ({reason})"
        if _is_ingest_lag_shape(reason):
            msg += f". {_INGEST_LAG_HINT}"
        super().__init__(msg)
        self.path = path
        self.reason = reason


class ZymtraceTSVMalformed(Exception):
    """Raised when a declared zymtrace TSV is present but unparseable.

    Distinct from ZymtraceTSVMissing because empty-file vs corrupt-file are
    different bugs (capture-broken vs query-returned-no-rows-but-headers). The
    header-only / no-data-rows shape is usually ClickHouse ingest lag at
    capture time, not a corrupt file -- the message names that + the recapture
    path (see ``server/docs/zymtrace-query-hygiene.md``).
    """

    def __init__(self, path: Path, reason: str):
        msg = f"zymtrace TSV malformed: {path} ({reason})"
        if _is_ingest_lag_shape(reason):
            msg += f". {_INGEST_LAG_HINT}"
        super().__init__(msg)
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class KernelImportResult:
    """Returned by ``import_zymtrace_kernels`` on a successful (or skipped) run."""

    bundle: Path
    kernels_json_path: Path | None  # None when manifest doesn't declare zymtrace
    skipped_reason: str | None      # set when kernels_json_path is None
    top_kernel_count: int
    category_count: int
    gpu_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": str(self.bundle),
            "kernels_json_path": str(self.kernels_json_path) if self.kernels_json_path else None,
            "skipped_reason": self.skipped_reason,
            "top_kernel_count": self.top_kernel_count,
            "category_count": self.category_count,
            "gpu_count": self.gpu_count,
        }


def _read_manifest(bundle: Path) -> dict[str, Any] | None:
    """Read capture_sources.json. Return None if absent."""
    p = bundle / _MANIFEST
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ZymtraceTSVMalformed(p, f"capture_sources.json is not valid JSON: {e}") from e


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSVWithNames file into a list of {column: value} dicts.

    Raises ZymtraceTSVMissing if the file is absent or 0 bytes.
    Raises ZymtraceTSVMalformed if header-only, wrong column count, or
    other parse-time issues. Does NOT enforce specific column names here;
    the schema-aware readers below do.
    """
    if not path.is_file():
        raise ZymtraceTSVMissing(path, reason="absent")
    if path.stat().st_size == 0:
        raise ZymtraceTSVMissing(path, reason="empty")
    lines = path.read_text().splitlines()
    if len(lines) == 0:
        raise ZymtraceTSVMissing(path, reason="empty")
    if len(lines) == 1:
        raise ZymtraceTSVMalformed(path, reason="header-only (no data rows)")
    header = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for ln_no, ln in enumerate(lines[1:], start=2):
        if ln == "":
            continue
        cells = ln.split("\t")
        if len(cells) != len(header):
            raise ZymtraceTSVMalformed(
                path,
                reason=f"line {ln_no}: column count {len(cells)} != header {len(header)}",
            )
        rows.append(dict(zip(header, cells)))
    if not rows:
        raise ZymtraceTSVMalformed(path, reason="no data rows after header")
    return rows


# CSV-friendly regex categorizer for top_kernels emission. Mirror of the
# multiIf bucketing in perf-tune-glm51/03f-variant-runner.sh query 5d so
# that JSON consumers can re-derive a kernel's category from its name
# without re-querying the per-category TSV.
import re as _re

_CATEGORY_RULES: list[tuple[_re.Pattern[str], str]] = [
    (_re.compile(r"(?i)(multimem|allreduce|flashinfer.*allreduce|nccl)"), "NCCL"),
    (_re.compile(r"(?i)(routingIndices|finalizeKernel|moe)"), "MoE"),
    (_re.compile(r"(?i)fmha.*Sm[0-9]+"), "FMHA"),
    (_re.compile(r"(?i)bmm_(E2m1|Bfloat16_E2m1)"), "BMM-NVFP4"),
    (_re.compile(r"(?i)^triton_"), "Triton-fused"),
    (_re.compile(r"(?i)(cublas|nvjet|splitKreduce)"), "cuBLAS"),
    (_re.compile(r"(?i)(FillFunctor|copy_kernel|elementwise|distribution_)"), "Elementwise"),
]


def _categorize(kernel_name: str) -> str:
    for rx, cat in _CATEGORY_RULES:
        if rx.search(kernel_name):
            return cat
    return "Other"


def _build_top_kernels(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Map 5b TSV rows into the kernels.json top_kernels array."""
    out = []
    for r in rows:
        try:
            samples = int(r["samples"])
        except (KeyError, ValueError) as e:
            raise ZymtraceTSVMalformed(
                Path("<top-gpu-frames.tsv>"), reason=f"bad samples cell in row {r}: {e}"
            ) from e
        if "kernel" not in r:
            raise ZymtraceTSVMalformed(
                Path("<top-gpu-frames.tsv>"), reason=f"missing kernel column in row {r}"
            )
        # An empty value is a legitimate UNSYMBOLIZED frame (profiling data, not
        # malformed) -- keep it as <unresolved> rather than crashing the import.
        name = r.get("kernel") or "<unresolved>"
        out.append({"name": name, "samples": samples, "category": _categorize(name)})
    return out


def _build_per_gpu(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        try:
            samples = int(r["samples"])
        except (KeyError, ValueError) as e:
            raise ZymtraceTSVMalformed(
                Path("<per-gpu.tsv>"), reason=f"bad samples cell in row {r}: {e}"
            ) from e
        out.append({
            "gpu_name": r.get("gpu_name", ""),
            "gpu_uuid": r.get("gpu_uuid", ""),
            "samples": samples,
        })
    return out


def _build_per_category(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        cat = r.get("category", "")
        if not cat:
            raise ZymtraceTSVMalformed(
                Path("<per-category.tsv>"), reason=f"missing category column in row {r}"
            )
        try:
            samples = int(r["samples"])
        except (KeyError, ValueError) as e:
            raise ZymtraceTSVMalformed(
                Path("<per-category.tsv>"), reason=f"bad samples cell in row {r}: {e}"
            ) from e
        out[cat] = samples
    return out


def _build_top_python(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        try:
            samples = int(r["samples"])
        except (KeyError, ValueError) as e:
            raise ZymtraceTSVMalformed(
                Path("<top-python-during-cuda.tsv>"),
                reason=f"bad samples cell in row {r}: {e}",
            ) from e
        if "python_frame" not in r:
            raise ZymtraceTSVMalformed(
                Path("<top-python-during-cuda.tsv>"),
                reason=f"missing python_frame column in row {r}",
            )
        # Empty value = legitimate UNSYMBOLIZED host frame (not malformed); keep it.
        frame = r.get("python_frame") or "<unresolved>"
        out.append({"frame": frame, "samples": samples})
    return out


def import_zymtrace_kernels(
    bundle: Path,
    cell_dir: Path,
    *,
    dry_run: bool = False,
) -> KernelImportResult:
    """Import zymtrace per-kernel data from a bundle into a cell directory.

    Args:
        bundle: ``*-deploy/experiments/artifacts/inference-perf-bench/<bundle>/``
            path. Will be inspected for ``capture_sources.json`` + ``zymtrace/``.
        cell_dir: target ``<campaign>/cells/<cell_id>/`` directory where
            ``kernels.json`` will be written.
        dry_run: if True, parse + validate but do NOT write kernels.json.

    Returns:
        ``KernelImportResult``. ``kernels_json_path`` is ``None`` if the
        manifest did not declare zymtrace (bundle is correctly a no-op
        for the kernel breakdown page).

    Raises:
        ZymtraceTSVMissing: manifest declared zymtrace but a required TSV
            is absent or 0 bytes.
        ZymtraceTSVMalformed: manifest declared zymtrace but a TSV is
            present yet unparseable.
    """
    bundle = bundle.expanduser().resolve()
    cell_dir = cell_dir.expanduser().resolve()

    manifest = _read_manifest(bundle)
    if manifest is None or "zymtrace" not in (manifest.get("captured_sources") or []):
        # Bundle didn't claim zymtrace coverage. Correct no-op.
        return KernelImportResult(
            bundle=bundle,
            kernels_json_path=None,
            skipped_reason="capture_sources.json absent or does not declare zymtrace",
            top_kernel_count=0,
            category_count=0,
            gpu_count=0,
        )

    # Manifest declares zymtrace -> all 5 TSVs are mandatory.
    zym_dir = bundle / "zymtrace"
    tsv_paths = {name: zym_dir / name for name in _TSV_FILES}

    # Parse each TSV. Loud failure on any miss is the contract.
    _ = _read_tsv(tsv_paths["kernel-class.tsv"])  # kept for forward compat; not in kernels.json today
    top_gpu = _build_top_kernels(_read_tsv(tsv_paths["top-gpu-frames.tsv"]))
    per_gpu = _build_per_gpu(_read_tsv(tsv_paths["per-gpu.tsv"]))
    per_category = _build_per_category(_read_tsv(tsv_paths["per-category.tsv"]))
    top_python = _build_top_python(_read_tsv(tsv_paths["top-python-during-cuda.tsv"]))

    kernels_payload = {
        "schema_version": 1,
        "captured_sources": manifest.get("captured_sources", []),
        "top_kernels": top_gpu,
        "per_gpu": per_gpu,
        "per_category": per_category,
        "top_python_during_cuda": top_python,
    }

    out_path = cell_dir / _KERNELS_OUTPUT
    if not dry_run:
        cell_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(kernels_payload, indent=2, sort_keys=True))

    return KernelImportResult(
        bundle=bundle,
        kernels_json_path=out_path,
        skipped_reason=None,
        top_kernel_count=len(top_gpu),
        category_count=len(per_category),
        gpu_count=len(per_gpu),
    )
