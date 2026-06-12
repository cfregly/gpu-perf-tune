"""nsys per-kernel breakdown importer (Nsight-Systems ``cuda_gpu_kern_sum``).

Reads an nsys ``cuda_gpu_kern_sum`` report (text, from ``nsys stats --report
cuda_gpu_kern_sum <rep>``) and normalizes it into the SAME ``kernels.json``
schema that :mod:`zymtrace_kernels` emits, so the renderer's kernel-breakdown
(page 3) + zymtrace-x-DCGM cross-attribution (page 6b) pick it up when a
zymtrace flamegraph is unavailable (e.g. the per-process GPU implant didn't
intercept). The per-kernel WEIGHT is **Total GPU time (ns)** (nsys is a true
GPU-time measurement, not a sample proxy), used in the ``samples`` field so the
downstream share math is identical.

Generalizes the one-off
``perf-tune-glm51/experiments/artifacts/gqa-nvfp4kv/qwen3a3b-gqanvfp4kv-20260601T010321Z/commands/nsys_to_kernels_json.py``
(2026-06-01 GQA NVFP4-KV per-kernel finding) into a first-class verb.

Declared-coverage contract (mirrors zymtrace_kernels / ncu_kernels):

- ``capture_sources.json`` absent OR ``"nsys"`` not in ``captured_sources`` ->
  return a result with ``kernels_json_path=None`` + a ``skipped_reason``
  (silent no-op; the bundle never claimed nsys coverage).
- Manifest declares ``"nsys"`` -> ``nsys/cuda_gpu_kern_sum.txt`` (overridable)
  is REQUIRED; missing / empty / no-parsable-rows raises a loud exception.

Emitted ``kernels.json`` shape (same as zymtrace)::

    {"schema_version": 1, "captured_sources": ["nsys"],
     "top_kernels": [{"name", "samples", "category"}],
     "per_gpu": [{"gpu_name", "gpu_uuid", "samples"}],
     "per_category": {category: samples},
     "top_python_during_cuda": []}   # nsys --sample=none -> no host python sampling
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MANIFEST = "capture_sources.json"
_KERNELS_OUTPUT = "kernels.json"
_DEFAULT_KERN_SUM = "cuda_gpu_kern_sum.txt"

# Reuse zymtrace's category rules + a BF16-MoE-bmm rule (the Qwen3 MoE expert
# GEMMs are ``bmm_Bfloat16_Bfloat16Bfloat16...``, which the NVFP4-only bmm rule
# misses). Kept local (not imported) so a change here can't perturb the zymtrace
# importer's behavior.
_CATEGORY_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(multimem|allreduce|flashinfer.*allreduce|nccl)"), "NCCL"),
    (re.compile(r"(?i)bmm_(E2m1|Bfloat16_E2m1)"), "BMM-NVFP4"),
    (re.compile(r"(?i)^bmm_Bfloat16"), "MoE"),
    (re.compile(r"(?i)(routingindices|finalizekernel|::moe::|moe_forward|moe::dev)"), "MoE"),
    (re.compile(r"(?i)fmha.*Sm[0-9]+"), "FMHA"),
    (re.compile(r"(?i)^triton_"), "Triton-fused"),
    (re.compile(r"(?i)(cublas|nvjet|splitkreduce)"), "cuBLAS"),
    (re.compile(r"(?i)(fillfunctor|copy_kernel|elementwise|distribution_)"), "Elementwise"),
]

# A cuda_gpu_kern_sum data row: "<time%>  <total_ns>  <instances>  <avg> <med> <min> <max> <std>  <Name>"
_ROW = re.compile(r"^\s*\d+\.\d+\s+(\d+)\s+\d+\s+.*\s+(\S+)\s*$")


class NsysKernSumMissing(Exception):
    """Declared nsys bundle is missing / empty the cuda_gpu_kern_sum report."""


class NsysKernSumMalformed(Exception):
    """cuda_gpu_kern_sum present but no parseable kernel rows."""


# Gate-0 hint: an empty/no-row kern_sum on GB300 <org-id> is usually the CUDA
# image-vs-driver CUPTI skew (12.x toolkit vs 13.x driver -> CUPTI_ERROR_INVALID_DEVICE),
# not capture hygiene. See nsys-capture-hygiene rule "Gate 0".
_CUPTI_SKEW_HINT = (
    " -- if 0 kernels on GB300 <org-id>, this is likely the CUDA 12.x-image vs 13.x-driver"
    " CUPTI skew (CUPTI_ERROR_INVALID_DEVICE), NOT capture hygiene; grep the nsys log for"
    " 'CUDA versions. CUPTI/Runtime/Driver' and use a CUDA-13-aligned image or zymtrace"
)


@dataclass(frozen=True)
class NsysKernelImportResult:
    bundle: Path
    kernels_json_path: Path | None
    skipped_reason: str | None
    top_kernel_count: int
    category_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": str(self.bundle),
            "kernels_json_path": str(self.kernels_json_path) if self.kernels_json_path else None,
            "skipped_reason": self.skipped_reason,
            "top_kernel_count": self.top_kernel_count,
            "category_count": self.category_count,
        }


def _categorize(name: str) -> str:
    for rx, cat in _CATEGORY_RULES:
        if rx.search(name):
            return cat
    return "Other"


def _read_manifest(bundle: Path) -> dict[str, Any] | None:
    p = bundle / _MANIFEST
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _parse_kern_sum(path: Path) -> list[tuple[int, str]]:
    if not path.is_file():
        raise NsysKernSumMissing(f"{path} absent")
    if path.stat().st_size == 0:
        raise NsysKernSumMissing(f"{path} empty" + _CUPTI_SKEW_HINT)
    rows: list[tuple[int, str]] = []
    for ln in path.read_text().splitlines():
        m = _ROW.match(ln)
        if not m:
            continue
        rows.append((int(m.group(1)), m.group(2).rstrip("\u2026.")))
    if not rows:
        raise NsysKernSumMalformed(f"{path}: no parseable cuda_gpu_kern_sum rows" + _CUPTI_SKEW_HINT)
    return rows


def import_nsys_kernels(
    bundle: Path,
    cell_dir: Path,
    *,
    kern_sum_name: str = _DEFAULT_KERN_SUM,
    gpu_name: str = "NVIDIA B200",
    dry_run: bool = False,
) -> NsysKernelImportResult:
    """Import an nsys cuda_gpu_kern_sum into a campaign cell's kernels.json.

    Returns a result with ``kernels_json_path=None`` (skip) if the bundle's
    capture_sources.json does not declare ``"nsys"``.
    """
    bundle = bundle.expanduser().resolve()
    cell_dir = cell_dir.expanduser().resolve()

    manifest = _read_manifest(bundle)
    if manifest is None or "nsys" not in (manifest.get("captured_sources") or []):
        return NsysKernelImportResult(
            bundle=bundle,
            kernels_json_path=None,
            skipped_reason="capture_sources.json absent or does not declare nsys",
            top_kernel_count=0,
            category_count=0,
        )

    rows = _parse_kern_sum(bundle / "nsys" / kern_sum_name)
    top_kernels: list[dict[str, Any]] = []
    per_category: dict[str, int] = {}
    total = 0
    for total_ns, name in rows:
        cat = _categorize(name)
        top_kernels.append({"name": name, "samples": total_ns, "category": cat})
        per_category[cat] = per_category.get(cat, 0) + total_ns
        total += total_ns

    payload = {
        "schema_version": 1,
        "captured_sources": manifest.get("captured_sources", []),
        "top_kernels": top_kernels,
        "per_gpu": [{"gpu_name": gpu_name, "gpu_uuid": "nsys-aggregate", "samples": total}],
        "per_category": per_category,
        "top_python_during_cuda": [],
    }

    out_path = cell_dir / _KERNELS_OUTPUT
    if not dry_run:
        cell_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return NsysKernelImportResult(
        bundle=bundle,
        kernels_json_path=out_path,
        skipped_reason=None,
        top_kernel_count=len(top_kernels),
        category_count=len(per_category),
    )
