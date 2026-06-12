"""NCU per-kernel breakdown importer for ``inference-kernel-ncu-profile`` bundles.

Reads ``<bundle>/ncu-profiles/*-sol.csv`` (SpeedOfLight section) +
``*-raw.csv`` (raw metrics with FLOPS / bytes) and normalises them into a
single ``<cell-dir>/ncu_kernels.json`` for the renderer's
``sol_roofline_scatter.py`` page to pick up.

This is the per-kernel byte-grounded counterpart to
``zymtrace_kernels.py`` (which produces the time-share-proxy
``kernels.json``). The two outputs sit side-by-side in the same cell
directory; the renderer chooses the higher-rigor page when both exist.

Declared-coverage contract
--------------------------

Same shape as ``zymtrace_kernels.py``:

- ``capture_sources.json`` absent OR ``"ncu"`` not in ``captured_sources``
  -> return a result with ``ncu_kernels_json_path=None`` and a
  ``skipped_reason``. Silently skip; the bundle never claimed coverage.
- Manifest declares ncu -> ``ncu-profiles/`` MUST contain at least one
  ``*-sol.csv`` + matching ``*-raw.csv`` pair. Any missing / empty /
  malformed file raises a loud exception.

Schema
------

The emitted ``ncu_kernels.json`` shape::

    {
        "schema_version": 1,
        "captured_sources": ["ncu"],
        "hw_key": "b200_sm100",
        "kernels": [
            {
                "name": "multimem_all_reduce_kernel",
                "category": "NCCL",
                "kernel_time_ns": 12340,
                "dram_bytes_total": 1200000000.0,
                "sm_flops_total": 230000000.0,
                "tensor_flops_total": 4100000000.0,
                "arithmetic_intensity_flops_per_byte": 0.19,
                "achieved_dram_pct_peak": 92.0,
                "achieved_sm_pct_peak": 8.5,
                "achieved_occupancy_pct": 31.2,
                "block_limit_factor": "registers",
                "achieved_tflops": 18.6
            }
        ]
    }

The renderer's ``sol_roofline_scatter.py`` consumes this file directly.
The ``hw_key`` field links each kernel to the corresponding peak in
``configs/sol-ceilings.yaml``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Shared categorisation rules with zymtrace_kernels.py. Re-imported here
# so this importer doesn't need a runtime dependency on the zymtrace one,
# but the rules MUST stay in sync (test below asserts this).
_CATEGORY_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(multimem|allreduce|flashinfer.*allreduce|nccl)"), "NCCL"),
    (re.compile(r"(?i)(routingIndices|finalizeKernel|moe)"), "MoE"),
    (re.compile(r"(?i)fmha.*Sm[0-9]+"), "FMHA"),
    (re.compile(r"(?i)bmm_(E2m1|Bfloat16_E2m1)"), "BMM-NVFP4"),
    (re.compile(r"(?i)^triton_"), "Triton-fused"),
    (re.compile(r"(?i)(cublas|nvjet|splitKreduce)"), "cuBLAS"),
    (re.compile(r"(?i)(FillFunctor|copy_kernel|elementwise|distribution_)"), "Elementwise"),
]

_MANIFEST = "capture_sources.json"
_NCU_KERNELS_OUTPUT = "ncu_kernels.json"
_PROFILES_SUBDIR = "ncu-profiles"

# ncu reports byte counters (e.g. dram__bytes.sum) with a unit that varies by
# magnitude -- "byte", "Kbyte", "Mbyte", "Gbyte", "Tbyte". In the wide
# (--page raw) CSV that unit lives in a separate units row (NOT the column
# name), so a unit-unaware sum is off by 1e3/1e6/1e9. This maps the unit to a
# byte multiplier (ncu uses decimal SI prefixes; binary Ki/Mi/Gi handled too).
_BYTE_UNIT_SCALE: dict[str, float] = {
    "byte": 1.0, "bytes": 1.0,
    "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12,
    "kib": 1024.0, "mib": 1024.0**2, "gib": 1024.0**3, "tib": 1024.0**4,
}

# Tensor-core "ops" -> FLOPs. ncu's sm__ops_path_tensor_*.sum counts each math
# op (multiply AND add counted separately), so the reported value ALREADY equals
# the FLOP count (a MAC = 2 ops = 2 FLOP) -> multiplier is 1.0, not 2.0.
# Empirically confirmed on SM100 (2026-06-01): the fp8 fmha-DSA kernel reported
# sm__ops_path_tensor_src_fp4_fp6_fp8_dst_fp32.sum = 6,845,104,128, which exactly
# matches the analytic attention FLOPs 2*(192*8*2048*576)+2*(192*8*2048*512) (the
# MAC*2 is already inside the metric). If a future metric counts MACs, set 2.0.
_TENSOR_OP_TO_FLOP = 1.0


def _byte_scale(unit: str | None) -> float:
    """Byte multiplier for an ncu byte-counter unit string (default 1.0)."""
    if not unit:
        return 1.0
    return _BYTE_UNIT_SCALE.get(unit.strip().lower(), 1.0)


# ncu reports raw time counters (gpu__time_active.sum) in a version-dependent
# unit (often "usecond") that the raw page does NOT normalise. Map it to ns
# (default 1.0 = already ns) so a --metrics-only capture (no SoL Duration) gets
# a correct FLOPS-rate denominator. The SoL-section Duration is normalised
# separately by _duration_ns; this covers the no-SoL path.
_TIME_UNIT_SCALE: dict[str, float] = {
    "ns": 1.0, "nsecond": 1.0, "nanosecond": 1.0,
    "us": 1e3, "usecond": 1e3, "microsecond": 1e3,
    "ms": 1e6, "msecond": 1e6, "millisecond": 1e6,
    "s": 1e9, "second": 1e9,
}


def _time_scale(unit: str | None) -> float:
    """ns multiplier for an ncu time-counter unit string (default 1.0 = ns)."""
    if not unit:
        return 1.0
    return _TIME_UNIT_SCALE.get(unit.strip().lower(), 1.0)


class NcuCsvMissing(Exception):
    """Raised when a declared ncu bundle has no usable -sol.csv / -raw.csv pair.

    The capture_sources.json manifest declared ``"ncu"`` so at least one
    matched pair is required. Either no files exist under
    ``ncu-profiles/`` or all files are 0 bytes.
    """

    def __init__(self, path: Path, reason: str = "absent"):
        super().__init__(f"ncu CSV missing: {path} ({reason})")
        self.path = path
        self.reason = reason


class NcuCsvMalformed(Exception):
    """Raised when a declared ncu CSV is present but unparseable.

    Distinct from ``NcuCsvMissing``: empty-file vs corrupt-file are
    different bugs (ncu-export-broken vs ncu-import-produced-bad-output).
    Same no-silent-degradation contract as ``ZymtraceTSVMalformed``.
    """

    def __init__(self, path: Path, reason: str):
        super().__init__(f"ncu CSV malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class NcuImportResult:
    """Returned by ``import_ncu_kernels`` on a successful (or skipped) run."""

    bundle: Path
    ncu_kernels_json_path: Path | None
    skipped_reason: str | None
    kernel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": str(self.bundle),
            "ncu_kernels_json_path": (
                str(self.ncu_kernels_json_path) if self.ncu_kernels_json_path else None
            ),
            "skipped_reason": self.skipped_reason,
            "kernel_count": self.kernel_count,
        }


def _categorize(kernel_name: str) -> str:
    for rx, cat in _CATEGORY_RULES:
        if rx.search(kernel_name):
            return cat
    return "Other"


def _read_manifest(bundle: Path) -> dict[str, Any] | None:
    p = bundle / _MANIFEST
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise NcuCsvMalformed(p, f"capture_sources.json is not valid JSON: {e}") from e


def _pivot_long_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pivot ncu-2026 long/melted CSV rows into wide per-kernel-instance rows.

    ncu's ``--page details`` export (e.g. ``--section SpeedOfLight``) emits one
    row PER METRIC, with columns ``Kernel Name``, ``Metric Name``,
    ``Metric Unit``, ``Metric Value`` (plus ``ID`` identifying the launch
    instance). The wide aggregator expects one row per kernel instance with
    friendly metric columns (e.g. ``DRAM Throughput [%]``). This collapses the
    long rows back into that wide shape, keying each metric column as
    ``"<Metric Name> [<unit>]"`` so the existing tolerant ``_find_column``
    lookups match. Rows with an empty Metric Name (ncu's per-section rule
    rows, e.g. ``SOLBottleneck``) are skipped.
    """
    wide: "OrderedDict[tuple[str, str], dict[str, str]]" = OrderedDict()
    for r in rows:
        name = (r.get("Kernel Name") or "").strip()
        metric = (r.get("Metric Name") or "").strip()
        if not name or not metric:
            continue
        inst = (r.get("ID") or "").strip()
        key = (inst, name)
        bucket = wide.get(key)
        if bucket is None:
            bucket = {"ID": inst, "Kernel Name": name}
            wide[key] = bucket
        unit = (r.get("Metric Unit") or "").strip()
        col = f"{metric} [{unit}]" if unit else metric
        bucket[col] = (r.get("Metric Value") or "").strip()
    return list(wide.values())


def _read_ncu_csv(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Read an ncu --csv output file into ``(row dicts, units-by-column)``.

    Two ncu CSV shapes are handled:

    - **Wide** (``--page raw``): a header row with metric names as columns,
      THEN a units row (Kernel Name empty, each cell a unit string like
      ``Mbyte`` / ``inst`` / ``%``), THEN one row per kernel instance. The
      units row is extracted into the returned ``units`` map (keyed by column
      name) before being filtered out, so callers can scale byte counters.
    - **Long / melted** (``--page details``, e.g. ``--section SpeedOfLight``
      on ncu 2026): one row PER METRIC with ``Metric Name`` / ``Metric Value``
      columns. These are pivoted back to the wide shape via
      ``_pivot_long_rows`` (which already folds the unit into the column name),
      so the returned ``units`` map is empty for this shape.

    Some ncu versions emit a metadata preamble; we skip lines until the
    header row (identified by the ``Kernel Name`` column).

    Raises ``NcuCsvMissing`` if absent or 0 bytes. Raises
    ``NcuCsvMalformed`` if no header row is found or rows have a column
    count mismatch.
    """
    if not path.is_file():
        raise NcuCsvMissing(path, reason="absent")
    if path.stat().st_size == 0:
        raise NcuCsvMissing(path, reason="empty")
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise NcuCsvMissing(path, reason="empty")

    # ncu CSV preamble can include comment-style lines starting with
    # "==PROF==" or similar. The real header is the first line that
    # contains "Kernel Name" (most reliable identifier across versions).
    header_idx = None
    for i, ln in enumerate(lines):
        if "Kernel Name" in ln or '"Kernel Name"' in ln:
            header_idx = i
            break
    if header_idx is None:
        raise NcuCsvMalformed(
            path, reason="no header row with 'Kernel Name' column"
        )

    reader = csv.DictReader(lines[header_idx:])
    fieldnames = reader.fieldnames or []
    rows = list(reader)
    # ncu-2026 long/melted format (--page details): one row per metric with
    # "Metric Name"/"Metric Value" columns. Pivot back to the wide shape (the
    # unit is folded into the column name there, so units map stays empty).
    if "Metric Name" in fieldnames and "Metric Value" in fieldnames:
        pivoted = _pivot_long_rows(rows)
        if not pivoted:
            raise NcuCsvMalformed(
                path, reason="long-format CSV has no (kernel, metric) rows"
            )
        return pivoted, {}
    # ncu 2026.1.1 emits a units-row between the header and the first data
    # row (e.g. dram bytes column shows "Mbyte" instead of a numeric value,
    # Kernel Name column is empty). Capture that units row (keyed by column)
    # BEFORE filtering -- callers need it to scale byte counters to bytes --
    # then drop it. The data-row check uses Kernel Name across all known ncu
    # CSV variants; if empty/missing, it's not a real kernel-instance row.
    def _is_data_row(r: dict[str, str]) -> bool:
        return bool((r.get("Kernel Name") or r.get("kernel_name") or "").strip())

    units: dict[str, str] = {}
    for r in rows:
        if _is_data_row(r):
            continue
        # First non-data row is the units row; record its non-empty cells.
        for col, val in r.items():
            if col and val and val.strip():
                units.setdefault(col, val.strip())
        break
    rows = [r for r in rows if _is_data_row(r)]
    if not rows:
        raise NcuCsvMalformed(path, reason="header present but no data rows")
    return rows, units


def _find_column(row: dict[str, str], *substrings: str) -> str | None:
    """Tolerant column lookup: returns the value of the first column whose
    name contains ALL of the given substrings (case-insensitive)."""
    for key in row.keys():
        kl = key.lower()
        if all(s.lower() in kl for s in substrings):
            return row[key]
    return None


def _parse_float(s: str | None) -> float | None:
    """Parse a possibly-formatted ncu numeric cell. Returns None if absent
    or unparseable. Handles ``"1,234.5"`` (locale-formatted) and ``"N/A"``."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace('"', "")
    if not s or s.upper() in ("N/A", "NA", "INF", "-INF"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _duration_ns(row: dict[str, str]) -> float | None:
    """Extract kernel Duration from a (pivoted) SoL row, normalised to ns.

    ncu's SoL section reports ``Duration`` with an explicit unit (us by
    default; ms / ns on very short / long kernels). The pivoted column is
    ``"Duration [<unit>]"``; we convert to ns so the JSON field is unit-stable.
    """
    for key in row.keys():
        kl = key.lower()
        if not kl.startswith("duration"):
            continue
        v = _parse_float(row[key])
        if v is None:
            return None
        if "[ns]" in kl or "[nsecond]" in kl:
            return v
        if "[ms]" in kl or "[msecond]" in kl:
            return v * 1_000_000.0
        if "[s]" in kl or "[second]" in kl:
            return v * 1_000_000_000.0
        # ncu SoL Duration defaults to microseconds.
        return v * 1_000.0
    return None


def _classify_block_limit(row: dict[str, str]) -> str | None:
    """Pick the block-limit-factor from a SOL-CSV row.

    ncu reports ``Block Limit Registers``, ``Block Limit Shared Mem``,
    ``Block Limit Warps`` as numeric values; the smallest one is the
    binding constraint. Returns one of {"registers", "shared_mem",
    "warps", None}.
    """
    candidates: list[tuple[str, float]] = []
    for name, key in [
        ("registers", "Block Limit Registers"),
        ("shared_mem", "Block Limit Shared Mem"),
        ("warps", "Block Limit Warps"),
    ]:
        v = _parse_float(_find_column(row, key))
        if v is not None:
            candidates.append((name, v))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1])
    return candidates[0][0]


def _aggregate_per_kernel(
    sol_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    raw_units: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Merge SOL + raw CSV rows by kernel name.

    ncu --launch-count N produces N rows per kernel (one per replayed
    launch). We average the percentages and sum the byte/flop counters
    so the emitted ``ncu_kernels.json`` has one row per kernel name.

    ``raw_units`` maps each raw-page column name to its ncu unit string (from
    the units row); byte counters are scaled to bytes accordingly. None / an
    empty map means "assume bytes" (back-compat with unit-less callers).
    """
    raw_units = raw_units or {}
    # Group SOL rows by kernel name (use the part before any "(" for
    # nicer display; ncu reports the full mangled signature).
    sol_by_kernel: dict[str, list[dict[str, str]]] = {}
    for r in sol_rows:
        name = r.get("Kernel Name") or r.get("kernel_name") or r.get("ID")
        if not name:
            raise NcuCsvMalformed(
                Path("<sol-csv>"), reason=f"row missing 'Kernel Name': {list(r.keys())[:5]}"
            )
        sol_by_kernel.setdefault(name, []).append(r)

    raw_by_kernel: dict[str, list[dict[str, str]]] = {}
    for r in raw_rows:
        name = r.get("Kernel Name") or r.get("kernel_name") or r.get("ID")
        if not name:
            raise NcuCsvMalformed(
                Path("<raw-csv>"), reason=f"row missing 'Kernel Name': {list(r.keys())[:5]}"
            )
        raw_by_kernel.setdefault(name, []).append(r)

    out: list[dict[str, Any]] = []
    for name, sol_group in sol_by_kernel.items():
        # Mean across launches for percentages.
        dram_pct = _mean_optional([
            _parse_float(_find_column(r, "DRAM Throughput", "%")) for r in sol_group
        ])
        sm_pct = _mean_optional([
            _parse_float(_find_column(r, "Compute (SM) Throughput", "%")) for r in sol_group
        ])
        occupancy = _mean_optional([
            _parse_float(_find_column(r, "Achieved Occupancy")) for r in sol_group
        ])
        # SoL Duration (normalised to ns); preferred over the raw-page time
        # below because it carries an explicit unit. None when the SoL
        # section / Duration metric is absent.
        sol_time_ns = _mean_optional([_duration_ns(r) for r in sol_group])
        # Block-limit factor: take the most common across launches.
        block_limit = _mode([_classify_block_limit(r) for r in sol_group])

        # Sum across launches for byte / flop counters from raw page.
        # ncu raw page column names vary by version; we sum every column
        # that matches "dram__bytes" / "sm__sass_thread_inst_executed_op"
        # / "gpu__time_active". Tolerant lookup keeps this robust across
        # ncu 2025.x -> 2026.x column renames.
        raw_group = raw_by_kernel.get(name, [])
        # Occupancy fallback: ncu --set=basic SoL section may omit
        # "Achieved Occupancy"; the raw page carries the equivalent
        # sm__warps_active.avg.pct_of_peak_sustained_active.
        if occupancy is None:
            occupancy = _mean_optional([
                _parse_float(
                    _find_column(r, "sm__warps_active.avg.pct_of_peak_sustained_active")
                )
                for r in raw_group
            ])
        # DRAM bytes, unit-scaled to bytes (the units row gave e.g. "Mbyte").
        # ncu expands a requested counter into .avg/.max/.min/.sum columns; the
        # per-kernel TOTAL is the .sum column. Match that specifically -- a naive
        # "first dram__bytes" match picks .avg (a ~1/launch-count fraction).
        def _dram_bytes_scaled(r: dict[str, str]) -> float | None:
            chosen = None
            for key in r.keys():
                kl = key.lower()
                if "dram__bytes" not in kl or "per_second" in kl or "pct" in kl:
                    continue
                if kl.endswith(".sum"):
                    chosen = key
                    break
                if chosen is None:  # fallback: a bare dram__bytes col with no .sum sibling
                    chosen = key
            if chosen is None:
                return None
            v = _parse_float(r[chosen])
            if v is None:
                return None
            return v * _byte_scale(raw_units.get(chosen))

        dram_bytes = _sum_optional([_dram_bytes_scaled(r) for r in raw_group])
        # FLOPs come from two disjoint counter families:
        #   - CUDA-core SCALAR ops: sm__sass_thread_inst_executed_op_*.sum
        #     (softmax / scaling / dequant etc.) -- summed as-is.
        #   - TENSOR-core MMA ops: sm__ops_path_tensor_*.sum -- the dominant
        #     compute in fp8/NVFP4 GEMM/attention kernels, which the scalar
        #     counters do NOT see. Scaled MAC->FLOP by _TENSOR_OP_TO_FLOP.
        # Tracked separately (emitted for traceability) and summed into the
        # total used for arithmetic intensity + achieved TFLOPS, so a
        # tensor-core kernel's roofline point is physically meaningful rather
        # than a scalar-only lower bound.
        scalar_per_launch: list[float] = []
        tensor_per_launch: list[float] = []
        for r in raw_group:
            scalar = 0.0
            tensor = 0.0
            had_scalar = had_tensor = False
            for key, val in r.items():
                kl = key.lower()
                # Only the .sum aggregate (NOT .avg/.max/.min, nor derived
                # .sum.per_second / .sum.pct_of_peak). endswith is precise.
                if not kl.endswith(".sum"):
                    continue
                v = _parse_float(val)
                if v is None:
                    continue
                if "sm__sass_thread_inst_executed_op" in kl:
                    scalar += v
                    had_scalar = True
                elif "sm__ops_path_tensor" in kl:
                    tensor += v
                    had_tensor = True
            if had_scalar:
                scalar_per_launch.append(scalar)
            if had_tensor:
                tensor_per_launch.append(tensor)
        sm_flops_total = sum(scalar_per_launch) if scalar_per_launch else None
        tensor_flops_total = (
            sum(tensor_per_launch) * _TENSOR_OP_TO_FLOP if tensor_per_launch else None
        )

        # Wall-clock time. Two distinct quantities:
        #  - time_ns_sum: active time SUMMED across launches -- the correct
        #    denominator for the FLOPS rate, since dram_bytes / *_flops_total are
        #    ALSO summed across launches (sum/sum keeps the rate per-launch-true).
        #  - time_ns (reported kernel_time_ns): prefer the SoL Duration (explicit
        #    unit); else the summed raw time (preserves the legacy semantic that
        #    multi-launch tests assert).
        def _raw_time_ns(r: dict[str, str]) -> float | None:
            for pat in ("gpu__time_active.sum", "gpu__time_duration.sum", "gpu__time_duration"):
                for key in r.keys():
                    if pat in key.lower():
                        v = _parse_float(r[key])
                        if v is None:
                            return None
                        return v * _time_scale(raw_units.get(key))
            return None

        time_ns_sum = _sum_optional([_raw_time_ns(r) for r in raw_group])
        time_ns = sol_time_ns if sol_time_ns is not None else time_ns_sum
        # FLOPS-rate denominator: total active time across launches. PREFER the
        # SoL Duration (x launch count) -- it carries an explicit unit and is
        # normalised to ns by _duration_ns. The raw gpu__time_active.sum is in a
        # version-dependent unit (often usecond) that the raw page does NOT
        # unit-convert, so using it directly would inflate the rate ~1000x.
        n_launch = len(raw_group) or len(sol_group) or 1
        rate_time_ns = (
            sol_time_ns * n_launch if sol_time_ns is not None else time_ns_sum
        )

        # Total FLOPs = scalar (CUDA-core) + tensor (MMA). Either may be None;
        # combine the present ones. For a tensor-core kernel this is dominated
        # by tensor_flops_total -- the whole point of the L4 roofline.
        total_flops = None
        if sm_flops_total is not None or tensor_flops_total is not None:
            total_flops = (sm_flops_total or 0.0) + (tensor_flops_total or 0.0)

        # Derived: arithmetic intensity (sum/sum) + achieved TFLOPS (sum/sum).
        ai = None
        if dram_bytes and total_flops and dram_bytes > 0:
            ai = total_flops / dram_bytes
        achieved_tflops = None
        if total_flops and rate_time_ns and rate_time_ns > 0:
            achieved_tflops = total_flops / (rate_time_ns * 1e-9) / 1e12

        # Trim mangled name down to the head (everything before "<" or "(")
        # so the JSON is human-readable but the original full name is
        # kept under "name_full" for traceability.
        short_name = re.split(r"[<(]", name, maxsplit=1)[0]

        out.append({
            "name": short_name,
            "name_full": name,
            "category": _categorize(name),
            "kernel_time_ns": time_ns,
            "dram_bytes_total": dram_bytes,
            "sm_flops_total": sm_flops_total,
            "tensor_flops_total": tensor_flops_total,
            "arithmetic_intensity_flops_per_byte": ai,
            "achieved_dram_pct_peak": dram_pct,
            "achieved_sm_pct_peak": sm_pct,
            "achieved_occupancy_pct": occupancy,
            "block_limit_factor": block_limit,
            "achieved_tflops": achieved_tflops,
        })

    return out


def _mean_optional(vals: list[float | None]) -> float | None:
    """Mean of non-None values; None if all are None."""
    keep = [v for v in vals if v is not None]
    if not keep:
        return None
    return sum(keep) / len(keep)


def _sum_optional(vals: list[float | None]) -> float | None:
    """Sum of non-None values; None if all are None."""
    keep = [v for v in vals if v is not None]
    if not keep:
        return None
    return sum(keep)


def _mode(vals: list[str | None]) -> str | None:
    """Modal non-None value; None if all are None."""
    keep = [v for v in vals if v is not None]
    if not keep:
        return None
    counts: dict[str, int] = {}
    for v in keep:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def import_ncu_kernels(
    bundle: Path,
    cell_dir: Path,
    *,
    hw_key: str = "b200_sm100",
    dry_run: bool = False,
) -> NcuImportResult:
    """Import ncu per-kernel data from a bundle into a cell directory.

    Args:
        bundle: ``*-deploy/experiments/artifacts/ncu-perkernel/<bundle>/``
            path. Must contain ``capture_sources.json`` declaring "ncu" +
            an ``ncu-profiles/`` dir with paired ``*-sol.csv`` and
            ``*-raw.csv`` files (one pair per captured kernel).
        cell_dir: target ``<campaign>/cells/<cell_id>/`` directory where
            ``ncu_kernels.json`` will be written.
        hw_key: which ``sol-ceilings.yaml`` hardware key the kernels were
            captured on. Defaults to ``b200_sm100``; pass
            ``"gb300_nvl72"`` for GB300 captures.
        dry_run: if True, parse + validate but do NOT write
            ``ncu_kernels.json``.

    Returns:
        ``NcuImportResult``. ``ncu_kernels_json_path`` is ``None`` if the
        manifest did not declare ncu (correct no-op).

    Raises:
        NcuCsvMissing: manifest declared ncu but no CSV pair found.
        NcuCsvMalformed: manifest declared ncu but a CSV is unparseable.
    """
    bundle = bundle.expanduser().resolve()
    cell_dir = cell_dir.expanduser().resolve()

    manifest = _read_manifest(bundle)
    if manifest is None or "ncu" not in (manifest.get("captured_sources") or []):
        return NcuImportResult(
            bundle=bundle,
            ncu_kernels_json_path=None,
            skipped_reason="capture_sources.json absent or does not declare ncu",
            kernel_count=0,
        )

    profiles_dir = bundle / _PROFILES_SUBDIR
    if not profiles_dir.is_dir():
        raise NcuCsvMissing(profiles_dir, reason="absent")

    # Find every -sol.csv with a matching -raw.csv sibling.
    sol_files = sorted(profiles_dir.glob("*-sol.csv"))
    if not sol_files:
        raise NcuCsvMissing(profiles_dir, reason="no *-sol.csv files found")

    all_sol_rows: list[dict[str, str]] = []
    all_raw_rows: list[dict[str, str]] = []
    raw_units: dict[str, str] = {}
    for sol_path in sol_files:
        raw_path = sol_path.with_name(sol_path.name.replace("-sol.csv", "-raw.csv"))
        if not raw_path.is_file():
            raise NcuCsvMissing(raw_path, reason="absent (no matching -raw.csv)")
        sol_rows, _sol_units = _read_ncu_csv(sol_path)
        raw_rows, raw_units_one = _read_ncu_csv(raw_path)
        all_sol_rows.extend(sol_rows)
        all_raw_rows.extend(raw_rows)
        # Units are per-column and identical across files of the same shape;
        # merge (later files don't override an earlier non-empty unit).
        for col, unit in raw_units_one.items():
            raw_units.setdefault(col, unit)

    kernels = _aggregate_per_kernel(all_sol_rows, all_raw_rows, raw_units)

    payload = {
        "schema_version": 1,
        "captured_sources": manifest.get("captured_sources", []),
        "hw_key": hw_key,
        "kernels": kernels,
    }

    out_path = cell_dir / _NCU_KERNELS_OUTPUT
    if not dry_run:
        cell_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return NcuImportResult(
        bundle=bundle,
        ncu_kernels_json_path=out_path,
        skipped_reason=None,
        kernel_count=len(kernels),
    )
