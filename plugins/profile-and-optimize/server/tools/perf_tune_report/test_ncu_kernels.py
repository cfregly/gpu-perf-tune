"""Tests for the ncu_kernels declared-coverage contract.

Mirrors ``test_zymtrace_kernels.py`` 8-scenario structure but for ncu
CSVs:

1. positive_path        -- manifest declares ncu + paired sol+raw CSVs -> ncu_kernels.json emitted
2. declared_no_profiles -- manifest declares ncu + ncu-profiles dir absent -> NcuCsvMissing
3. declared_no_pair     -- manifest declares ncu + sol.csv exists but raw.csv missing -> NcuCsvMissing
4. declared_empty       -- manifest declares ncu + 0-byte CSV -> NcuCsvMissing
5. declared_malformed   -- manifest declares ncu + CSV with no "Kernel Name" header -> NcuCsvMalformed
6. no_manifest_skip     -- no manifest -> silent skip, no ncu_kernels.json
7. dry_run              -- dry_run=True parses + validates but writes no file
8. multi_launch         -- --launch-count 5 yields 5 rows per kernel; importer averages SOL pct + sums byte/flop counts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.importers.ncu_kernels import (
    NcuCsvMalformed,
    NcuCsvMissing,
    import_ncu_kernels,
)


# ---------------------------------------------------------------------------
# Synthetic ncu CSV helpers
# ---------------------------------------------------------------------------


_SOL_HEADER = (
    '"ID","Kernel Name","DRAM Throughput [%]","Compute (SM) Throughput [%]",'
    '"Achieved Occupancy","Block Limit Registers","Block Limit Shared Mem","Block Limit Warps"'
)


_RAW_HEADER = (
    '"ID","Kernel Name","dram__bytes.sum","sm__sass_thread_inst_executed_op_dadd_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_dmul_pred_on.sum","gpu__time_active.sum"'
)


def _sol_csv(rows: list[tuple[str, float, float, float, float, float, float]]) -> str:
    """rows: list of (kernel_name, dram_pct, sm_pct, occupancy, regs_limit, smem_limit, warps_limit)."""
    lines = [_SOL_HEADER]
    for i, (name, dram, sm, occ, regs, smem, warps) in enumerate(rows):
        lines.append(f'"{i}","{name}","{dram}","{sm}","{occ}","{regs}","{smem}","{warps}"')
    return "\n".join(lines) + "\n"


def _raw_csv(rows: list[tuple[str, float, float, float, float]]) -> str:
    """rows: list of (kernel_name, dram_bytes, dadd_flops, dmul_flops, time_ns)."""
    lines = [_RAW_HEADER]
    for i, (name, bytes_, dadd, dmul, time_ns) in enumerate(rows):
        lines.append(f'"{i}","{name}","{bytes_}","{dadd}","{dmul}","{time_ns}"')
    return "\n".join(lines) + "\n"


_VALID_MANIFEST = {
    "schema_version": 1,
    "captured_sources": ["ncu"],
    "captured_at": "2026-05-27T20:30:00Z",
    "captured_by": "test_ncu_kernels@pytest",
    "cluster": "synthetic",
    "pod_name": "test-pod-1",
}


def _make_bundle(
    root: Path,
    *,
    manifest: dict | None,
    profiles: dict[str, str] | None = None,
) -> Path:
    """Build a synthetic ncu bundle directory.

    Args:
        root: bundle root dir to create.
        manifest: capture_sources.json contents (None = don't write).
        profiles: ``{"ncu-NAME-sol.csv": "<content>", "ncu-NAME-raw.csv": "<content>"}``.
            Pass empty dict to create ncu-profiles/ but with no files;
            pass None to NOT create ncu-profiles/ at all.
    """
    root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (root / "capture_sources.json").write_text(json.dumps(manifest, indent=2))
    if profiles is not None:
        prof_dir = root / "ncu-profiles"
        prof_dir.mkdir(exist_ok=True)
        for name, content in profiles.items():
            (prof_dir / name).write_text(content)
    return root


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


def test_positive_path(tmp_path):
    """S1: manifest declares ncu + paired sol+raw CSVs -> ncu_kernels.json."""
    sol = _sol_csv([
        ("multimem_all_reduce_kernel<bfloat16>", 92.0, 8.5, 0.31, 8.0, 6.0, 4.0),
        ("bmm_E2m1E2m1_Fp32_sm100f", 45.0, 88.0, 0.62, 8.0, 6.0, 8.0),
    ])
    raw = _raw_csv([
        ("multimem_all_reduce_kernel<bfloat16>", 1.2e9, 1.5e8, 8e7, 12340.0),
        ("bmm_E2m1E2m1_Fp32_sm100f", 5e8, 8e9, 5e9, 23000.0),
    ])
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol, "ncu-k1-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-1"

    result = import_ncu_kernels(bundle, cell_dir)

    assert result.ncu_kernels_json_path is not None
    assert result.ncu_kernels_json_path.is_file()
    assert result.skipped_reason is None
    assert result.kernel_count == 2

    payload = json.loads(result.ncu_kernels_json_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["hw_key"] == "b200_sm100"
    assert "ncu" in payload["captured_sources"]
    assert len(payload["kernels"]) == 2

    # Categorisation
    nccl = next(k for k in payload["kernels"] if k["name"] == "multimem_all_reduce_kernel")
    assert nccl["category"] == "NCCL"
    assert nccl["achieved_dram_pct_peak"] == 92.0
    assert nccl["dram_bytes_total"] == 1.2e9
    assert nccl["sm_flops_total"] == 1.5e8 + 8e7  # dadd + dmul
    # Block limit factor: warps_limit (4.0) < shared (6.0) < regs (8.0) -> "warps"
    assert nccl["block_limit_factor"] == "warps"
    # Arithmetic intensity: flops/bytes
    assert nccl["arithmetic_intensity_flops_per_byte"] == pytest.approx(
        (1.5e8 + 8e7) / 1.2e9, rel=1e-6
    )

    bmm = next(k for k in payload["kernels"] if k["name"] == "bmm_E2m1E2m1_Fp32_sm100f")
    assert bmm["category"] == "BMM-NVFP4"
    assert bmm["achieved_sm_pct_peak"] == 88.0


def test_declared_no_profiles_dir(tmp_path):
    """S2: manifest declares ncu but no ncu-profiles/ dir -> NcuCsvMissing."""
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, profiles=None)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-2"

    with pytest.raises(NcuCsvMissing) as ei:
        import_ncu_kernels(bundle, cell_dir)
    assert ei.value.reason == "absent"


def test_declared_no_pair(tmp_path):
    """S3: manifest declares ncu + sol.csv exists but raw.csv missing -> NcuCsvMissing."""
    sol = _sol_csv([("multimem_all_reduce_kernel", 92.0, 8.5, 0.31, 8.0, 6.0, 4.0)])
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol},  # no matching raw.csv
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-3"

    with pytest.raises(NcuCsvMissing) as ei:
        import_ncu_kernels(bundle, cell_dir)
    assert "raw.csv" in str(ei.value.path) or "no matching" in ei.value.reason


def test_declared_empty_file(tmp_path):
    """S4: 0-byte CSV -> NcuCsvMissing(reason=empty)."""
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={
            "ncu-k1-sol.csv": "",
            "ncu-k1-raw.csv": _raw_csv([("k1", 1e9, 1e8, 1e8, 1000.0)]),
        },
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-4"

    with pytest.raises(NcuCsvMissing) as ei:
        import_ncu_kernels(bundle, cell_dir)
    assert ei.value.reason == "empty"


def test_declared_malformed_no_header(tmp_path):
    """S5: CSV lacks 'Kernel Name' header -> NcuCsvMalformed."""
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={
            "ncu-k1-sol.csv": "garbage,data,no,kernel,header\n1,2,3,4,5\n",
            "ncu-k1-raw.csv": _raw_csv([("k1", 1e9, 1e8, 1e8, 1000.0)]),
        },
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-5"

    with pytest.raises(NcuCsvMalformed) as ei:
        import_ncu_kernels(bundle, cell_dir)
    assert "Kernel Name" in ei.value.reason


def test_no_manifest_silent_skip(tmp_path):
    """S6: no manifest -> silent skip, no ncu_kernels.json emitted."""
    bundle = _make_bundle(tmp_path / "bundle", manifest=None, profiles=None)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-6"

    result = import_ncu_kernels(bundle, cell_dir)
    assert result.ncu_kernels_json_path is None
    assert result.skipped_reason is not None
    assert "ncu" in result.skipped_reason


def test_manifest_without_ncu_in_sources_skip(tmp_path):
    """S6b: manifest declares OTHER sources (e.g. only zymtrace) -> silent skip."""
    manifest = dict(_VALID_MANIFEST)
    manifest["captured_sources"] = ["zymtrace"]  # NOT ncu
    bundle = _make_bundle(tmp_path / "bundle", manifest=manifest, profiles=None)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-6b"

    result = import_ncu_kernels(bundle, cell_dir)
    assert result.ncu_kernels_json_path is None


def test_dry_run_does_not_write(tmp_path):
    """S7: dry_run=True parses + validates but emits no file."""
    sol = _sol_csv([("k1", 50.0, 50.0, 0.5, 8.0, 6.0, 4.0)])
    raw = _raw_csv([("k1", 1e9, 1e8, 1e8, 1000.0)])
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol, "ncu-k1-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-7"

    result = import_ncu_kernels(bundle, cell_dir, dry_run=True)
    assert result.ncu_kernels_json_path is not None
    assert not result.ncu_kernels_json_path.exists()  # didn't actually write
    assert result.kernel_count == 1


def test_multi_launch_aggregates_correctly(tmp_path):
    """S8: --launch-count N emits N rows per kernel.

    The importer averages SOL percentages and sums byte/flop counts.
    """
    # Same kernel, 3 launches with slightly different numbers
    sol = _sol_csv([
        ("k1", 90.0, 10.0, 0.30, 8, 6, 4),
        ("k1", 92.0, 11.0, 0.32, 8, 6, 4),
        ("k1", 94.0, 12.0, 0.34, 8, 6, 4),
    ])
    raw = _raw_csv([
        ("k1", 1e9, 1e8, 1e8, 1000.0),
        ("k1", 1e9, 1e8, 1e8, 1000.0),
        ("k1", 1e9, 1e8, 1e8, 1000.0),
    ])
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol, "ncu-k1-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-8"

    result = import_ncu_kernels(bundle, cell_dir)
    payload = json.loads(result.ncu_kernels_json_path.read_text())
    assert result.kernel_count == 1
    k = payload["kernels"][0]
    # Averaged
    assert k["achieved_dram_pct_peak"] == pytest.approx(92.0, rel=1e-6)
    assert k["achieved_sm_pct_peak"] == pytest.approx(11.0, rel=1e-6)
    # Summed
    assert k["dram_bytes_total"] == 3e9
    assert k["sm_flops_total"] == 3 * 2e8  # 3 launches x (1e8 dadd + 1e8 dmul)
    assert k["kernel_time_ns"] == 3000.0


def test_hw_key_param_propagates_to_payload(tmp_path):
    """hw_key parameter selects which sol-ceilings.yaml column to use."""
    sol = _sol_csv([("k1", 50.0, 50.0, 0.5, 8.0, 6.0, 4.0)])
    raw = _raw_csv([("k1", 1e9, 1e8, 1e8, 1000.0)])
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol, "ncu-k1-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-gb300"

    result = import_ncu_kernels(bundle, cell_dir, hw_key="gb300_nvl72")
    payload = json.loads(result.ncu_kernels_json_path.read_text())
    assert payload["hw_key"] == "gb300_nvl72"


def test_invalid_manifest_json_raises(tmp_path):
    """Manifest exists but is bad JSON -> NcuCsvMalformed."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "capture_sources.json").write_text("{not json")
    cell_dir = tmp_path / "campaign" / "cells" / "cell-bad-manifest"

    with pytest.raises(NcuCsvMalformed):
        import_ncu_kernels(bundle, cell_dir)


def _long_sol_csv(kernel: str, metrics: list[tuple[str, str, str]]) -> str:
    """Build an ncu-2026 long/melted SoL CSV (one row per metric).

    metrics: list of (metric_name, metric_unit, metric_value).
    Mirrors the real ``--page details --section SpeedOfLight`` export shape.
    """
    header = (
        '"ID","Process ID","Kernel Name","Section Name",'
        '"Metric Name","Metric Unit","Metric Value","Rule Name"'
    )
    lines = [header]
    for mname, munit, mval in metrics:
        lines.append(
            f'"0","942","{kernel}","GPU Speed Of Light Throughput",'
            f'"{mname}","{munit}","{mval}",""'
        )
    # ncu also emits a trailing rule row with empty Metric Name -> must be skipped.
    lines.append(
        f'"0","942","{kernel}","SpeedOfLight","","","","SOLBottleneck"'
    )
    return "\n".join(lines) + "\n"


def test_long_format_sol_only_basic_set(tmp_path):
    """S10: ncu-2026 long-format SoL CSV from a ``--set=basic`` capture.

    The SoL section carries throughput percentages + Duration but NO
    FLOPS / DRAM-byte counters, so arithmetic intensity is unrecoverable.
    The importer MUST: pivot the long rows, populate the %SoL fields +
    kernel_time_ns, and leave arithmetic_intensity / sm_flops_total /
    dram_bytes_total / achieved_tflops null (never fabricated). This is the
    real glm51-ncu-20260529 triton_red_fused_2 shape.
    """
    sol = _long_sol_csv(
        "triton_red_fused_2",
        [
            ("Memory Throughput", "%", "51.85"),
            ("DRAM Throughput", "%", "19.04"),
            ("Compute (SM) Throughput", "%", "74.60"),
            ("Duration", "us", "89.47"),
        ],
    )
    # Basic-set raw page: no dram__bytes / sm__sass_thread_inst_executed_op
    # columns; carries the occupancy proxy instead.
    raw = (
        '"ID","Kernel Name","sm__warps_active.avg.pct_of_peak_sustained_active"\n'
        '"0","triton_red_fused_2","91.13"\n'
    )
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-glm51-sol.csv": sol, "ncu-glm51-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-long"

    result = import_ncu_kernels(bundle, cell_dir)
    assert result.kernel_count == 1
    payload = json.loads(result.ncu_kernels_json_path.read_text())
    k = payload["kernels"][0]
    assert k["name"] == "triton_red_fused_2"
    assert k["category"] == "Triton-fused"
    assert k["achieved_sm_pct_peak"] == pytest.approx(74.60)
    assert k["achieved_dram_pct_peak"] == pytest.approx(19.04)
    assert k["achieved_occupancy_pct"] == pytest.approx(91.13)  # raw fallback
    assert k["kernel_time_ns"] == pytest.approx(89470.0)  # 89.47 us -> ns
    # AI + compute counters are genuinely unmeasured at --set=basic: null.
    assert k["arithmetic_intensity_flops_per_byte"] is None
    assert k["sm_flops_total"] is None
    assert k["dram_bytes_total"] is None
    assert k["achieved_tflops"] is None


def test_ncu_2026_1_1_units_row_filtered(tmp_path):
    """S9: ncu 2026.1.1 emits a units-row between header and first data row
    (Kernel Name column is empty; numeric columns show the unit string e.g.
    "byte" instead of a value). The importer MUST filter that row out before
    aggregation so the bogus "kernel-with-empty-name" doesn't fail the
    `_aggregate_per_kernel()` check.

    Resolves TODO-NCU-IMPORTER-UNITS-ROW per profile_and_optimize OPERATOR-TODO.md.
    """
    # Simulate the ncu 2026.1.1 units-row by inserting an empty-Kernel-Name
    # row between the header and the real kernel-instance row.
    sol_with_units_row = (
        _SOL_HEADER + "\n"
        '"","","%","%","","","",""\n'  # units-row (Kernel Name empty, "%" units)
        + '"0","triton_red_fused","45.0","88.0","0.62","8.0","6.0","8.0"\n'
    )
    raw_with_units_row = (
        _RAW_HEADER + "\n"
        '"","","byte","inst","inst","ns"\n'  # units-row
        + '"0","triton_red_fused","5e8","8e9","5e9","23000.0"\n'
    )
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={
            "ncu-k1-sol.csv": sol_with_units_row,
            "ncu-k1-raw.csv": raw_with_units_row,
        },
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-units-row"

    # Should NOT raise; should produce exactly 1 kernel (the units-row filtered).
    result = import_ncu_kernels(bundle, cell_dir)
    assert result.kernel_count == 1
    payload = json.loads(result.ncu_kernels_json_path.read_text())
    assert len(payload["kernels"]) == 1
    assert payload["kernels"][0]["name"] == "triton_red_fused"


def test_dram_bytes_mbyte_unit_scaled_to_bytes(tmp_path):
    """S11 (1a): ncu reports dram__bytes.sum in "Mbyte" via the units row.

    The importer MUST read that unit and scale the value to bytes (x1e6),
    so dram_bytes_total + the arithmetic-intensity denominator are correct.
    Regression for the unit-unaware sum that produced ~1e6-too-small bytes
    (the published dram_bytes_total=11.64 / 90 artifacts).
    """
    sol = (
        _SOL_HEADER + "\n"
        '"","","%","%","","","",""\n'
        + '"0","fmha_test","48.6","24.6","0.23","8.0","6.0","8.0"\n'
    )
    raw = (
        _RAW_HEADER + "\n"
        '"","","Mbyte","inst","inst","usecond"\n'  # units row: bytes in Mbyte
        + '"0","fmha_test","248.4","1000","2000","66.6"\n'
    )
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol, "ncu-k1-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-mbyte"

    result = import_ncu_kernels(bundle, cell_dir)
    k = json.loads(result.ncu_kernels_json_path.read_text())["kernels"][0]
    # 248.4 Mbyte -> 248.4e6 bytes (NOT 248.4 raw).
    assert k["dram_bytes_total"] == pytest.approx(248.4e6)
    # AI denominator uses the scaled bytes.
    assert k["arithmetic_intensity_flops_per_byte"] == pytest.approx(
        3000.0 / 248.4e6, rel=1e-6
    )


def test_tensor_core_flops_folded_into_ai(tmp_path):
    """S12 (1b): a tensor-core kernel carries sm__ops_path_tensor_*.sum.

    Those tensor MMA ops (the dominant compute in fp8/NVFP4 kernels) MUST be
    summed into tensor_flops_total (x2 MAC->FLOP) and folded into the
    arithmetic intensity + achieved TFLOPS -- NOT left out (which would make
    the AI a scalar-only lower bound). Regression for the op-count-only AI.
    """
    sol = _sol_csv([("fmhaSm100f_test", 48.6, 24.6, 0.23, 8.0, 6.0, 8.0)])
    # Raw page: byte unit (x1 for easy math) + one CUDA-core scalar op col +
    # one tensor-core op col + time.
    raw = (
        '"ID","Kernel Name","dram__bytes.sum",'
        '"sm__sass_thread_inst_executed_op_ffma_pred_on.sum",'
        '"sm__ops_path_tensor_src_fp8_dst_fp32.sum","gpu__time_active.sum"\n'
        '"","","byte","inst","inst","nsecond"\n'
        '"0","fmhaSm100f_test","1000000.0","100.0","5000000.0","1000.0"\n'
    )
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-k1-sol.csv": sol, "ncu-k1-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-tensor"

    result = import_ncu_kernels(bundle, cell_dir)
    k = json.loads(result.ncu_kernels_json_path.read_text())["kernels"][0]
    # tensor_flops_total = 5e6 ops x 1.0 (the metric already counts FLOPs) = 5e6.
    assert k["tensor_flops_total"] == pytest.approx(5e6)
    # scalar stays separate (CUDA-core ffma only).
    assert k["sm_flops_total"] == pytest.approx(100.0)
    # AI uses scalar + tensor over bytes -- dominated by tensor, NOT 100/1e6.
    assert k["arithmetic_intensity_flops_per_byte"] == pytest.approx(
        (5e6 + 100.0) / 1e6, rel=1e-6
    )
    # achieved TFLOPS computed off the total (non-null, tensor-inclusive).
    assert k["achieved_tflops"] is not None and k["achieved_tflops"] > 0


def test_raw_time_usecond_unit_scaled_for_tflops(tmp_path):
    """S13: with no SoL Duration, the FLOPS-rate denominator comes from the raw
    gpu__time_active.sum, which ncu reports in usecond. It MUST be unit-scaled
    to ns -- else achieved_tflops is ~1000x inflated. (Real-capture regression.)
    """
    # No SoL Duration column -> rate falls back to raw time.
    sol = (
        '"ID","Kernel Name","DRAM Throughput [%]","Compute (SM) Throughput [%]"\n'
        '"0","k_t","48.0","25.0"\n'
    )
    # 1 GFLOP over 1000 usecond (= 1e-3 s) -> 1e9 / 1e-3 / 1e12 = 1.0 TFLOPS
    # WHEN the usecond unit is scaled to ns. Unscaled (treating 1000 as ns) it
    # would be ~1000 TFLOPS -- so 1.0 proves the unit scaling fired.
    raw = (
        '"ID","Kernel Name","dram__bytes.sum",'
        '"sm__ops_path_tensor_src_fp4_fp6_fp8_dst_fp32.sum","gpu__time_active.sum"\n'
        '"","","byte","inst","usecond"\n'
        '"0","k_t","1000000.0","1000000000.0","1000.0"\n'
    )
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=_VALID_MANIFEST,
        profiles={"ncu-kt-sol.csv": sol, "ncu-kt-raw.csv": raw},
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-tns"
    result = import_ncu_kernels(bundle, cell_dir)
    k = json.loads(result.ncu_kernels_json_path.read_text())["kernels"][0]
    assert k["achieved_tflops"] == pytest.approx(1.0, rel=1e-6)
