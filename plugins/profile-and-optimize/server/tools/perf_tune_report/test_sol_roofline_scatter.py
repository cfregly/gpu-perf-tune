"""Tests for the page-5 SoL roofline scatter renderer.

Mirrors ``test_sol_roofline.py`` scenario set for the ncu-data-driven
counterpart.

Scenarios:
A. compute helpers     -- _peak_compute_tflops, _peak_bandwidth_tbps
B. render_page contract -- non-empty / hardware key required
C. discover_ncu_payloads -- happy path + missing + malformed
D. End-to-end via render_report:
    - ncu_kernels.json present + sol-ceilings.yaml present  -> 5+ pages
    - ncu_kernels.json present + no yaml                     -> 4 pages (page 5 skipped)
    - ncu_kernels.json absent                                -> 4 or fewer pages
    - ncu_kernels.json malformed                             -> NcuKernelsJsonMalformed
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.renderer import sol_roofline_scatter
from tools.perf_tune_report.renderer.render_report import (
    NcuKernelsJsonMalformed,
    discover_ncu_payloads,
    render_report,
)
from tools.perf_tune_report.renderer.sol_roofline_scatter import (
    _peak_bandwidth_tbps,
    _peak_compute_tflops,
)
from tools.perf_tune_report.schema import BACKEND_VLLM_SWEEP, STATUS_FULL, AtlasCell


# ---------------------------------------------------------------------------
# Fixtures (re-derived from test_sol_roofline.py to avoid coupling)
# ---------------------------------------------------------------------------


_MIN_CEILINGS: dict = {
    "b200_sm100": {
        "hw_name": "NVIDIA B200 test",
        "nvfp4_dense_pflops": {"value": 9.0, "units": "PFLOPS", "source": "test"},
        "fp8_dense_pflops": {"value": 4.5, "units": "PFLOPS", "source": "test"},
        "bf16_dense_pflops": {"value": 2.25, "units": "PFLOPS", "source": "test"},
        "hbm3e_tbps": {"value": 8.0, "units": "TB/s", "source": "test"},
        "nvlink5_tbps": {"value": 1.8, "units": "TB/s", "source": "test"},
    },
    "category_ceiling_map": {
        "NCCL": {"metric": "nvlink5_tbps", "bound": "bandwidth"},
        "MoE": {"metric": "nvfp4_dense_pflops", "bound": "compute"},
        "FMHA": {"metric": "hbm3e_tbps", "bound": "bandwidth"},
        "BMM-NVFP4": {"metric": "nvfp4_dense_pflops", "bound": "compute"},
        "Triton-fused": {"metric": "bf16_dense_pflops", "bound": "compute"},
        "cuBLAS": {"metric": "bf16_dense_pflops", "bound": "compute"},
        "Elementwise": {"metric": "hbm3e_tbps", "bound": "bandwidth"},
        "Other": {"metric": "hbm3e_tbps", "bound": "bandwidth"},
    },
}


_NCU_PAYLOAD = {
    "schema_version": 1,
    "captured_sources": ["ncu"],
    "hw_key": "b200_sm100",
    "kernels": [
        {
            "name": "multimem_all_reduce_kernel",
            "name_full": "multimem_all_reduce_kernel<bfloat16>",
            "category": "NCCL",
            "kernel_time_ns": 12340.0,
            "dram_bytes_total": 1.2e9,
            "sm_flops_total": 2.3e8,
            "arithmetic_intensity_flops_per_byte": 0.19,
            "achieved_dram_pct_peak": 92.0,
            "achieved_sm_pct_peak": 8.5,
            "achieved_occupancy_pct": 31.2,
            "block_limit_factor": "registers",
            "achieved_tflops": 18.6,
        },
        {
            "name": "bmm_E2m1E2m1_Fp32_sm100f",
            "name_full": "bmm_E2m1E2m1_Fp32_sm100f<...>",
            "category": "BMM-NVFP4",
            "kernel_time_ns": 23000.0,
            "dram_bytes_total": 5e8,
            "sm_flops_total": 1.3e10,
            "arithmetic_intensity_flops_per_byte": 26.0,
            "achieved_dram_pct_peak": 45.0,
            "achieved_sm_pct_peak": 88.0,
            "achieved_occupancy_pct": 62.0,
            "block_limit_factor": "shared_mem",
            "achieved_tflops": 565.0,
        },
    ],
}


def _write_ceilings(tmp_path: Path, data: dict | None = None) -> Path:
    cfg_dir = tmp_path / "perf-tune-report" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfg_dir / "sol-ceilings.yaml"
    yaml_path.write_text(yaml.safe_dump(data if data is not None else _MIN_CEILINGS))
    return yaml_path


def _mk_atlas(campaign_dir: Path, cell_id: str, *, hardware: str = "B200") -> Path:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    cell = AtlasCell(
        cell_id=cell_id,
        model="glm-5.1",
        hardware=hardware,
        quant="NVFP4",
        tensor_parallel=8,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=12288,
        concurrency=192,
        status=STATUS_FULL,
        ttft_avg_ms=120.0,
        request_throughput_avg=0.7,
        output_tps_per_user=21.0,
        output_tps_per_gpu=494.0,
        backend=BACKEND_VLLM_SWEEP,
        raw_path="/dev/null",
        captured_at="2026-05-27T13:53:07Z",
        notes="scatter test",
        extra={},
    )
    atlas_path = campaign_dir / "atlas.jsonl"
    atlas_path.write_text(json.dumps(dataclasses.asdict(cell)) + "\n")
    return atlas_path


def _write_ncu_kernels(campaign_dir: Path, cell_id: str, payload: dict | None = None) -> Path:
    cell_dir = campaign_dir / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / "ncu_kernels.json"
    path.write_text(json.dumps(payload if payload is not None else _NCU_PAYLOAD))
    return path


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    return pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type /Page ")


# ---------------------------------------------------------------------------
# A. compute helpers
# ---------------------------------------------------------------------------


def test_peak_compute_picks_nvfp4_first():
    val, src = _peak_compute_tflops(_MIN_CEILINGS["b200_sm100"])
    assert src == "nvfp4_dense_pflops"
    assert val == 9000.0  # 9 PFLOPS -> 9000 TFLOPS


def test_peak_compute_falls_back_to_fp8_then_bf16():
    hw = {
        "fp8_dense_pflops": {"value": 4.5, "units": "PFLOPS"},
        "bf16_dense_pflops": {"value": 2.25, "units": "PFLOPS"},
    }
    val, src = _peak_compute_tflops(hw)
    assert src == "fp8_dense_pflops"
    assert val == 4500.0


def test_peak_compute_fallback_when_no_keys():
    val, src = _peak_compute_tflops({})
    assert src == "fallback"
    assert val == 1000.0


def test_peak_bandwidth_picks_hbm3e_first():
    val, src = _peak_bandwidth_tbps(_MIN_CEILINGS["b200_sm100"])
    assert src == "hbm3e_tbps"
    assert val == 8.0


def test_peak_bandwidth_falls_back_to_hbm3():
    hw = {"hbm3_tbps": {"value": 3.35}}
    val, src = _peak_bandwidth_tbps(hw)
    assert src == "hbm3_tbps"
    assert val == 3.35


# ---------------------------------------------------------------------------
# B. render_page contract
# ---------------------------------------------------------------------------


def test_render_page_raises_when_empty(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        with pytest.raises(ValueError, match="cell_ncu is empty"):
            sol_roofline_scatter.render_page(fig, OrderedDict(), _MIN_CEILINGS, "b200_sm100")
    finally:
        plt.close(fig)


def test_render_page_raises_when_hw_key_missing(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        with pytest.raises(ValueError, match="not in ceilings"):
            sol_roofline_scatter.render_page(
                fig,
                OrderedDict([("c1", _NCU_PAYLOAD)]),
                _MIN_CEILINGS,
                "nonexistent_hw_key",
            )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# C. discover_ncu_payloads
# ---------------------------------------------------------------------------


def test_discover_ncu_payloads_empty_when_no_cells(tmp_path):
    assert len(discover_ncu_payloads(tmp_path)) == 0


def test_discover_ncu_payloads_finds_one(tmp_path):
    _write_ncu_kernels(tmp_path, "cell-1")
    payloads = discover_ncu_payloads(tmp_path)
    assert "cell-1" in payloads
    assert payloads["cell-1"]["hw_key"] == "b200_sm100"


def test_discover_ncu_payloads_missing_required_fields_raises(tmp_path):
    cell_dir = tmp_path / "cells" / "cell-bad"
    cell_dir.mkdir(parents=True)
    (cell_dir / "ncu_kernels.json").write_text('{"schema_version": 1}')

    with pytest.raises(NcuKernelsJsonMalformed) as ei:
        discover_ncu_payloads(tmp_path)
    assert "missing required fields" in ei.value.reason


def test_discover_ncu_payloads_invalid_json_raises(tmp_path):
    cell_dir = tmp_path / "cells" / "cell-bad"
    cell_dir.mkdir(parents=True)
    (cell_dir / "ncu_kernels.json").write_text("{not json")

    with pytest.raises(NcuKernelsJsonMalformed) as ei:
        discover_ncu_payloads(tmp_path)
    assert "not valid JSON" in ei.value.reason


# ---------------------------------------------------------------------------
# D. End-to-end via render_report
# ---------------------------------------------------------------------------


def test_renderer_page5_drawn_with_ncu_and_yaml(tmp_path, monkeypatch):
    """Both ncu_kernels.json + sol-ceilings.yaml present -> page 5 drawn."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    _write_ncu_kernels(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="page-5 test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # Page 1 (scatter grid), 2 (heatmap), 5 (ncu scatter) at minimum.
    # Page 3 + 4 are skipped because there's no kernels.json (zymtrace).
    assert page_count >= 3, f"expected >=3 pages, got {page_count}"


def test_renderer_page5_skipped_without_yaml(tmp_path, monkeypatch):
    """No sol-ceilings.yaml -> page 5 silently skipped."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    _write_ncu_kernels(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="no-yaml test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # 1 + 2 + 2b (TPM) + completeness page (page 5 + others omitted) = 4
    assert page_count == 4, f"expected exactly 4 pages, got {page_count}"


def test_renderer_page5_skipped_when_no_ncu(tmp_path, monkeypatch):
    """No ncu_kernels.json -> page 5 silently skipped (only pages 1+2)."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    campaign.mkdir(parents=True)
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="no-ncu test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # 1 + 2 + 2b (TPM) + completeness page (page 5 + others omitted) = 4
    assert page_count == 4, f"expected exactly 4 pages, got {page_count}"


def test_renderer_malformed_ncu_kernels_raises(tmp_path, monkeypatch):
    """Malformed ncu_kernels.json -> NcuKernelsJsonMalformed."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    cell_dir = campaign / "cells" / "cell-1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "ncu_kernels.json").write_text('{"schema_version": 1}')  # missing fields
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    with pytest.raises(NcuKernelsJsonMalformed):
        render_report(atlas, out_pdf, title="malformed ncu test")


def test_renderer_uses_payload_hw_key_when_atlas_mismatches(tmp_path, monkeypatch):
    """ncu_kernels.json hw_key overrides atlas-derived key for the scatter ceiling."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    # Add gb300 column so override has somewhere to land
    ceilings = dict(_MIN_CEILINGS)
    ceilings["gb300_nvl72"] = {
        "hw_name": "GB300 test",
        "nvfp4_dense_pflops": {"value": 15.0, "units": "PFLOPS"},
        "hbm3e_tbps": {"value": 8.0, "units": "TB/s"},
    }
    _write_ceilings(tmp_path, data=ceilings)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    payload_gb300 = dict(_NCU_PAYLOAD)
    payload_gb300["hw_key"] = "gb300_nvl72"
    _write_ncu_kernels(campaign, "cell-1", payload=payload_gb300)
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")  # atlas says B200

    out_pdf = tmp_path / "out.pdf"
    # Should not raise; the renderer prefers payload hw_key.
    render_report(atlas, out_pdf, title="hw-key-override test")
    assert out_pdf.is_file()


def test_renderer_ncu_only_zero_atlas_rows_renders_page5_l4(tmp_path, monkeypatch):
    """A pure-ncu roofline campaign with a 0-row bench atlas (no serve sweep)
    still renders page 5 and records sol_rigor=L4.

    Regression for the residual where ``hardware_key_for_atlas([])`` returns
    None on an empty atlas, so page 5 was omitted and ``sol_rigor`` recorded
    ``none`` even though ``ncu_kernels.json`` carries its own ``hw_key``. The
    fix falls back to the ncu payload's hw_key to resolve the ceilings.
    """
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    _write_ncu_kernels(campaign, "cell-1")  # payload hw_key=b200_sm100
    # 0-row atlas: a pure-ncu campaign never ran a serve sweep, so the atlas
    # has no rows -> no atlas-derived hardware key.
    campaign.mkdir(parents=True, exist_ok=True)
    atlas = campaign / "atlas.jsonl"
    atlas.write_text("")

    out_pdf = tmp_path / "out.pdf"
    status = render_report(atlas, out_pdf, title="ncu-only L4 test")

    assert status.sol_rigor == "L4"
    assert status.sol_complete is True
    assert any("page 5" in p for p in status.rendered_pages)
    # page 5 must NOT be in the omitted list.
    assert not any("page 5" in o["page"] for o in status.omitted_pages)


def test_renderer_full_stack_5_pages_with_zymtrace_and_ncu(tmp_path, monkeypatch):
    """kernels.json + ncu_kernels.json + yaml all present -> 5 pages."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    # Both zymtrace kernels.json AND ncu_kernels.json under same cell dir.
    cell_dir = campaign / "cells" / "cell-1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "kernels.json").write_text(json.dumps({
        "schema_version": 1,
        "captured_sources": ["zymtrace"],
        "top_kernels": [
            {"name": "multimem_all_reduce_kernel", "samples": 1000, "category": "NCCL"}
        ],
        "per_gpu": [{"gpu_name": "B200", "gpu_uuid": "u1", "samples": 1000}],
        "per_category": {"NCCL": 500, "BMM-NVFP4": 500},
        "top_python_during_cuda": [{"frame": "f", "samples": 1}],
    }))
    (cell_dir / "ncu_kernels.json").write_text(json.dumps(_NCU_PAYLOAD))
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="full-stack 5-page test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # Pages 1 (scatter grid), 2 (heatmap), 3 (kernel breakdown), 4 (sol roofline),
    # 5 (ncu scatter)
    assert page_count >= 5, f"expected >=5 pages, got {page_count}"


# ---------------------------------------------------------------------------
# E. v1.23.2: empty-state messaging + DCGM-fallback scatter
# ---------------------------------------------------------------------------


_NCU_PAYLOAD_ALL_NULL = {
    "schema_version": 1,
    "captured_sources": ["ncu"],
    "hw_key": "b200_sm100",
    "kernels": [
        {
            "name": "triton_red_fused_2",
            "name_full": "triton_red_fused_2",
            "category": "Triton-fused",
            "kernel_time_ns": 89.504,
            "dram_bytes_total": None,
            "sm_flops_total": None,
            "arithmetic_intensity_flops_per_byte": None,
            "achieved_dram_pct_peak": None,
            "achieved_sm_pct_peak": None,
            "achieved_occupancy_pct": None,
            "block_limit_factor": None,
            "achieved_tflops": None,
        }
    ],
}


_DCGM_PAYLOAD_WITH_ATTRIBUTION = {
    "schema_version": 1,
    "hw_key": "b200_sm100",
    "n_gpus": 8,
    "duration_s": 1380.0,
    "per_category_attribution": [
        {
            "category": "FMHA",
            "bound": "bandwidth",
            "ceiling_metric": "hbm3e_tbps",
            "time_share_pct": 30.0,
            "attributed_bytes_total": 1.0e15,
            "attributed_flops_total": 5.0e15,
            "sol_pct_bw": 30.2,
            "sol_pct_compute": None,
        },
        {
            "category": "BMM-NVFP4",
            "bound": "compute",
            "ceiling_metric": "nvfp4_dense_pflops",
            "time_share_pct": 50.0,
            "attributed_bytes_total": 5.0e14,
            "attributed_flops_total": 1.0e17,
            "sol_pct_bw": None,
            "sol_pct_compute": 12.5,
        },
    ],
}


def test_render_page_dcgm_fallback_plots_when_ncu_all_null():
    """v1.23.2: when all ncu kernels have null AI/tflops, DCGM fallback fires."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        # Should NOT raise; should plot DCGM fallback points instead.
        sol_roofline_scatter.render_page(
            fig,
            OrderedDict([("c1", _NCU_PAYLOAD_ALL_NULL)]),
            _MIN_CEILINGS,
            "b200_sm100",
            cell_dcgm=OrderedDict([("c1", _DCGM_PAYLOAD_WITH_ATTRIBUTION)]),
        )
        # Find the scatter axis (gs[1, 0]) and confirm it has scatter collections.
        axes_with_scatter = [a for a in fig.axes if a.collections]
        assert axes_with_scatter, "expected at least one scatter on the page"
        # The DCGM fallback should produce 2 scatter points (one per category).
        total_points = sum(len(c.get_offsets()) for a in axes_with_scatter for c in a.collections)
        assert total_points >= 2, f"expected >=2 DCGM scatter points, got {total_points}"
    finally:
        plt.close(fig)


def test_render_page_empty_state_message_when_no_data():
    """v1.23.2: when neither ncu kernels NOR dcgm fallback have data, empty-state msg renders."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        sol_roofline_scatter.render_page(
            fig,
            OrderedDict([("c1", _NCU_PAYLOAD_ALL_NULL)]),
            _MIN_CEILINGS,
            "b200_sm100",
            cell_dcgm=None,
        )
        # Find any text containing the empty-state hint.
        all_text = []
        for ax in fig.axes:
            for child in ax.texts:
                all_text.append(child.get_text())
        joined = " ".join(all_text)
        assert "No roofline-ready measurements found" in joined
        assert "TODO-NCU-FULL-SET-RECAPTURE" in joined
    finally:
        plt.close(fig)


def test_render_page_dcgm_fallback_skipped_when_attribution_empty():
    """v1.23.2: dcgm payload without per_category_attribution -> empty-state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dcgm_no_attr = {**_DCGM_PAYLOAD_WITH_ATTRIBUTION, "per_category_attribution": []}
    fig = plt.figure(figsize=(8, 11))
    try:
        sol_roofline_scatter.render_page(
            fig,
            OrderedDict([("c1", _NCU_PAYLOAD_ALL_NULL)]),
            _MIN_CEILINGS,
            "b200_sm100",
            cell_dcgm=OrderedDict([("c1", dcgm_no_attr)]),
        )
        # Empty-state text should still appear because no DCGM points fired.
        all_text = []
        for ax in fig.axes:
            for child in ax.texts:
                all_text.append(child.get_text())
        joined = " ".join(all_text)
        assert "No roofline-ready measurements found" in joined
    finally:
        plt.close(fig)


_NCU_PAYLOAD_SOL_ONLY = {
    "schema_version": 1,
    "captured_sources": ["ncu"],
    "hw_key": "b200_sm100",
    "kernels": [
        {
            "name": "triton_red_fused_2",
            "name_full": "triton_red_fused_2",
            "category": "Triton-fused",
            "kernel_time_ns": 89470.0,
            "dram_bytes_total": None,
            "sm_flops_total": None,
            "arithmetic_intensity_flops_per_byte": None,  # AI unmeasured (basic set)
            "achieved_dram_pct_peak": 19.04,
            "achieved_sm_pct_peak": 74.60,  # SM throughput IS measured
            "achieved_occupancy_pct": 91.13,
            "block_limit_factor": None,
            "achieved_tflops": None,
        }
    ],
}


def test_render_page_sol_only_kernel_plotted_at_category_ceiling():
    """%SoL-only kernel (AI null, SM% measured) plots at SM% x category ceiling.

    Triton-fused -> bf16_dense_pflops (2.25 PFLOPS = 2250 TFLOPS). At 74.6%
    that is ~1678.5 TFLOPS, parked at the ridge AI. No DCGM fallback, no
    empty-state message, AI never fabricated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        status = sol_roofline_scatter.render_page(
            fig,
            OrderedDict([("c1", _NCU_PAYLOAD_SOL_ONLY)]),
            _MIN_CEILINGS,
            "b200_sm100",
            cell_dcgm=None,
        )
        # Returns a partial status flagging the AI-unmeasured limitation.
        assert status.partial is True
        assert status.reason == "ncu_scatter_solonly"
        axes_with_scatter = [a for a in fig.axes if a.collections]
        assert axes_with_scatter, "expected the SoL-only point to be plotted"
        # Exactly one point, at y ~= 0.746 * 2250 = 1678.5 TFLOPS.
        ys = [
            float(c.get_offsets()[0][1])
            for a in axes_with_scatter
            for c in a.collections
            if len(c.get_offsets())
        ]
        assert any(abs(y - 1678.5) < 1.0 for y in ys), f"expected ~1678.5 TFLOPS, got {ys}"
        all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        # Honest dynamic title + loud banner with how-to-fix.
        assert "PARTIAL: arithmetic intensity UNMEASURED" in all_text
        assert "(ncu byte+FLOP measured)" not in all_text
        assert "WARNING -- PARTIAL ROOFLINE" in all_text
        assert "HOW TO FIX" in all_text
        assert "No roofline-ready measurements found" not in all_text
    finally:
        plt.close(fig)


def test_render_page_status_full_for_real_ncu():
    """Real ncu AI points -> partial=False + the 'byte+FLOP measured' title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        status = sol_roofline_scatter.render_page(
            fig, OrderedDict([("c1", _NCU_PAYLOAD)]), _MIN_CEILINGS, "b200_sm100"
        )
        assert status.partial is False
        assert status.reason == ""
        all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "(ncu byte+FLOP measured)" in all_text
        assert "PARTIAL" not in all_text
    finally:
        plt.close(fig)


def test_render_page_status_empty_is_partial():
    """All-null ncu + no DCGM -> partial=True, reason ncu_scatter_empty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        status = sol_roofline_scatter.render_page(
            fig, OrderedDict([("c1", _NCU_PAYLOAD_ALL_NULL)]), _MIN_CEILINGS,
            "b200_sm100", cell_dcgm=None,
        )
        assert status.partial is True
        assert status.reason == "ncu_scatter_empty"
    finally:
        plt.close(fig)


def test_render_page_status_dcgm_fallback_not_partial():
    """A measured DCGM workload-level fallback is NOT partial."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        status = sol_roofline_scatter.render_page(
            fig,
            OrderedDict([("c1", _NCU_PAYLOAD_ALL_NULL)]),
            _MIN_CEILINGS,
            "b200_sm100",
            cell_dcgm=OrderedDict([("c1", _DCGM_PAYLOAD_WITH_ATTRIBUTION)]),
        )
        assert status.partial is False
    finally:
        plt.close(fig)


def test_sol_only_helper_returns_none_for_bandwidth_category():
    """Bandwidth-bound categories have no compute ceiling for the SoL-only y."""
    assert sol_roofline_scatter._category_compute_ceiling_tflops(
        "FMHA", _MIN_CEILINGS, "b200_sm100"
    ) is None
    # Compute-bound Triton-fused -> bf16 2.25 PFLOPS -> 2250 TFLOPS.
    assert sol_roofline_scatter._category_compute_ceiling_tflops(
        "Triton-fused", _MIN_CEILINGS, "b200_sm100"
    ) == 2250.0


def test_report_status_records_partial_page5_solonly(tmp_path, monkeypatch):
    """End-to-end: a %SoL-only page 5 lands in report_status.json partial_pages
    (still in rendered_pages), and the completeness page renders."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    _write_ncu_kernels(campaign, "cell-1", payload=_NCU_PAYLOAD_SOL_ONLY)
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="partial page-5 test")

    status = json.loads((campaign / "report_status.json").read_text())
    assert "ncu SoL scatter (page 5)" in status["rendered_pages"]
    partial_titles = [p["page"] for p in status["partial_pages"]]
    assert any("page 5" in t for t in partial_titles)
    # Each partial entry carries why + how-to-fix.
    for p in status["partial_pages"]:
        assert p["why"] and p["how_to_fix"]
    # The PDF still renders (with the loud completeness page).
    assert out_pdf.is_file()


def test_render_page_real_ncu_kernels_skip_dcgm_fallback():
    """v1.23.2: when ncu kernels have real AI/tflops, DCGM fallback is NOT used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        sol_roofline_scatter.render_page(
            fig,
            OrderedDict([("c1", _NCU_PAYLOAD)]),  # has 2 real kernels
            _MIN_CEILINGS,
            "b200_sm100",
            cell_dcgm=OrderedDict([("c1", _DCGM_PAYLOAD_WITH_ATTRIBUTION)]),
        )
        # No "DCGM workload-level" caption should appear (real ncu data wins).
        all_text = []
        for ax in fig.axes:
            for child in ax.texts:
                all_text.append(child.get_text())
        joined = " ".join(all_text)
        assert "fallback: DCGM workload-level" not in joined
        assert "No roofline-ready measurements found" not in joined
    finally:
        plt.close(fig)
