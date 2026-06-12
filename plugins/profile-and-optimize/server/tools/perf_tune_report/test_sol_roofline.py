"""Tests for the Speed-of-Light (SoL) roofline renderer page.

Covers ``renderer.sol_roofline`` (the page-drawing module) and the
``discover_sol_inputs`` wiring in ``renderer.render_report``.

Scenarios:

A. load_ceilings_*       -- yaml loader happy path + 3 malformed shapes
B. hardware_key_*        -- atlas -> ceiling-yaml-key mapping
C. compute_category_sol_ -- per-category SoL row computation
D. discover_sol_inputs_  -- yaml discovery + SOL_CEILINGS_YAML env override
E. renderer_4page_*      -- end-to-end render with kernels.json + yaml
F. renderer_3page_*      -- end-to-end render with kernels.json but no yaml
G. renderer_malformed_*  -- malformed yaml raises SoLCeilingsMalformed

The fixture inputs (kernels.json payload + AtlasCell) mirror the shapes
``test_zymtrace_kernels.py`` already exercises, so this file stays
narrowly focused on the SoL-page-specific contract.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.renderer import sol_roofline
from tools.perf_tune_report.renderer.render_report import (
    discover_sol_inputs,
    render_report,
)
from tools.perf_tune_report.renderer.sol_roofline import (
    SoLCeilingsMalformed,
    compute_category_sol,
    hardware_key_for_atlas,
    load_ceilings,
)
from tools.perf_tune_report.schema import BACKEND_VLLM_SWEEP, STATUS_FULL, AtlasCell


# ---------------------------------------------------------------------------
# Fixtures
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
    "gb300_nvl72": {
        "hw_name": "NVIDIA GB300 test",
        "nvfp4_dense_pflops": {"value": 15.0, "units": "PFLOPS", "source": "test"},
        "hbm3e_tbps": {"value": 8.0, "units": "TB/s", "source": "test"},
        "nvlink5_tbps": {"value": 1.8, "units": "TB/s", "source": "test"},
        "bf16_dense_pflops": {"value": 3.75, "units": "PFLOPS", "source": "test"},
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


_KERNELS_PAYLOAD = {
    "schema_version": 1,
    "captured_sources": ["zymtrace"],
    "top_kernels": [
        {"name": "multimem_all_reduce_kernel", "samples": 558, "category": "NCCL"},
        {"name": "bmm_E2m1E2m1_Fp32_sm100f", "samples": 328, "category": "BMM-NVFP4"},
        {"name": "fmhaSm100fKernel_Persistent", "samples": 180, "category": "FMHA"},
    ],
    "per_gpu": [
        {"gpu_name": "NVIDIA B200", "gpu_uuid": "uuid-1", "samples": 35961},
    ],
    "per_category": {
        "cuBLAS": 119199,
        "Triton-fused": 61950,
        "NCCL": 16510,
        "BMM-NVFP4": 19212,
        "FMHA": 9226,
    },
    "top_python_during_cuda": [
        {"frame": "vllm.engine.AsyncLLMEngine._run_engine", "samples": 12345},
    ],
}


def _write_ceilings(tmp_path: Path, data: dict | None = None) -> Path:
    """Write the workspace-canonical YAML at ``<tmp>/configs/sol-ceilings.yaml``."""
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
        notes="sol roofline test",
        extra={},
    )
    atlas_path = campaign_dir / "atlas.jsonl"
    atlas_path.write_text(json.dumps(dataclasses.asdict(cell)) + "\n")
    return atlas_path


def _write_kernels_json(campaign_dir: Path, cell_id: str) -> Path:
    cell_dir = campaign_dir / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / "kernels.json"
    path.write_text(json.dumps(_KERNELS_PAYLOAD))
    return path


# ---------------------------------------------------------------------------
# A. load_ceilings
# ---------------------------------------------------------------------------


def test_load_ceilings_happy_path(tmp_path):
    yaml_path = _write_ceilings(tmp_path)
    data = load_ceilings(yaml_path)
    assert "b200_sm100" in data
    assert "category_ceiling_map" in data
    assert data["b200_sm100"]["hbm3e_tbps"]["value"] == 8.0


def test_load_ceilings_missing_category_map_raises(tmp_path):
    bad = {"b200_sm100": _MIN_CEILINGS["b200_sm100"]}
    yaml_path = _write_ceilings(tmp_path, data=bad)
    with pytest.raises(SoLCeilingsMalformed) as ei:
        load_ceilings(yaml_path)
    assert "category_ceiling_map" in ei.value.reason


def test_load_ceilings_invalid_yaml_raises(tmp_path):
    yaml_path = tmp_path / "perf-tune-report" / "configs" / "sol-ceilings.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("{this is :: not yaml at all")
    with pytest.raises(SoLCeilingsMalformed) as ei:
        load_ceilings(yaml_path)
    assert "not valid YAML" in ei.value.reason


def test_load_ceilings_top_level_not_mapping_raises(tmp_path):
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("- a\n- b\n")  # YAML list at top level
    with pytest.raises(SoLCeilingsMalformed) as ei:
        load_ceilings(yaml_path)
    assert "must be a mapping" in ei.value.reason


# ---------------------------------------------------------------------------
# B. hardware_key_for_atlas
# ---------------------------------------------------------------------------


def _row(hardware: str) -> AtlasCell:
    return AtlasCell(
        cell_id="c", model="m", hardware=hardware, quant="NVFP4",
        tensor_parallel=8, parallel_strategy="TP", mtp=False,
        max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
    )


def test_hardware_key_b200():
    assert hardware_key_for_atlas([_row("B200"), _row("B200")]) == "b200_sm100"


def test_hardware_key_gb300():
    assert hardware_key_for_atlas([_row("GB300")]) == "gb300_nvl72"


def test_hardware_key_h100():
    assert hardware_key_for_atlas([_row("H100")]) == "h100_sxm"


def test_hardware_key_unknown_returns_none():
    assert hardware_key_for_atlas([_row("MI300X")]) is None


def test_hardware_key_empty_rows_returns_none():
    assert hardware_key_for_atlas([]) is None


def test_hardware_key_picks_most_common():
    rows = [_row("B200"), _row("B200"), _row("H100")]
    assert hardware_key_for_atlas(rows) == "b200_sm100"


# ---------------------------------------------------------------------------
# C. compute_category_sol
# ---------------------------------------------------------------------------


def test_compute_category_sol_skips_zero_samples():
    cat_rows = compute_category_sol(
        _KERNELS_PAYLOAD,
        _MIN_CEILINGS["b200_sm100"],
        _MIN_CEILINGS["category_ceiling_map"],
    )
    cats = {r["category"] for r in cat_rows}
    # MoE / Elementwise / Other are absent from per_category -> not emitted
    assert "MoE" not in cats
    assert "Elementwise" not in cats
    assert {"NCCL", "BMM-NVFP4", "FMHA", "cuBLAS", "Triton-fused"}.issubset(cats)


def test_compute_category_sol_time_share_sums_to_100():
    cat_rows = compute_category_sol(
        _KERNELS_PAYLOAD,
        _MIN_CEILINGS["b200_sm100"],
        _MIN_CEILINGS["category_ceiling_map"],
    )
    total = sum(r["time_share_pct"] for r in cat_rows)
    assert abs(total - 100.0) < 0.01


def test_compute_category_sol_carries_ceiling_metadata():
    cat_rows = compute_category_sol(
        _KERNELS_PAYLOAD,
        _MIN_CEILINGS["b200_sm100"],
        _MIN_CEILINGS["category_ceiling_map"],
    )
    nccl_row = next(r for r in cat_rows if r["category"] == "NCCL")
    assert nccl_row["ceiling_metric"] == "nvlink5_tbps"
    assert nccl_row["ceiling_value"] == 1.8
    assert nccl_row["ceiling_units"] == "TB/s"
    assert nccl_row["bound"] == "bandwidth"
    bmm_row = next(r for r in cat_rows if r["category"] == "BMM-NVFP4")
    assert bmm_row["ceiling_metric"] == "nvfp4_dense_pflops"
    assert bmm_row["bound"] == "compute"


# ---------------------------------------------------------------------------
# D. discover_sol_inputs
# ---------------------------------------------------------------------------


def test_discover_sol_inputs_finds_yaml_in_parent(tmp_path, monkeypatch):
    """YAML at <tmp>/configs/sol-ceilings.yaml is discoverable
    from a campaign at <tmp>/perf-tune-report/campaigns/<x>/."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    campaign.mkdir(parents=True)
    rows = [_row("B200")]
    result = discover_sol_inputs(campaign, rows)
    assert result is not None
    ceilings, hw_key = result
    assert hw_key == "b200_sm100"
    assert ceilings["b200_sm100"]["hbm3e_tbps"]["value"] == 8.0


def test_discover_sol_inputs_returns_none_when_no_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    assert discover_sol_inputs(campaign, [_row("B200")]) is None


def test_discover_sol_inputs_returns_none_when_unknown_hardware(tmp_path, monkeypatch):
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    campaign.mkdir(parents=True)
    assert discover_sol_inputs(campaign, [_row("MI300X")]) is None


def test_discover_sol_inputs_env_disable(tmp_path, monkeypatch):
    _write_ceilings(tmp_path)
    monkeypatch.setenv("SOL_CEILINGS_YAML", "disable")
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    campaign.mkdir(parents=True)
    assert discover_sol_inputs(campaign, [_row("B200")]) is None


def test_discover_sol_inputs_env_override(tmp_path, monkeypatch):
    yaml_path = tmp_path / "elsewhere.yaml"
    yaml_path.write_text(yaml.safe_dump(_MIN_CEILINGS))
    monkeypatch.setenv("SOL_CEILINGS_YAML", str(yaml_path))
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    result = discover_sol_inputs(campaign, [_row("B200")])
    assert result is not None
    _, hw_key = result
    assert hw_key == "b200_sm100"


# ---------------------------------------------------------------------------
# E + F + G. End-to-end via render_report
# ---------------------------------------------------------------------------


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    return pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type /Page ")


def test_renderer_four_page_when_yaml_and_kernels_present(tmp_path, monkeypatch):
    """SoL page draws when both kernels.json + sol-ceilings.yaml exist."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    _write_kernels_json(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="4-page SoL test")

    assert out_pdf.is_file()
    page_count = _count_pdf_pages(out_pdf.read_bytes())
    assert page_count >= 4, f"expected 4+ pages, got {page_count}"


def test_renderer_three_page_when_yaml_missing(tmp_path, monkeypatch):
    """No sol-ceilings.yaml -> page 4 silently skipped, page 3 still drawn."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    _write_kernels_json(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="3-page no-SoL test")

    assert out_pdf.is_file()
    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # 1 + 2 + 2b (TPM) + page 3 (kernels) + completeness (omits SoL 4/5/6/6b) = 5
    assert page_count == 5, f"expected exactly 5 pages, got {page_count}"


def test_renderer_three_page_when_hardware_unknown(tmp_path, monkeypatch):
    """YAML present but hardware not in mapping -> page 4 silently skipped."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    _write_ceilings(tmp_path)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    _write_kernels_json(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1", hardware="MI300X")  # not in mapping

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="3-page unknown-hw test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # 1 + 2 + 2b (TPM) + page 3 (kernels) + completeness (omits SoL 4/5/6/6b) = 5
    assert page_count == 5, f"expected exactly 5 pages, got {page_count}"


def test_renderer_malformed_yaml_raises(tmp_path, monkeypatch):
    """Malformed YAML (missing category_ceiling_map) -> SoLCeilingsMalformed.

    Same no-silent-degradation rule as KernelsJsonMalformed.
    """
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    bad = {"b200_sm100": _MIN_CEILINGS["b200_sm100"]}
    _write_ceilings(tmp_path, data=bad)
    campaign = tmp_path / "perf-tune-report" / "campaigns" / "test-campaign"
    _write_kernels_json(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1", hardware="B200")

    out_pdf = tmp_path / "out.pdf"
    with pytest.raises(SoLCeilingsMalformed) as ei:
        render_report(atlas, out_pdf, title="malformed-sol test")
    assert "category_ceiling_map" in ei.value.reason


def test_render_page_raises_when_cell_kernels_empty(tmp_path):
    """sol_roofline.render_page contract: caller MUST pass non-empty."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        from collections import OrderedDict
        with pytest.raises(ValueError, match="cell_kernels is empty"):
            sol_roofline.render_page(
                fig,
                OrderedDict(),
                [],
                _MIN_CEILINGS,
                "b200_sm100",
            )
    finally:
        plt.close(fig)


def test_render_page_raises_when_hardware_key_missing():
    """Defense-in-depth: render_page rejects an HW key absent from ceilings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    fig = plt.figure(figsize=(8, 11))
    try:
        with pytest.raises(SoLCeilingsMalformed):
            sol_roofline.render_page(
                fig,
                OrderedDict([("cell-1", _KERNELS_PAYLOAD)]),
                [],
                _MIN_CEILINGS,
                "nonexistent_hw_key",
            )
    finally:
        plt.close(fig)
