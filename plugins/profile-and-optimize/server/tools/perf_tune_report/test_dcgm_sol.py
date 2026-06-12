"""Tests for the page-6 DCGM workload SoL renderer.

Scenarios:

A. render_page contract: empty -> ValueError
B. discover_dcgm_payloads: happy path + missing fields raises + invalid JSON raises
C. End-to-end via render_report:
   - dcgm_correlation.json present -> page 6 drawn
   - dcgm_correlation.json absent -> page 6 skipped silently
   - malformed dcgm_correlation.json -> DcgmCorrelationJsonMalformed
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.renderer import dcgm_category_attribution, dcgm_sol
from tools.perf_tune_report.renderer.render_report import (
    DcgmCorrelationJsonMalformed,
    discover_dcgm_payloads,
    render_report,
)
from tools.perf_tune_report.schema import BACKEND_VLLM_SWEEP, STATUS_FULL, AtlasCell


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_DCGM_PAYLOAD = {
    "schema_version": 1,
    "captured_sources": ["dcgm"],
    "hw_key": "b200_sm100",
    "sweep_start_utc": "2026-05-27T13:30:00Z",
    "sweep_end_utc": "2026-05-27T13:32:00Z",
    "duration_s": 120.0,
    "n_gpus": 8,
    "dcgm_group_level": "prof",
    "scrape_interval_s": 15,
    "short_sweep_warning": False,
    "resources": [
        {
            "peak_key": "hbm3e_tbps",
            "metric": "DCGM_FI_PROF_DRAM_ACTIVE",
            "is_fallback": False,
            "n_gpus": 8,
            "measured_bytes_total": 3.84e15,
            "measured_bytes_per_s": 4e12,
            "measured_tflops_avg": None,
            "peak_per_gpu": 8.0,
            "peak_per_gpu_units": "TB/s",
            "peak_aggregate": 64.0,
            "sol_pct": 50.0,
            "notes": [],
        },
        {
            "peak_key": "nvlink5_tbps",
            "metric": "DCGM_FI_PROF_NVLINK_TX_BYTES + DCGM_FI_PROF_NVLINK_RX_BYTES",
            "is_fallback": False,
            "n_gpus": 8,
            "measured_bytes_total": 6e13,
            "measured_bytes_per_s": 0.5e12,
            "measured_tflops_avg": None,
            "peak_per_gpu": 1.8,
            "peak_per_gpu_units": "TB/s",
            "peak_aggregate": 14.4,
            "sol_pct": 27.8,
            "notes": [],
        },
        {
            "peak_key": "nvfp4_dense_pflops",
            "metric": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
            "is_fallback": False,
            "n_gpus": 8,
            "measured_bytes_total": None,
            "measured_bytes_per_s": None,
            "measured_tflops_avg": 1800.0,
            "peak_per_gpu": 9.0,
            "peak_per_gpu_units": "PFLOPS",
            "peak_aggregate": 72.0,
            "sol_pct": 20.0,
            "notes": [],
        },
    ],
    "queries": [
        {"peak_key": "hbm3e_tbps", "metric": "DCGM_FI_PROF_DRAM_ACTIVE",
         "promql": "avg by (gpu) (DCGM_FI_PROF_DRAM_ACTIVE{...})", "unit": "ratio", "is_fallback": False},
    ],
    "dry_run": False,
}


def _mk_atlas(campaign_dir: Path, cell_id: str) -> Path:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    cell = AtlasCell(
        cell_id=cell_id,
        model="glm-5.1",
        hardware="B200",
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
        notes="dcgm test",
        extra={},
    )
    atlas_path = campaign_dir / "atlas.jsonl"
    atlas_path.write_text(json.dumps(dataclasses.asdict(cell)) + "\n")
    return atlas_path


def _write_dcgm(campaign_dir: Path, cell_id: str, payload: dict | None = None) -> Path:
    cell_dir = campaign_dir / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / "dcgm_correlation.json"
    path.write_text(json.dumps(payload if payload is not None else _DCGM_PAYLOAD))
    return path


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    return pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type /Page ")


# ---------------------------------------------------------------------------
# A. render_page contract
# ---------------------------------------------------------------------------


def test_render_page_raises_when_empty():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        with pytest.raises(ValueError, match="cell_dcgm is empty"):
            dcgm_sol.render_page(fig, OrderedDict())
    finally:
        plt.close(fig)


def test_render_page_handles_no_resources():
    """Payload exists but resources list is empty -> draws header + 'no resources' message."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        empty = dict(_DCGM_PAYLOAD)
        empty["resources"] = []
        # Should NOT raise
        dcgm_sol.render_page(fig, OrderedDict([("c1", empty)]))
    finally:
        plt.close(fig)


def test_render_page_handles_counter_fallback_caveat():
    """group_level=counter triggers caveat text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        counter = dict(_DCGM_PAYLOAD)
        counter["dcgm_group_level"] = "counter"
        dcgm_sol.render_page(fig, OrderedDict([("c1", counter)]))
    finally:
        plt.close(fig)


def test_render_page_handles_short_sweep_caveat():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        short = dict(_DCGM_PAYLOAD)
        short["short_sweep_warning"] = True
        dcgm_sol.render_page(fig, OrderedDict([("c1", short)]))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# B. discover_dcgm_payloads
# ---------------------------------------------------------------------------


def test_discover_dcgm_payloads_empty_when_no_cells(tmp_path):
    assert len(discover_dcgm_payloads(tmp_path)) == 0


def test_discover_dcgm_payloads_finds_one(tmp_path):
    _write_dcgm(tmp_path, "cell-1")
    payloads = discover_dcgm_payloads(tmp_path)
    assert "cell-1" in payloads
    assert payloads["cell-1"]["hw_key"] == "b200_sm100"


def test_discover_dcgm_payloads_missing_fields_raises(tmp_path):
    cell_dir = tmp_path / "cells" / "cell-bad"
    cell_dir.mkdir(parents=True)
    (cell_dir / "dcgm_correlation.json").write_text('{"schema_version": 1}')
    with pytest.raises(DcgmCorrelationJsonMalformed) as ei:
        discover_dcgm_payloads(tmp_path)
    assert "missing required fields" in ei.value.reason


def test_discover_dcgm_payloads_invalid_json_raises(tmp_path):
    cell_dir = tmp_path / "cells" / "cell-bad"
    cell_dir.mkdir(parents=True)
    (cell_dir / "dcgm_correlation.json").write_text("{not json")
    with pytest.raises(DcgmCorrelationJsonMalformed) as ei:
        discover_dcgm_payloads(tmp_path)
    assert "not valid JSON" in ei.value.reason


# ---------------------------------------------------------------------------
# C. End-to-end via render_report
# ---------------------------------------------------------------------------


def test_renderer_page6_drawn_with_dcgm(tmp_path, monkeypatch):
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    _write_dcgm(campaign, "cell-1")
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="page-6 test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # Pages 1 + 2 + 2b (TPM) + page 6 (this one) + the completeness page that
    # records the omitted 3/4/5/6b (no kernels.json, no ncu, no attribution).
    assert page_count == 5, f"expected exactly 5 pages, got {page_count}"


def test_renderer_page6_skipped_without_dcgm(tmp_path, monkeypatch):
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="no-dcgm test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # Pages 1 + 2 + 2b (TPM) + the completeness page (all conditional omitted).
    assert page_count == 4, f"expected 4 pages, got {page_count}"


def test_renderer_malformed_dcgm_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    cell_dir = campaign / "cells" / "cell-1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "dcgm_correlation.json").write_text('{"schema_version": 1}')
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    with pytest.raises(DcgmCorrelationJsonMalformed):
        render_report(atlas, out_pdf, title="malformed dcgm test")


# ---------------------------------------------------------------------------
# D. dcgm_category_attribution page (Phase B3 / B4)
# ---------------------------------------------------------------------------


_ATTRIBUTION_ROWS = [
    {
        "category": "NCCL",
        "time_share_pct": 30.0,
        "attributed_bytes_total": 2.6e14,
        "attributed_flops_total": None,
        "effective_bw_during_category_window": 0.9e12,
        "effective_tflops_during_category_window": None,
        "sol_pct_bw": 50.0,
        "sol_pct_compute": None,
        "bound": "bandwidth",
        "ceiling_metric": "nvlink5_tbps",
    },
    {
        "category": "BMM-NVFP4",
        "time_share_pct": 15.0,
        "attributed_bytes_total": 1.3e14,
        "attributed_flops_total": 2.16e17,
        "effective_bw_during_category_window": None,
        "effective_tflops_during_category_window": 1800.0,
        "sol_pct_bw": None,
        "sol_pct_compute": 20.0,
        "bound": "compute",
        "ceiling_metric": "nvfp4_dense_pflops",
    },
    {
        "category": "FMHA",
        "time_share_pct": 25.0,
        "attributed_bytes_total": 0.96e15,
        "attributed_flops_total": None,
        "effective_bw_during_category_window": 4e12,
        "effective_tflops_during_category_window": None,
        "sol_pct_bw": 50.0,
        "sol_pct_compute": None,
        "bound": "bandwidth",
        "ceiling_metric": "hbm3e_tbps",
    },
]


def _attribution_payload() -> dict:
    p = dict(_DCGM_PAYLOAD)
    p["per_category_attribution"] = list(_ATTRIBUTION_ROWS)
    return p


def test_category_attribution_render_page_raises_when_empty_dict():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        with pytest.raises(ValueError, match="cell_dcgm is empty"):
            dcgm_category_attribution.render_page(fig, OrderedDict())
    finally:
        plt.close(fig)


def test_category_attribution_render_page_raises_when_no_attribution():
    """Payload exists but per_category_attribution is empty -> ValueError."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        empty = dict(_DCGM_PAYLOAD)
        empty["per_category_attribution"] = []
        with pytest.raises(ValueError, match="per_category_attribution is empty"):
            dcgm_category_attribution.render_page(fig, OrderedDict([("c1", empty)]))
    finally:
        plt.close(fig)


def test_category_attribution_render_page_happy_path():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 11))
    try:
        dcgm_category_attribution.render_page(
            fig, OrderedDict([("c1", _attribution_payload())])
        )
    finally:
        plt.close(fig)


def test_category_attribution_render_handles_unmapped_category():
    """Unknown category still renders (appended after canonical order)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _attribution_payload()
    payload["per_category_attribution"] = list(payload["per_category_attribution"]) + [{
        "category": "ExoticUnknownCategory",
        "time_share_pct": 5.0,
        "attributed_bytes_total": None,
        "attributed_flops_total": None,
        "effective_bw_during_category_window": None,
        "effective_tflops_during_category_window": None,
        "sol_pct_bw": None,
        "sol_pct_compute": None,
        "bound": None,
        "ceiling_metric": None,
    }]

    fig = plt.figure(figsize=(8, 11))
    try:
        dcgm_category_attribution.render_page(fig, OrderedDict([("c1", payload)]))
    finally:
        plt.close(fig)


def test_renderer_page6b_drawn_with_attribution(tmp_path, monkeypatch):
    """When per_category_attribution is non-empty, page 6b is added (total = 4 pages)."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    _write_dcgm(campaign, "cell-1", payload=_attribution_payload())
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="page-6b test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # 1 + 2 + 2b (TPM) + page 6 + page 6b + completeness page (omits 3/4/5) = 6
    assert page_count == 6, f"expected 6 pages, got {page_count}"


def test_renderer_page6b_skipped_when_attribution_empty(tmp_path, monkeypatch):
    """per_category_attribution=[] -> only page 6 drawn (3 pages total)."""
    monkeypatch.delenv("SOL_CEILINGS_YAML", raising=False)
    campaign = tmp_path / "campaign"
    payload = dict(_DCGM_PAYLOAD)
    payload["per_category_attribution"] = []
    _write_dcgm(campaign, "cell-1", payload=payload)
    atlas = _mk_atlas(campaign, "cell-1")

    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="page-6b-skip test")

    page_count = _count_pdf_pages(out_pdf.read_bytes())
    # 1 + 2 + 2b (TPM) + page 6 + completeness page (omits 3/4/5/6b) = 5
    assert page_count == 5, f"expected 5 pages, got {page_count}"
