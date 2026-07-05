"""Tests for the pre-publish methodology + kernel-rubric (K/R/H/P/A) gates.

Enforces two workspace rules mechanically at publish time:

- CLAUDE.md "Benchmark methodology hygiene" -- every MEASURED atlas row must
  carry a warm/cold ``cache_mode`` + shape provenance. An unlabeled
  throughput/latency number is the warm-vs-cold comparability trap.
- CLAUDE.md "Custom-kernel work: classify before you climb" -- an L4
  kernel-comparison campaign must carry a ``krhpa:`` block classifying the
  candidate AND named baseline on (K,R,H,P,A).

Both mirror the verdict gate: under ``--strict`` they raise
``CampaignIncompleteError``; otherwise they record + warn (always-publish).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.lake_writer import (
    CampaignIncompleteError,
    RenderStatusSummary,
    close_reason_problems,
    krhpa_problems,
    methodology_problems,
    publish,
    read_close_reason,
)
from tools.perf_tune_report.test_publish_to_lake import (
    _FakeS3Client,
    _make_atlas_row,
    _stage_campaign,
    _stub_cfg,
    _write_status,
)

pytest.importorskip("pyarrow")


# --- methodology_problems units -------------------------------------------


def test_methodology_unknown_cache_mode_flagged():
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="unknown")]
    p = methodology_problems(rows)
    assert any("cache_mode=unknown" in x and "cellA" in x for x in p)


def test_methodology_cold_and_warm_ok():
    rows = [
        _make_atlas_row(cell_id="cellA", cache_mode="cold"),
        _make_atlas_row(cell_id="cellB", cache_mode="warm"),
    ]
    assert methodology_problems(rows) == []


# --- close_reason units (grind discipline, principle i) -------------------


def test_close_reason_empty_is_open_ok(tmp_path):
    """No close_reason = an OPEN investigation -> no problem (next_lever is enforced separately)."""
    (tmp_path / "config.yaml").write_text("focus: throughput\n", encoding="utf-8")
    assert close_reason_problems(tmp_path) == []
    assert read_close_reason(tmp_path) == ""


def test_close_reason_valid_ok(tmp_path):
    for reason in ("beat-target", "measured-plateau", "infra-wall"):
        (tmp_path / "config.yaml").write_text(f"close_reason: {reason}\n", encoding="utf-8")
        assert close_reason_problems(tmp_path) == [], reason
        assert read_close_reason(tmp_path) == reason


def test_close_reason_invalid_flagged(tmp_path):
    """Closing on a non-measured reason (first-principles/cost/temporary-plateau) is forbidden."""
    (tmp_path / "config.yaml").write_text("close_reason: vibes\n", encoding="utf-8")
    p = close_reason_problems(tmp_path)
    assert any("not a measured close-reason" in x for x in p)


def test_methodology_failed_cell_is_exempt():
    """A failed/evicted cell carries no measurement -> no warm/cold needed."""
    rows = [
        _make_atlas_row(
            cell_id="dead",
            status="failed",
            ttft_avg_ms=None,
            request_throughput_avg=None,
            output_tps_per_user=None,
            output_tps_per_gpu=None,
            cache_mode="unknown",
        )
    ]
    assert methodology_problems(rows) == []


def test_methodology_missing_shape_flagged():
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold", max_num_batched_tokens=0)]
    p = methodology_problems(rows)
    assert any("max_num_batched_tokens" in x for x in p)


def test_methodology_empty_rows_ok():
    assert methodology_problems([]) == []


# --- full-context descriptor gate (no bare numbers, principle j) ----------


def test_methodology_full_descriptor_ok():
    """A measured row with a complete descriptor (the _make_atlas_row default) passes."""
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold")]
    assert methodology_problems(rows) == []


@pytest.mark.parametrize("fld", ["dataset", "cudagraph_mode", "kv_cache_dtype", "image"])
def test_methodology_unknown_descriptor_str_field_flagged(fld):
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold", **{fld: "unknown"})]
    p = methodology_problems(rows)
    assert any(f"{fld}=unknown" in x and "cellA" in x for x in p)


def test_methodology_missing_gmu_flagged():
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold", gpu_memory_utilization=None)]
    p = methodology_problems(rows)
    assert any("gpu_memory_utilization" in x and "cellA" in x for x in p)


def test_methodology_missing_isl_osl_flagged():
    """A vllm-bench (non-aa) measured row must record per-request ISL/OSL -- the
    workload shape, not just max_num_batched_tokens (docs/METHODOLOGY.md)."""
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold",
                            mean_input_tokens=None, mean_output_tokens=None)]
    p = methodology_problems(rows)
    assert any("mean_input_tokens/mean_output_tokens" in x and "cellA" in x for x in p)


def test_methodology_isl_osl_exempt_aa_dataset():
    """aa-* workloads define their shape by the dataset name -> ISL/OSL exempt."""
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold", dataset="aa-10k",
                            mean_input_tokens=None, mean_output_tokens=None)]
    assert methodology_problems(rows) == []


def test_methodology_isl_osl_exempt_aiperf_backend():
    """aiperf / drive_load backends legitimately leave ISL/OSL None -> exempt."""
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold", bench_backend="drive_load",
                            mean_input_tokens=None, mean_output_tokens=None)]
    assert methodology_problems(rows) == []


def test_methodology_descriptor_gate_exempts_failed_cell():
    """A failed cell carries no measurement -> descriptor not required."""
    rows = [
        _make_atlas_row(
            cell_id="dead", status="failed",
            ttft_avg_ms=None, request_throughput_avg=None,
            output_tps_per_user=None, output_tps_per_gpu=None,
            cache_mode="unknown", dataset="unknown", cudagraph_mode="unknown",
            kv_cache_dtype="unknown", image="unknown", gpu_memory_utilization=None,
        )
    ]
    assert methodology_problems(rows) == []


# --- krhpa_problems units --------------------------------------------------


def _status(sol_rigor: str) -> RenderStatusSummary:
    return RenderStatusSummary(
        rendered=True, sol_complete=True, plot_ready_points=1, omitted_pages="",
        dcgm_grounded=True, sol_rigor=sol_rigor,
    )


_KRHPA_VALID = (
    "name: test\n"
    "next_lever: 'frontier-exhausted: H1 candidate cannot beat the H4 tensor-core baseline'\n"
    "krhpa:\n"
    "  candidate: {K: 4, R: 2, H: 1, P: 2, A: 1, name: 'warp-decode (Triton FMA)'}\n"
    "  baseline: {K: 4, R: 1, H: 4, P: 4, A: 1, name: 'FlashInfer-TRTLLM bmm_sm100f'}\n"
)


def test_krhpa_non_l4_is_exempt(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("name: test\n")  # no krhpa block
    assert krhpa_problems(tmp_path, _status("L1")) == []
    assert krhpa_problems(tmp_path, _status("none")) == []


def test_krhpa_l4_missing_block_flagged(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("name: test\n")
    p = krhpa_problems(tmp_path, _status("L4"))
    assert any("missing a krhpa: block" in x for x in p)


def test_krhpa_l4_exempt_reason_ok(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        "name: test\n"
        "krhpa_exempt_reason: engine/config profiling, not a custom-kernel comparison\n"
    )
    assert krhpa_problems(tmp_path, _status("L4")) == []


def test_krhpa_l4_malformed_flagged(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        "name: test\n"
        "krhpa:\n"
        "  candidate: {K: 9, R: 2, H: 1, P: 2, A: 1, name: 'x'}\n"  # K out of range
        "  baseline: {K: 4, R: 1, H: 4, P: 4, A: 1}\n"  # missing name
    )
    p = krhpa_problems(tmp_path, _status("L4"))
    joined = " ".join(p)
    assert "krhpa.candidate.K must be an int in 1..4" in joined
    assert "krhpa.baseline.name must be a non-empty string" in joined


def test_krhpa_l4_valid_ok(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(_KRHPA_VALID)
    assert krhpa_problems(tmp_path, _status("L4")) == []


# --- publish() gate integration -------------------------------------------


def _restage_atlas(campaign_dir: Path, rows) -> None:
    import json

    with (campaign_dir / "atlas.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def test_publish_strict_refuses_unlabeled_cache_mode(tmp_path: Path):
    """Default staged rows are cache_mode=unknown -> --strict refuses."""
    campaign_dir = _stage_campaign(tmp_path)  # default rows: cache_mode unknown
    with pytest.raises(CampaignIncompleteError, match="cache_mode=unknown"):
        publish(
            campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True,
            s3_client_factory=lambda _cfg: _FakeS3Client(),
        )


def test_publish_no_strict_records_unlabeled_and_lands(tmp_path: Path):
    """Always-publish: an unlabeled campaign LANDS under the library default
    (strict=False) with the gap recorded (visible on atlas_v1.cache_mode)."""
    campaign_dir = _stage_campaign(tmp_path)
    result = publish(
        campaign_dir, cfg=_stub_cfg(), dry_run=True,
        s3_client_factory=lambda _cfg: _FakeS3Client(),
    )
    assert result.campaign.row_count == 1


def test_publish_strict_passes_when_all_labeled(tmp_path: Path):
    """All rows cold + dcgm_grounded -> --strict publishes cleanly."""
    campaign_dir = _stage_campaign(tmp_path)
    _restage_atlas(campaign_dir, [
        _make_atlas_row(cell_id="cellA", cache_mode="cold"),
        _make_atlas_row(cell_id="cellB", cache_mode="cold"),
    ])
    _write_status(campaign_dir, dcgm_grounded=True)  # clear the hard DCGM gate
    publish(
        campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True,
        s3_client_factory=lambda _cfg: _FakeS3Client(),
    )  # must not raise


def test_publish_strict_refuses_l4_without_krhpa(tmp_path: Path):
    """An L4 kernel campaign with labeled rows but no krhpa: block is refused."""
    campaign_dir = _stage_campaign(tmp_path)
    _restage_atlas(campaign_dir, [_make_atlas_row(cell_id="cellA", cache_mode="cold")])
    _write_status(campaign_dir, sol_rigor="L4", dcgm_grounded=True)
    (campaign_dir / "config.yaml").write_text("name: test\n")  # no krhpa
    with pytest.raises(CampaignIncompleteError, match="krhpa"):
        publish(
            campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True,
            s3_client_factory=lambda _cfg: _FakeS3Client(),
        )


def test_publish_strict_passes_l4_with_krhpa(tmp_path: Path):
    campaign_dir = _stage_campaign(tmp_path)
    _restage_atlas(campaign_dir, [_make_atlas_row(cell_id="cellA", cache_mode="cold")])
    _write_status(campaign_dir, sol_rigor="L4", dcgm_grounded=True)
    (campaign_dir / "config.yaml").write_text(_KRHPA_VALID)
    publish(
        campaign_dir, cfg=_stub_cfg(), dry_run=True, strict=True,
        s3_client_factory=lambda _cfg: _FakeS3Client(),
    )  # must not raise
