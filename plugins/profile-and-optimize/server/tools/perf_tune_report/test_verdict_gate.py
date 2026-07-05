"""Tests for the verdict-rigor publish gate.

Enforces CLAUDE.md "Verdict rigor: DRAFT vs VERDICT": a campaign published with
``verdict.json`` ``tier=verdict`` must carry the controlled+metric+baseline
provenance, else the publish gate fails loud (CampaignIncompleteError). A
``draft`` tier (or absent verdict.json) is ungated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.lake_writer import (
    CampaignIncompleteError,
    VerdictSummary,
    build_campaign_row,
    publish,
    read_verdict,
    verdict_problems,
)
from tools.perf_tune_report.test_publish_to_lake import (
    _make_atlas_row,
    _stage_campaign,
    _stub_cfg,
)

pytest.importorskip("pyarrow")


def _write_verdict(campaign_dir: Path, **fields) -> None:
    (campaign_dir / "verdict.json").write_text(json.dumps(fields), encoding="utf-8")


# --- read_verdict + verdict_problems units --------------------------------


def test_read_verdict_absent_defaults_to_draft(tmp_path: Path):
    assert read_verdict(tmp_path).tier == "draft"


def test_verdict_problems_draft_is_empty():
    assert verdict_problems(VerdictSummary(tier="draft")) == []


def test_verdict_problems_verdict_incomplete_lists_each_gap():
    p = verdict_problems(VerdictSummary(tier="verdict"))
    joined = " ".join(p)
    assert ">=3 trials" in joined
    assert "same_node" in joined
    assert "baseline_named" in joined


def test_verdict_problems_latency_claim_requires_tpot_or_itl():
    v = VerdictSummary(
        tier="verdict", trials=3, same_node=True, baseline_named=True,
        latency_claim=True, decode_metric="throughput",
    )
    assert any("decode_metric in tpot|itl" in x for x in verdict_problems(v))
    v.decode_metric = "tpot"
    assert verdict_problems(v) == []


def test_verdict_problems_which_kernel_requires_per_kernel_ref():
    v = VerdictSummary(
        tier="verdict", trials=3, same_node=True, baseline_named=True,
        which_kernel_claim=True, per_kernel_ref=False,
    )
    assert any("per_kernel_ref" in x for x in verdict_problems(v))


# --- publish() gate -------------------------------------------------------


def test_publish_draft_is_ungated(tmp_path: Path):
    cd = _stage_campaign(tmp_path)  # complete report_status, no verdict.json
    publish(cd, cfg=_stub_cfg(), dry_run=True)  # must not raise


def test_publish_verdict_incomplete_lands_as_draft(tmp_path: Path):
    """Always-publish policy (v1.33.0): an unsupported verdict claim is
    auto-DOWNGRADED to draft + lands (not refused). --strict still raises."""
    cd = _stage_campaign(tmp_path)
    _write_verdict(cd, tier="verdict", trials=1, same_node=False, baseline_named=False)
    publish(cd, cfg=_stub_cfg(), dry_run=True)  # must not raise
    table = build_campaign_row(cd, [_make_atlas_row()])
    assert table.column("verdict_tier")[0].as_py() == "draft"  # downgraded


def test_publish_verdict_incomplete_strict_raises(tmp_path: Path):
    cd = _stage_campaign(tmp_path)
    _write_verdict(cd, tier="verdict", trials=1, same_node=False, baseline_named=False)
    with pytest.raises(CampaignIncompleteError, match="verdict_tier=verdict"):
        publish(cd, cfg=_stub_cfg(), dry_run=True, strict=True)


def test_publish_verdict_complete_passes(tmp_path: Path):
    cd = _stage_campaign(tmp_path)
    _write_verdict(
        cd, tier="verdict", trials=3, same_node=True, baseline_named=True,
        latency_claim=True, decode_metric="tpot",
    )
    publish(cd, cfg=_stub_cfg(), dry_run=True)  # must not raise


def test_publish_verdict_incomplete_allow_incomplete_lands_as_draft(tmp_path: Path):
    cd = _stage_campaign(tmp_path)
    _write_verdict(cd, tier="verdict", trials=1)
    publish(cd, cfg=_stub_cfg(), dry_run=True, allow_incomplete=True)  # override


def test_campaign_row_records_verdict_tier(tmp_path: Path):
    cd = _stage_campaign(tmp_path)
    _write_verdict(
        cd, tier="verdict", trials=3, same_node=True, baseline_named=True,
        latency_claim=True, decode_metric="tpot",
    )
    table = build_campaign_row(cd, [_make_atlas_row()])
    assert table.column("verdict_tier")[0].as_py() == "verdict"
