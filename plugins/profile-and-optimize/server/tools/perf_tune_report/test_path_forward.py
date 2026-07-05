"""Tests for the path-forward / next_lever publish gate.

Enforces CLAUDE.md "Always ship an actionable path-forward" (the performance
ratchet / "Always be grinding"): every published campaign MUST declare a
``next_lever`` -- the specific next change to move a metric further, or an
explicit ``frontier-exhausted: <evidence>`` when a dimension is grounded at its
Speed-of-Light ceiling. Under ``--strict`` an absent next_lever raises
``CampaignIncompleteError``; otherwise it is recorded (next_lever="" on the
campaign_v1 row) + warned -- the same machinery as the SoL / verdict / krhpa /
source gates. This is the campaign-level analog of the per-finding ``next_lever``
the ``value_view`` GRIND FRONTIER already enforces.
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
    build_campaign_row,
    next_lever_problems,
    publish,
    read_next_lever,
)
from tools.perf_tune_report.test_publish_to_lake import (
    _FakeS3Client,
    _make_atlas_row,
    _stage_campaign,
    _stub_cfg,
    _write_status,
)

pytest.importorskip("pyarrow")

_NO_NEXT_LEVER = "name: test\ncells: []\n"


def _strip_next_lever(campaign_dir: Path) -> None:
    """Overwrite the staged config.yaml with one carrying no next_lever."""
    (campaign_dir / "config.yaml").write_text(_NO_NEXT_LEVER)


def _relabel_cold(campaign_dir: Path) -> None:
    """Relabel atlas rows cold so the methodology gate does not mask next_lever."""
    rows = [_make_atlas_row(cell_id="cellA", cache_mode="cold")]
    with (campaign_dir / "atlas.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


# --- read_next_lever + next_lever_problems units --------------------------


def test_read_next_lever_absent_is_empty(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(_NO_NEXT_LEVER)
    assert read_next_lever(tmp_path) == ""


def test_read_next_lever_from_config(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("name: t\nnext_lever: 'dispatch-depth (HYPOTHESIS)'\n")
    assert read_next_lever(tmp_path) == "dispatch-depth (HYPOTHESIS)"


def test_read_next_lever_from_source_md_fallback(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(_NO_NEXT_LEVER)
    (tmp_path / "SOURCE.md").write_text("# C\n\n- next_lever: from source\n")
    assert read_next_lever(tmp_path) == "from source"


def test_read_next_lever_malformed_config_does_not_crash(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("name: [unterminated\n")
    assert read_next_lever(tmp_path) == ""


def test_next_lever_problems_present_is_empty(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("name: t\nnext_lever: ship batching\n")
    assert next_lever_problems(tmp_path) == []


def test_next_lever_problems_frontier_exhausted_is_allowed(tmp_path: Path):
    """A genuinely-maxed dimension opts out via the non-empty escape value."""
    (tmp_path / "config.yaml").write_text(
        "name: t\nnext_lever: 'frontier-exhausted: HBM SoL 94% at c=256'\n"
    )
    assert next_lever_problems(tmp_path) == []


def test_next_lever_problems_absent_flags(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(_NO_NEXT_LEVER)
    p = next_lever_problems(tmp_path)
    assert p and "next_lever" in p[0]


# --- publish() gate -------------------------------------------------------


def test_publish_with_next_lever_lands(tmp_path: Path):
    """_stage_campaign declares a next_lever -> publish lands."""
    cd = _stage_campaign(tmp_path)
    publish(cd, cfg=_stub_cfg(), dry_run=True)  # must not raise


def test_publish_strict_refuses_without_next_lever(tmp_path: Path):
    """Isolated: all other gates cleared, ONLY next_lever absent -> --strict refuses."""
    cd = _stage_campaign(tmp_path)
    _relabel_cold(cd)
    _write_status(cd, dcgm_grounded=True)  # clear the methodology + DCGM gates
    _strip_next_lever(cd)
    with pytest.raises(CampaignIncompleteError, match="next_lever"):
        publish(
            cd, cfg=_stub_cfg(), dry_run=True, strict=True,
            s3_client_factory=lambda _cfg: _FakeS3Client(),
        )


def test_publish_no_strict_records_missing_next_lever_and_lands(tmp_path: Path):
    """Always-publish: a campaign with no next_lever LANDS under the default
    (strict=False) with the gap recorded (next_lever='' on the campaign_v1 row)."""
    cd = _stage_campaign(tmp_path)
    _strip_next_lever(cd)
    result = publish(
        cd, cfg=_stub_cfg(), dry_run=True,
        s3_client_factory=lambda _cfg: _FakeS3Client(),
    )
    assert result.campaign.row_count == 1


def test_campaign_row_records_next_lever(tmp_path: Path):
    cd = _stage_campaign(tmp_path)
    table = build_campaign_row(cd, [_make_atlas_row()])
    assert "dispatch-depth" in table.column("next_lever")[0].as_py()


def test_campaign_row_next_lever_empty_when_absent(tmp_path: Path):
    cd = _stage_campaign(tmp_path)
    _strip_next_lever(cd)
    table = build_campaign_row(cd, [_make_atlas_row()])
    assert table.column("next_lever")[0].as_py() == ""
