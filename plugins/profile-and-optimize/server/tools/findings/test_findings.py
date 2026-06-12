"""Unit tests for the findings library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from tools.findings import findings_cli as f


def _record_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = dict(
        findings_yaml=str(tmp_path / "findings.yaml"),
        id="test-id",
        severity="critical",
        source_skill="ib-bw-check",
        source_query="ib-bw-total",
        headline="test headline",
        recommended_action="do the thing",
        status="open",
        evidence_path="raw/W1-ib-bw-total.json",
        affected_entity=["zone=<ZONE>"],
        notes="test note",
        json=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_record_creates_yaml(tmp_path: Path) -> None:
    args = _record_args(tmp_path)
    rc = f._record(args)
    assert rc == 0
    data = yaml.safe_load((tmp_path / "findings.yaml").read_text())
    assert len(data["findings"]) == 1
    finding = data["findings"][0]
    assert finding["id"] == "test-id"
    assert finding["severity"] == "critical"
    assert finding["affected_entities"] == [{"kind": "zone", "value": "<ZONE>"}]
    assert "detected_at_utc" in finding


def test_record_replaces_same_id(tmp_path: Path) -> None:
    f._record(_record_args(tmp_path, headline="first"))
    f._record(_record_args(tmp_path, headline="second"))
    data = yaml.safe_load((tmp_path / "findings.yaml").read_text())
    assert len(data["findings"]) == 1
    assert data["findings"][0]["headline"] == "second"


def test_record_validates_severity(tmp_path: Path, capsys) -> None:
    args = _record_args(tmp_path, severity="bogus")
    rc = f._record(args)
    assert rc != 0


def test_render_groups_by_severity(tmp_path: Path) -> None:
    f._record(_record_args(tmp_path, id="c1", severity="critical", headline="crit-1"))
    f._record(_record_args(tmp_path, id="h1", severity="high", headline="high-1"))
    f._record(_record_args(tmp_path, id="m1", severity="medium", headline="med-1"))

    args = argparse.Namespace(
        findings_yaml=str(tmp_path / "findings.yaml"),
        out=str(tmp_path / "findings.md"),
        json=False,
    )
    rc = f._render(args)
    assert rc == 0
    md = (tmp_path / "findings.md").read_text()
    assert "Critical" in md
    assert "C1" in md and "crit-1" in md
    assert "High" in md
    assert "H1" in md and "high-1" in md
    assert "Medium" in md
    assert "M1" in md and "med-1" in md


def test_diff_detects_new_resolved_and_status_changes(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    cur = tmp_path / "cur.yaml"
    f._record(_record_args(tmp_path, findings_yaml=str(base), id="a", severity="critical", headline="A"))
    f._record(_record_args(tmp_path, findings_yaml=str(base), id="b", severity="high", headline="B"))
    # current: drop b, add c, mark a as in_progress
    f._record(_record_args(tmp_path, findings_yaml=str(cur), id="a", severity="critical", headline="A", status="in_progress"))
    f._record(_record_args(tmp_path, findings_yaml=str(cur), id="c", severity="medium", headline="C"))

    args = argparse.Namespace(
        baseline=str(base),
        current=str(cur),
        out=str(tmp_path / "diff.md"),
        json=False,
    )
    rc = f._diff(args)
    assert rc == 0
    diff_md = (tmp_path / "diff.md").read_text()
    assert "**c**" in diff_md
    assert "**b**" in diff_md
    assert "open -> in_progress" in diff_md


def test_contract_has_3_verbs() -> None:
    assert set(f.CONTRACT.keys()) == {"record", "render", "diff"}
    assert f.CONTRACT["record"]["safety"] == "writes_artifacts"
    assert f.CONTRACT["render"]["safety"] == "read_only"
    assert f.CONTRACT["diff"]["safety"] == "read_only"
