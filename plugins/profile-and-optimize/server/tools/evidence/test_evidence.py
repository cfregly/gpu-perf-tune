"""Unit tests for evidence init."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.evidence.evidence_cli import CONTRACT, build_parser, main


def _seed_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "fake-repo-root"
    root.mkdir()
    (root / "CLAUDE.md").write_text("# fake\n")
    (root / "tools").mkdir()
    return root


def test_contract_shape() -> None:
    spec = CONTRACT["init"]
    assert spec["safety"] == "writes_artifacts"
    assert spec["ack"] is None
    assert spec["json"] is True
    assert "--family" in spec["required"]
    assert "--intent" in spec["required"]


def test_init_creates_skeleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    rc = main([
        "init",
        "--family", "demo",
        "--intent", "smoke test for evidence-init",
        "--run-id", "test-bundle-001",
        "--json",
    ])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    bundle = Path(envelope["bundle_dir"])
    assert bundle.is_dir()
    assert (bundle / "SOURCE.md").is_file()
    assert (bundle / "summary.md").is_file()
    assert (bundle / "commands").is_dir()
    assert (bundle / "commands" / ".gitkeep").is_file()
    assert (bundle / "commands" / "README.md").is_file()
    # SOURCE.md includes the intent + team attribution.
    src = (bundle / "SOURCE.md").read_text()
    assert "smoke test for evidence-init" in src
    assert "the MLPerf team" in src


def test_init_refuses_to_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    main([
        "init", "--family", "demo", "--intent", "first",
        "--run-id", "fixed-id", "--json",
    ])
    capsys.readouterr()
    rc = main([
        "init", "--family", "demo", "--intent", "second",
        "--run-id", "fixed-id", "--json",
    ])
    assert rc == 2
    assert "already exists" in capsys.readouterr().err


def test_build_parser_help_does_not_crash() -> None:
    parser = build_parser()
    assert parser is not None
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
