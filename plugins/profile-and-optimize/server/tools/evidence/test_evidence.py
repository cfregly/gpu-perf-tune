"""Unit tests for evidence init."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.evidence.evidence_cli import CONTRACT, build_parser, main


SERVER_ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "/bin/bash"


def _seed_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "fake-repo-root"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\nversion = '0'\n")
    (root / "mcp_surface.py").write_text("# marker\n")
    (root / "tools").mkdir()
    skill = tmp_path / "skills" / "evidence-bundle-init" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Evidence bundle init\n")
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
    rc = main(
        [
            "init",
            "--family",
            "demo",
            "--intent",
            "smoke test for evidence-init",
            "--run-id",
            "test-bundle-001",
            "--json",
        ]
    )
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    bundle = Path(envelope["bundle_dir"])
    assert bundle.is_dir()
    assert (bundle / "SOURCE.md").is_file()
    assert (bundle / "summary.md").is_file()
    assert (bundle / "commands").is_dir()
    assert (bundle / "commands" / ".gitkeep").is_file()
    assert (bundle / "commands" / "README.md").is_file()
    commands_readme = (bundle / "commands" / "README.md").read_text()
    assert "tools/shared/capture_cmd.sh" in commands_readme
    assert "capture_rc=$?" in commands_readme
    assert "run()" not in commands_readme
    # SOURCE.md includes the intent and operator audit fields.
    src = (bundle / "SOURCE.md").read_text()
    assert "smoke test for evidence-init" in src
    assert "**Created by:**" in src
    skill_link = "../../../../../skills/evidence-bundle-init/SKILL.md"
    assert f"]({skill_link})" in src
    assert (bundle / skill_link).resolve().is_file()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--family", "/tmp/outside"),
        ("--family", "../outside"),
        ("--family", "demo/../../outside"),
        ("--family", "demo\\..\\outside"),
        ("--run-id", "/tmp/outside"),
        ("--run-id", "../outside"),
        ("--run-id", "nested/id"),
    ],
)
def test_init_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    arguments = [
        "init",
        "--family",
        "demo",
        "--intent",
        "path safety test",
        "--run-id",
        "safe-run",
        option,
        value,
        "--json",
    ]

    assert main(arguments) == 2
    assert "FATAL:" in capsys.readouterr().err
    assert not (tmp_path / "outside").exists()


def test_nested_family_keeps_generated_skill_link_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    assert (
        main(
            [
                "init",
                "--family",
                "campaign/llama31_8b",
                "--intent",
                "nested family test",
                "--run-id",
                "safe-run",
                "--json",
            ]
        )
        == 0
    )
    bundle = Path(json.loads(capsys.readouterr().out)["bundle_dir"])
    source = (bundle / "SOURCE.md").read_text()
    skill_link = "../../../../../../skills/evidence-bundle-init/SKILL.md"
    assert f"]({skill_link})" in source
    assert (bundle / skill_link).resolve().is_file()


def test_documented_capture_pattern_is_safe_under_errexit(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    helper = SERVER_ROOT / "tools" / "shared" / "capture_cmd.sh"
    script = (
        "set -e\n"
        "capture_rc=0\n"
        'ART_DIR="$1" bash "$2" deliberate-failure -- "$3" -c '
        "'printf out; printf err >&2; exit 7' || capture_rc=$?\n"
        'test "$capture_rc" -eq 7\n'
    )

    subprocess.run(
        [BASH, "-c", script, "capture-test", str(bundle), str(helper), BASH],
        check=True,
    )

    prefix = bundle / "commands" / "01-deliberate-failure"
    assert prefix.with_suffix(".stdout").read_text() == "out"
    assert prefix.with_suffix(".stderr").read_text() == "err"
    assert prefix.with_suffix(".exit").read_text() == "7\n"
    assert "exit\\ 7" in prefix.with_suffix(".cmd").read_text()


def test_init_refuses_to_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    main(
        [
            "init",
            "--family",
            "demo",
            "--intent",
            "first",
            "--run-id",
            "fixed-id",
            "--json",
        ]
    )
    capsys.readouterr()
    rc = main(
        [
            "init",
            "--family",
            "demo",
            "--intent",
            "second",
            "--run-id",
            "fixed-id",
            "--json",
        ]
    )
    assert rc == 2
    assert "already exists" in capsys.readouterr().err


def test_build_parser_help_does_not_crash() -> None:
    parser = build_parser()
    assert parser is not None
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
