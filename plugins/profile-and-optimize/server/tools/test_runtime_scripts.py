"""Offline regression tests for public runtime shell scripts."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BASH = shutil.which("bash") or "/bin/bash"


def _load_stage_model_parallel():
    script = (
        REPO_ROOT
        / "plugins"
        / "profile-and-optimize"
        / "server"
        / "tools"
        / "stage-model-parallel.py"
    )
    spec = importlib.util.spec_from_file_location("stage_model_parallel", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_model_parallel_contains_object_keys_under_destination(tmp_path: Path) -> None:
    module = _load_stage_model_parallel()
    destination = (tmp_path / "models").resolve()
    destination.mkdir()

    output = module._download_path(
        destination,
        "checkpoints/model",
        "checkpoints/model/layers/0.bin",
    )

    assert output == destination / "layers" / "0.bin"


def test_stage_model_parallel_rejects_nul_destination() -> None:
    module = _load_stage_model_parallel()

    with pytest.raises(SystemExit, match="without NUL bytes"):
        module._operator_destination("models\x00outside")


@pytest.mark.parametrize(
    "object_key",
    [
        "checkpoints/model/../outside.bin",
        "checkpoints/model/layers//0.bin",
        "other-prefix/outside.bin",
        "checkpoints/model/layers\\0.bin",
    ],
)
def test_stage_model_parallel_rejects_unsafe_object_keys(
    tmp_path: Path,
    object_key: str,
) -> None:
    module = _load_stage_model_parallel()
    destination = (tmp_path / "models").resolve()
    destination.mkdir()

    with pytest.raises(RuntimeError):
        module._download_path(destination, "checkpoints/model", object_key)


def test_stage_model_parallel_rejects_symlink_escape(tmp_path: Path) -> None:
    module = _load_stage_model_parallel()
    destination = (tmp_path / "models").resolve()
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="escapes destination root"):
        module._download_path(
            destination,
            "checkpoints/model",
            "checkpoints/model/linked/config.json",
        )


def test_plugin_validation_cache_invalidates_on_hook_change(tmp_path: Path) -> None:
    script_dir = tmp_path / "scripts"
    plugin_dir = tmp_path / "plugins" / "profile-and-optimize"
    skill_dir = plugin_dir / "skills" / "example"
    hooks_dir = plugin_dir / "hooks"
    marketplace_dir = tmp_path / ".claude-plugin"
    for directory in (script_dir, skill_dir, hooks_dir, marketplace_dir):
        directory.mkdir(parents=True, exist_ok=True)

    script = script_dir / "validate-cached.sh"
    shutil.copyfile(REPO_ROOT / "scripts" / "validate-cached.sh", script)
    (plugin_dir / ".claude-plugin").mkdir()
    (plugin_dir / ".claude-plugin" / "plugin.json").write_text("{}\n")
    (plugin_dir / ".mcp.json").write_text("{}\n")
    (marketplace_dir / "marketplace.json").write_text("{}\n")
    (skill_dir / "SKILL.md").write_text("---\nname: example\n---\n")
    hook = hooks_dir / "hooks.json"
    hook.write_text('{"hooks": []}\n')
    marker = tmp_path / "validator-runs"
    env = os.environ.copy()
    env["CLAUDE_PLUGIN_VALIDATE_CMD"] = f"printf 'run\\n' >> '{marker}'"

    subprocess.run([BASH, str(script)], env=env, check=True)
    subprocess.run([BASH, str(script)], env=env, check=True)
    assert marker.read_text().splitlines() == ["run"]

    hook.write_text('{"hooks": [{"event": "PreToolUse"}]}\n')
    subprocess.run([BASH, str(script)], env=env, check=True)
    assert marker.read_text().splitlines() == ["run", "run"]


def test_nsys_capture_paths_are_passed_as_argv(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "nsys-validate-capture.sh"
    marker = tmp_path / "injected"
    report = tmp_path / "capture'; touch injected; #.nsys-rep"
    with report.open("wb") as stream:
        stream.seek(1024 * 1024 - 1)
        stream.write(b"\0")

    fake_nsys = tmp_path / "fake nsys's"
    fake_nsys.write_text(
        "#!/usr/bin/env python3\n"
        "import sqlite3\n"
        "import sys\n"
        "output = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))\n"
        "connection = sqlite3.connect(output)\n"
        "connection.execute('create table CUPTI_ACTIVITY_KIND_KERNEL (value integer)')\n"
        "connection.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (1)')\n"
        "connection.commit()\n"
    )
    fake_nsys.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "LOCAL": "1",
            "MIN_REP_MB": "1",
            "NSYS": str(fake_nsys),
            "REP": str(report),
        }
    )

    result = subprocess.run(
        [BASH, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "KERNEL rows = 1" in result.stdout
    assert not marker.exists()
    assert "`docs/METHODOLOGY.md`" not in script.read_text()


def test_nsys_capture_rejects_non_numeric_size_before_arithmetic(
    tmp_path: Path,
) -> None:
    script = REPO_ROOT / "scripts" / "nsys-validate-capture.sh"
    marker = tmp_path / "arithmetic-injected"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"x")
    env = os.environ.copy()
    env.update(
        {
            "LOCAL": "1",
            "MIN_REP_MB": f"1+$(touch {marker})",
            "REP": str(report),
        }
    )

    result = subprocess.run(
        [BASH, str(script)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 1
    assert "MIN_REP_MB must be a positive integer" in result.stdout
    assert not marker.exists()


def test_profile_run_dry_run_uses_operator_launcher(tmp_path: Path) -> None:
    script = (
        REPO_ROOT
        / "plugins"
        / "profile-and-optimize"
        / "server"
        / "tools"
        / "pipeline"
        / "submission"
        / "profile"
        / "profile_run.sh"
    )
    artifact_root = tmp_path / "artifacts"
    marker = tmp_path / "launcher-ran"
    launcher = tmp_path / "sentinel-launcher"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "touch \"${PROFILE_RUN_SENTINEL:?}\"\n"
    )
    launcher.chmod(0o755)
    env = os.environ.copy()
    env["ART_ROOT"] = str(artifact_root)
    env["PROFILE_RUN_SENTINEL"] = str(marker)

    result = subprocess.run(
        [
            BASH,
            str(script),
            "--bench",
            "llama31_8b",
            "--run-id",
            "public-dry-run",
            "--nodes",
            "1",
            "--launcher",
            launcher,
            "--launcher-arg=--nodes",
            "--launcher-arg=two words",
            "--dry-run",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"launcher: {launcher} --nodes two\\ words" in result.stdout
    assert "tools/benchmarks" not in result.stdout + result.stderr
    assert not marker.exists()
    capture_log = (
        artifact_root
        / "campaign"
        / "llama31_8b"
        / "public-dry-run"
        / "profiling"
        / "capture.log"
    )
    assert capture_log.is_file()
    assert f"launcher={launcher} --nodes two\\ words" in capture_log.read_text()


def test_profile_run_requires_operator_launcher(tmp_path: Path) -> None:
    script = (
        REPO_ROOT
        / "plugins"
        / "profile-and-optimize"
        / "server"
        / "tools"
        / "pipeline"
        / "submission"
        / "profile"
        / "profile_run.sh"
    )
    result = subprocess.run(
        [
            BASH,
            str(script),
            "--bench",
            "llama31_8b",
            "--run-id",
            "missing-launcher",
            "--nodes",
            "1",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--launcher is required" in result.stderr
