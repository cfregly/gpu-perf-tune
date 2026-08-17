"""Regression tests for the packaged shell safety hooks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HOOKS = PACKAGE_ROOT / "hooks"
BASH = shutil.which("bash") or "/bin/bash"


def _run_script(
    script: Path,
    payload: str,
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    process = subprocess.run(
        [BASH, str(script), *(args or [])],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def _decision(verdict: dict[str, object]) -> str:
    output = verdict["hookSpecificOutput"]
    assert isinstance(output, dict)
    decision = output["permissionDecision"]
    assert isinstance(decision, str)
    return decision


def _guard(tmp_path: Path, output: str, *, exit_code: int = 0) -> Path:
    guard = tmp_path / "guard.sh"
    guard.write_text(
        f"#!/bin/bash\nprintf '%s\\n' {output!r}\nexit {exit_code}\n",
        encoding="utf-8",
    )
    return guard


def test_claude_adapter_round_trips_valid_permission(tmp_path: Path) -> None:
    guard = _guard(tmp_path, '{"permission":"ask","agent_message":"check first"}')
    verdict = _run_script(
        HOOKS / "claude-hook-adapter.sh",
        json.dumps(
            {
                "tool_input": {"command": "git commit -m test"},
                "cwd": str(tmp_path),
                "hook_event_name": "PreToolUse",
            }
        ),
        args=[str(guard)],
    )
    assert _decision(verdict) == "ask"


def test_claude_adapter_denies_invalid_input(tmp_path: Path) -> None:
    guard = _guard(tmp_path, '{"permission":"allow"}')
    verdict = _run_script(
        HOOKS / "claude-hook-adapter.sh",
        "not-json",
        args=[str(guard)],
    )
    assert _decision(verdict) == "deny"


def test_claude_adapter_denies_when_jq_is_unavailable(tmp_path: Path) -> None:
    guard = _guard(tmp_path, '{"permission":"allow"}')
    env = dict(os.environ)
    env["PATH"] = str(tmp_path / "empty-bin")
    verdict = _run_script(
        HOOKS / "claude-hook-adapter.sh",
        json.dumps({"tool_input": {"command": "echo test"}}),
        args=[str(guard)],
        env=env,
    )
    assert _decision(verdict) == "deny"


def test_claude_adapter_allows_disabled_provenance_gate_without_jq(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["PATH"] = str(tmp_path / "empty-bin")
    env.pop("PROVENANCE_COMMIT_GATE", None)
    verdict = _run_script(
        HOOKS / "claude-hook-adapter.sh",
        "input parsing is unnecessary while the registered guard is off",
        args=[str(HOOKS / "provenance-commit-gate.sh")],
        env=env,
    )
    assert _decision(verdict) == "allow"


def test_claude_adapter_denies_malformed_or_unknown_guard_verdict(tmp_path: Path) -> None:
    for output in ("not-json", "{}", '{"permission":"unknown"}'):
        guard = _guard(tmp_path, output)
        verdict = _run_script(
            HOOKS / "claude-hook-adapter.sh",
            json.dumps({"tool_input": {"command": "echo test"}}),
            args=[str(guard)],
        )
        assert _decision(verdict) == "deny"


def _git_repo(tmp_path: Path, source_text: str) -> Path:
    repo = tmp_path / "repo"
    source = repo / "experiments" / "run" / "SOURCE.md"
    source.parent.mkdir(parents=True)
    source.write_text(source_text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", str(source)], check=True)
    return repo


_VALID_PROVENANCE = """# Source

```provenance
schema: experiment_provenance_v1
identity:
  run_id: run
source:
  - repo: example/vllm
    commit: deadbeef
```
"""


def _provenance_env(mode: str, **overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PROVENANCE_COMMIT_GATE": mode,
            "PROFILE_AND_OPTIMIZE_PYTHON": sys.executable,
            **overrides,
        }
    )
    return env


def test_provenance_gate_denies_malformed_hook_input_when_enabled() -> None:
    verdict = _run_script(
        HOOKS / "provenance-commit-gate.sh",
        "not-json",
        env=_provenance_env("deny"),
    )
    assert verdict["permission"] == "deny"


def test_provenance_gate_uses_payload_checkout_and_fails_closed(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, "# missing provenance\n")
    verdict = _run_script(
        HOOKS / "provenance-commit-gate.sh",
        json.dumps({"command": "git commit -m test", "cwd": str(repo)}),
        env=_provenance_env("deny"),
    )
    assert verdict["permission"] == "deny"


def test_provenance_gate_accepts_valid_staged_block_from_explicit_checkout(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path, _VALID_PROVENANCE)
    verdict = _run_script(
        HOOKS / "provenance-commit-gate.sh",
        json.dumps({"command": "git commit -m test", "cwd": str(tmp_path)}),
        env=_provenance_env("deny", PROVENANCE_REPO_ROOT=str(repo)),
    )
    assert verdict["permission"] == "allow"


def test_provenance_gate_validates_staged_blob_not_worktree(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, "# staged file has no provenance block\n")
    source = repo / "experiments" / "run" / "SOURCE.md"
    source.write_text(_VALID_PROVENANCE, encoding="utf-8")

    verdict = _run_script(
        HOOKS / "provenance-commit-gate.sh",
        json.dumps({"command": "git commit -m test", "cwd": str(repo)}),
        env=_provenance_env("deny"),
    )

    assert verdict["permission"] == "deny"


def test_provenance_gate_ignores_unstaged_invalid_edit(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, _VALID_PROVENANCE)
    source = repo / "experiments" / "run" / "SOURCE.md"
    source.write_text("# unstaged file has no provenance block\n", encoding="utf-8")

    verdict = _run_script(
        HOOKS / "provenance-commit-gate.sh",
        json.dumps({"command": "git commit -m test", "cwd": str(repo)}),
        env=_provenance_env("deny"),
    )

    assert verdict["permission"] == "allow"
