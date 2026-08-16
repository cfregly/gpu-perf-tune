#!/usr/bin/env python3
"""Regression tests for the release transition and publication gates."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSITION = REPO_ROOT / "scripts" / "check-version-transition.py"
TAG_GATE = REPO_ROOT / "scripts" / "check-release-tag.sh"
CI_GATE = REPO_ROOT / "scripts" / "check-ci-status.py"
NOTES = REPO_ROOT / "scripts" / "extract-release-notes.py"
MANIFEST = Path("plugins/profile-and-optimize/.claude-plugin/plugin.json")


def run(*args: str, cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        run("git", "init", "-q", "-b", "main", cwd=self.repo)
        run("git", "config", "user.name", "Release Test", cwd=self.repo)
        run("git", "config", "user.email", "release-test@example.com", cwd=self.repo)
        manifest = self.repo / MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"version":"0.2.0"}\n', encoding="utf-8")
        run("git", "add", ".", cwd=self.repo)
        run("git", "commit", "-q", "-m", "legacy version", cwd=self.repo)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_version(self, version: str, message: str | None = None) -> None:
        (self.repo / "VERSION").write_text(f"{version}\n", encoding="utf-8")
        if message is not None:
            run("git", "add", "VERSION", cwd=self.repo)
            result = run("git", "commit", "-q", "-m", message, cwd=self.repo)
            self.assertEqual(result.returncode, 0, result.stderr)

    def transition(self, *args: str) -> subprocess.CompletedProcess[str]:
        return run("python3", str(TRANSITION), *args, cwd=self.repo)

    def test_dirty_version_uses_head_as_baseline(self) -> None:
        self.write_version("0.3.0")
        result = self.transition("--status-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "increase")

    def test_committed_increase_same_and_regression(self) -> None:
        self.write_version("0.3.0", "add VERSION")
        increased = self.transition("--ref", "HEAD", "--status-only")
        self.assertEqual(increased.returncode, 0, increased.stderr)
        self.assertEqual(increased.stdout.strip(), "increase")

        (self.repo / "note.txt").write_text("same version\n", encoding="utf-8")
        run("git", "add", "note.txt", cwd=self.repo)
        run("git", "commit", "-q", "-m", "same version", cwd=self.repo)
        same = self.transition("--ref", "HEAD", "--status-only")
        self.assertEqual(same.returncode, 0, same.stderr)
        self.assertEqual(same.stdout.strip(), "same")
        required = self.transition("--ref", "HEAD", "--require-increase")
        self.assertEqual(required.returncode, 1)

        self.write_version("0.2.1", "regress VERSION")
        regressed = self.transition("--ref", "HEAD")
        self.assertEqual(regressed.returncode, 1)
        self.assertIn("lower than", regressed.stderr)

    def test_leading_zero_version_is_invalid(self) -> None:
        self.write_version("00.3.0")
        result = self.transition()
        self.assertEqual(result.returncode, 2)
        self.assertIn("numeric X.Y.Z SemVer", result.stderr)

    def test_shallow_ref_check_fails_instead_of_bootstrapping(self) -> None:
        self.write_version("0.3.0", "add VERSION")
        with tempfile.TemporaryDirectory() as clone_parent:
            shallow = Path(clone_parent) / "shallow"
            cloned = run(
                "git",
                "clone",
                "-q",
                "--depth",
                "1",
                self.repo.as_uri(),
                str(shallow),
                cwd=self.repo,
            )
            self.assertEqual(cloned.returncode, 0, cloned.stderr)
            result = run(
                "python3",
                str(TRANSITION),
                "--ref",
                "HEAD",
                cwd=shallow,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("shallow repository", result.stderr)

    def test_exact_annotated_tag_gate(self) -> None:
        self.write_version("0.3.0", "add VERSION")
        run("git", "tag", "-a", "v0.3.0", "-m", "v0.3.0", cwd=self.repo)
        result = run("bash", str(TAG_GATE), "--require", cwd=self.repo)
        self.assertEqual(result.returncode, 0, result.stderr)

        run("git", "tag", "-d", "v0.3.0", cwd=self.repo)
        run("git", "tag", "v0.3.0", cwd=self.repo)
        lightweight = run("bash", str(TAG_GATE), "--require", cwd=self.repo)
        self.assertEqual(lightweight.returncode, 1)
        self.assertIn("not an annotated tag", lightweight.stderr)

    def test_ci_gate_binds_workflow_event_branch_and_sha(self) -> None:
        sha = "a" * 40
        good = {
            "workflowName": "ci",
            "event": "push",
            "headBranch": "main",
            "headSha": sha,
            "status": "completed",
            "conclusion": "success",
        }
        accepted = run(
            "python3",
            str(CI_GATE),
            sha,
            cwd=self.repo,
            stdin=json.dumps([good]),
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

        for field, wrong in (
            ("workflowName", "other"),
            ("event", "pull_request"),
            ("headBranch", "feature"),
            ("headSha", "b" * 40),
            ("conclusion", "failure"),
        ):
            payload = dict(good)
            payload[field] = wrong
            rejected = run(
                "python3",
                str(CI_GATE),
                sha,
                cwd=self.repo,
                stdin=json.dumps([payload]),
            )
            self.assertEqual(rejected.returncode, 1, field)

        pending = dict(good, status="in_progress", conclusion=None)
        rejected = run(
            "python3",
            str(CI_GATE),
            sha,
            cwd=self.repo,
            stdin=json.dumps([pending]),
        )
        self.assertEqual(rejected.returncode, 1)

        malformed = run(
            "python3",
            str(CI_GATE),
            sha,
            cwd=self.repo,
            stdin="not json",
        )
        self.assertEqual(malformed.returncode, 2)

    def test_release_notes_match_literal_version_heading(self) -> None:
        changelog = self.repo / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            "## [0x3x0] - 2026-08-15\n\nWrong section.\n\n"
            "## [0.3.0] - 2026-08-16\n\nRight section.\n\n"
            "## [0.2.0] - 2026-08-03\n\nOlder section.\n",
            encoding="utf-8",
        )
        result = run(
            "python3",
            str(NOTES),
            "v0.3.0",
            str(changelog),
            cwd=self.repo,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Right section.", result.stdout)
        self.assertNotIn("Wrong section.", result.stdout)
        self.assertNotIn("Older section.", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
