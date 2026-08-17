from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
SERVER_ROOT = REPO_ROOT / "plugins" / "profile-and-optimize" / "server"
SERVER_INSTALLER = SERVER_ROOT / "install.sh"
MCP_INSTALLER = SERVER_ROOT / "tools" / "profile_and_optimize_mcp" / "scripts" / "install_profile_and_optimize_mcp.sh"
AGENT_SKILL_INSTALLER = REPO_ROOT / "scripts" / "install-agent-skills.sh"
CURSOR_SKILL_INSTALLER = REPO_ROOT / "scripts" / "install-skills-into-cursor.sh"
SOURCE_SKILLS = REPO_ROOT / "plugins" / "profile-and-optimize" / "skills"


def make_fake_python(root: Path) -> Path:
    fake_python = root / "fake-python"
    fake_renderer = root / "fake-perftunereport"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "${1-}" == "-m" && "${2-}" == "venv" ]]; then
  mkdir -p "${4}/bin"
  cp "$0" "${4}/bin/python"
  cp "${FAKE_RENDERER_SOURCE}" "${4}/bin/perftunereport"
  exit 0
fi
if [[ "${1-}" == "-m" && "${2-}" == "pip" ]]; then
  exit 0
fi
if [[ "${1-}" == *mcp_surface.py && "${2-}" == "counts" ]]; then
  exit "${FAKE_COUNT_EXIT:-0}"
fi
if [[ "${1-}" == "-c" ]]; then
  exit "$((1 - ${FAKE_MATPLOTLIB:-0}))"
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_renderer.write_text(
        """#!/usr/bin/env bash
exit "${FAKE_RENDER_EXIT:-0}"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_renderer.chmod(0o755)
    return fake_python


class InstallerTests(unittest.TestCase):
    def test_server_installer_help_is_client_neutral(self) -> None:
        result = subprocess.run(
            ["bash", str(SERVER_INSTALLER), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("supported MCP client", result.stdout)
        self.assertNotIn("plugin's .mcp.json", result.stdout)

    def test_server_installer_rejects_missing_option_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            venv = root / "venv"

            result = subprocess.run(
                ["bash", str(SERVER_INSTALLER), "--venv", str(venv), "--python"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(venv.exists())
            self.assertIn("--python requires a value", result.stderr)

    def test_server_installer_count_smoke_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_python = make_fake_python(root)
            venv = root / "venv"

            result = subprocess.run(
                [
                    "bash",
                    str(SERVER_INSTALLER),
                    "--venv",
                    str(venv),
                    "--python",
                    str(fake_python),
                ],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "FAKE_RENDERER_SOURCE": str(root / "fake-perftunereport"),
                    "FAKE_COUNT_EXIT": "7",
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("canonical-count verification failed", result.stderr)
            self.assertNotIn("[done]", result.stdout)

    def test_server_installer_renderer_smoke_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_python = make_fake_python(root)
            venv = root / "venv"

            result = subprocess.run(
                [
                    "bash",
                    str(SERVER_INSTALLER),
                    "--venv",
                    str(venv),
                    "--python",
                    str(fake_python),
                ],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "FAKE_RENDERER_SOURCE": str(root / "fake-perftunereport"),
                    "FAKE_MATPLOTLIB": "1",
                    "FAKE_RENDER_EXIT": "9",
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("perftunereport report_smoke failed", result.stderr)
            self.assertNotIn("[done]", result.stdout)

    def test_server_installer_minimal_output_is_concise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_python = make_fake_python(root)

            result = subprocess.run(
                [
                    "bash",
                    str(SERVER_INSTALLER),
                    "--venv",
                    str(root / "venv"),
                    "--python",
                    str(fake_python),
                ],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "FAKE_RENDERER_SOURCE": str(root / "fake-perftunereport"),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[note] Optional report rendering is not installed", result.stdout)
            self.assertNotIn("ACTION NEEDED", result.stdout)
            self.assertNotIn("plugin .mcp.json", result.stdout)

    def test_mcp_dry_run_changes_nothing_and_keeps_existing_config_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            venv = root / "venv"
            config = root / "codex-config.toml"
            marker = "unchanged-config-marker"
            original = f'custom_value = "{marker}"\n'
            config.write_text(original, encoding="utf-8")

            result = subprocess.run(
                [
                    "bash",
                    str(MCP_INSTALLER),
                    "--venv",
                    str(venv),
                    "--client",
                    "codex",
                    "--registration",
                    "file",
                    "--codex-config",
                    str(config),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(venv.exists())
            self.assertEqual(config.read_text(encoding="utf-8"), original)
            self.assertNotIn(marker, result.stdout)
            self.assertIn("would create venv", result.stdout)
            self.assertIn("[mcp_servers.profile_and_optimize]", result.stdout)
            self.assertIn("No files or client settings were changed", result.stdout)

    def test_invalid_client_fails_before_installing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            venv = root / "venv"

            result = subprocess.run(
                [
                    "bash",
                    str(MCP_INSTALLER),
                    "--venv",
                    str(venv),
                    "--client",
                    "not-a-client",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(venv.exists())
            self.assertIn("invalid client", result.stderr)

    def test_mcp_installer_requires_an_explicit_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            venv = root / "venv"

            result = subprocess.run(
                [
                    "bash",
                    str(MCP_INSTALLER),
                    "--venv",
                    str(venv),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(venv.exists())
            self.assertIn("missing required --client", result.stderr)

    def test_agent_skill_installer_is_idempotent_for_codex_and_cursor(self) -> None:
        expected_names = {
            path.name for path in SOURCE_SKILLS.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
        }
        for client in ("codex", "cursor"):
            with self.subTest(client=client), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                skills_directory = root / "skills"
                command = [
                    "bash",
                    str(AGENT_SKILL_INSTALLER),
                    "--client",
                    client,
                    "--skills-dir",
                    str(skills_directory),
                ]

                first = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env={**os.environ, "HOME": str(root / "home")},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                second = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env={**os.environ, "HOME": str(root / "home")},
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertEqual(second.returncode, 0, second.stderr)
                links = sorted(skills_directory.iterdir())
                self.assertEqual({link.name for link in links}, expected_names)
                self.assertTrue(all(link.is_symlink() for link in links))
                self.assertIn(f"{len(expected_names)} linked", first.stdout)
                self.assertIn(f"{len(expected_names)} already-linked", second.stdout)

    def test_agent_skill_installer_preserves_conflicts_for_codex_and_cursor(self) -> None:
        skill_name = min(
            path.name for path in SOURCE_SKILLS.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
        )
        for client in ("codex", "cursor"):
            with self.subTest(client=client), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                skills_directory = root / "skills"
                conflict = skills_directory / skill_name
                conflict.mkdir(parents=True)
                marker = conflict / "keep.txt"
                marker.write_text("operator-owned\n", encoding="utf-8")

                result = subprocess.run(
                    [
                        "bash",
                        str(AGENT_SKILL_INSTALLER),
                        "--client",
                        client,
                        "--skills-dir",
                        str(skills_directory),
                    ],
                    cwd=REPO_ROOT,
                    env={**os.environ, "HOME": str(root / "home")},
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(result.returncode, 1)
                self.assertEqual(marker.read_text(encoding="utf-8"), "operator-owned\n")
                self.assertFalse(conflict.is_symlink())
                self.assertIn("refusing to replace", result.stderr)

    def test_cursor_compatibility_wrapper_uses_cursor_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            home = Path(temporary_directory) / "home"
            result = subprocess.run(
                ["bash", str(CURSOR_SKILL_INSTALLER)],
                cwd=REPO_ROOT,
                env={**os.environ, "HOME": str(home)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((home / ".cursor" / "skills").is_dir())


if __name__ == "__main__":
    unittest.main()
