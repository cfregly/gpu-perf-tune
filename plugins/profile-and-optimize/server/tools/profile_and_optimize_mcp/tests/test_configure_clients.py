from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "configure_clients.py"
SPEC = importlib.util.spec_from_file_location("configure_clients_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
configure_clients = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configure_clients)


def make_args(root: Path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        python=root / "venv with spaces" / "bin" / "python",
        repo_root=root / "server root",
        dry_run=dry_run,
        registration="file",
        cursor_config=root / "cursor.json",
        claude_config=root / "claude.json",
        codex_config=root / "config.toml",
        gemini_config=root / "gemini.json",
        antigravity_config=root / "antigravity.json",
    )


class JsonConfigTests(unittest.TestCase):
    def test_antigravity_default_uses_official_global_path(self) -> None:
        home = Path("/tmp/test-antigravity-home")
        with mock.patch.object(configure_clients.Path, "home", return_value=home):
            args = configure_clients.parse_args([])
        self.assertEqual(
            args.antigravity_config,
            home / ".gemini" / "config" / "mcp_config.json",
        )

    def test_update_preserves_config_symlink_and_target_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "managed" / "cursor-mcp.json"
            target.parent.mkdir()
            target.write_text('{"theme": "dark"}\n', encoding="utf-8")
            target.chmod(0o600)
            config = root / ".cursor-mcp.json"
            relative_target = target.relative_to(root)
            config.symlink_to(relative_target)

            configure_clients.update_json_mcp(config, make_args(root))

            self.assertTrue(config.is_symlink())
            self.assertEqual(Path(os.readlink(config)), relative_target)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(data["theme"], "dark")
            self.assertIn("profile_and_optimize", data["mcpServers"])

    def test_repeat_update_is_idempotent_and_preserves_unrelated_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "client.json"
            config.write_text(
                json.dumps(
                    {
                        "theme": "dark",
                        "mcpServers": {
                            "other": {"command": "/other/server"},
                            "profile_and_optimize": {"command": "/old/python"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = make_args(root)

            configure_clients.update_json_mcp(config, args)
            first = config.read_bytes()
            configure_clients.update_json_mcp(config, args)
            second = config.read_bytes()

            self.assertEqual(first, second)
            data = json.loads(second)
            self.assertEqual(data["theme"], "dark")
            self.assertEqual(data["mcpServers"]["other"]["command"], "/other/server")
            self.assertEqual(
                data["mcpServers"]["profile_and_optimize"]["command"],
                str(args.python),
            )

    def test_dry_run_never_prints_or_changes_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "client.json"
            secret = "existing-private-token"
            original = json.dumps({"private": secret, "mcpServers": {"other": {}}})
            config.write_text(original, encoding="utf-8")
            args = make_args(root, dry_run=True)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                configure_clients.update_json_mcp(config, args)

            self.assertEqual(config.read_text(encoding="utf-8"), original)
            self.assertNotIn(secret, output.getvalue())
            self.assertIn(str(args.python), output.getvalue())
            self.assertNotIn('"other"', output.getvalue())

    def test_invalid_json_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "client.json"
            original = "{not-json\n"
            config.write_text(original, encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                configure_clients.update_json_mcp(config, make_args(root))

            self.assertEqual(config.read_text(encoding="utf-8"), original)

    def test_main_makes_the_client_executable_path_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "client.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--client",
                    "cursor",
                    "--registration",
                    "file",
                    "--repo-root",
                    "server root",
                    "--python",
                    "venv/bin/python",
                    "--cursor-config",
                    str(config),
                    "--dry-run",
                ],
                cwd=root,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(config.exists())
            self.assertIn(str(root / "venv" / "bin" / "python"), result.stdout)
            self.assertIn(str(root / "server root"), result.stdout)


class CodexConfigTests(unittest.TestCase):
    def test_repeat_update_repairs_duplicate_target_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "config.toml"
            config.write_text(
                """model = "gpt-test"

[mcp_servers.other]
command = "/other/server"

[mcp_servers.profile_and_optimize]
command = "/old/one"

[mcp_servers.profile_and_optimize.env]
PROFILE_AND_OPTIMIZE_REPO_ROOT = "/old/root"

[mcp_servers.profile_and_optimize]
command = "/old/two"

[mcp_servers.profile_and_optimize.env]
PROFILE_AND_OPTIMIZE_REPO_ROOT = "/old/root/two"

[[profiles]]
name = "keep-me"
""",
                encoding="utf-8",
            )
            args = make_args(root)

            configure_clients.update_codex_toml(config, args)
            first = config.read_bytes()
            configure_clients.update_codex_toml(config, args)
            second = config.read_bytes()

            self.assertEqual(first, second)
            text = second.decode()
            self.assertEqual(text.count("[mcp_servers.profile_and_optimize]"), 1)
            self.assertEqual(text.count("[mcp_servers.profile_and_optimize.env]"), 1)
            data = tomllib.loads(text)
            self.assertEqual(data["model"], "gpt-test")
            self.assertEqual(data["profiles"], [{"name": "keep-me"}])
            self.assertEqual(data["mcp_servers"]["other"]["command"], "/other/server")
            self.assertEqual(
                data["mcp_servers"]["profile_and_optimize"]["command"],
                str(args.python),
            )

    def test_dry_run_is_private_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "config.toml"
            secret = "existing-private-token"
            original = f'private_value = "{secret}"\n'
            config.write_text(original, encoding="utf-8")
            args = make_args(root, dry_run=True)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                configure_clients.update_codex_toml(config, args)

            self.assertEqual(config.read_text(encoding="utf-8"), original)
            self.assertNotIn(secret, output.getvalue())
            self.assertIn("[mcp_servers.profile_and_optimize]", output.getvalue())

    def test_toml_strings_escape_quotes_and_backslashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "config.toml"
            args = make_args(root)
            args.python = Path('/tmp/quoted"path\\python')
            args.repo_root = Path('/tmp/quoted"root\\server')

            configure_clients.update_codex_toml(config, args)

            data = tomllib.loads(config.read_text(encoding="utf-8"))
            profile = data["mcp_servers"]["profile_and_optimize"]
            self.assertEqual(profile["command"], str(args.python))
            self.assertEqual(
                profile["env"]["PROFILE_AND_OPTIMIZE_REPO_ROOT"],
                str(args.repo_root),
            )

    def test_generated_config_contains_only_runtime_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "config.toml"
            configure_clients.update_codex_toml(config, make_args(root))

            profile = tomllib.loads(config.read_text(encoding="utf-8"))[
                "mcp_servers"
            ]["profile_and_optimize"]
            self.assertEqual(
                set(profile["env"]),
                {"PROFILE_AND_OPTIMIZE_REPO_ROOT"},
            )


class OfficialCliTests(unittest.TestCase):
    def test_claude_cli_registers_a_new_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = make_args(Path(temporary_directory))
            calls: list[list[str]] = []

            def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return_code = 1 if command[2] == "get" else 0
                return subprocess.CompletedProcess(command, return_code, "", "")

            registered = configure_clients.try_official_cli(
                "claude",
                args,
                which=lambda _: "/usr/local/bin/claude",
                run=run,
            )

            self.assertTrue(registered)
            self.assertEqual(len(calls), 2)
            add_command = calls[1]
            self.assertIn("--scope", add_command)
            self.assertIn("user", add_command)
            self.assertIn(str(args.python), add_command)
            self.assertLess(
                add_command.index("profile_and_optimize"),
                add_command.index("--env"),
            )

    def test_existing_registration_uses_file_fallback_without_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = make_args(Path(temporary_directory))
            calls: list[list[str]] = []

            def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "configured", "")

            registered = configure_clients.try_official_cli(
                "codex",
                args,
                which=lambda _: "/usr/local/bin/codex",
                run=run,
            )

            self.assertFalse(registered)
            self.assertEqual(calls, [["/usr/local/bin/codex", "mcp", "get", "profile_and_optimize"]])

    def test_failed_cli_add_uses_file_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = make_args(Path(temporary_directory))

            def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                return_code = 1 if command[2] == "get" else 7
                return subprocess.CompletedProcess(command, return_code, "", "failed")

            registered = configure_clients.try_official_cli(
                "codex",
                args,
                which=lambda _: "/usr/local/bin/codex",
                run=run,
            )

            self.assertFalse(registered)

    def test_configure_client_writes_fallback_when_cli_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = make_args(root)
            args.registration = "auto"

            with mock.patch.object(configure_clients, "try_official_cli", return_value=False):
                configure_clients.configure_client("codex", args)

            data = tomllib.loads(args.codex_config.read_text(encoding="utf-8"))
            self.assertEqual(
                data["mcp_servers"]["profile_and_optimize"]["command"],
                str(args.python),
            )

    def test_claude_file_fallback_marks_the_server_as_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = make_args(root)
            args.registration = "auto"

            with mock.patch.object(configure_clients, "try_official_cli", return_value=False):
                configure_clients.configure_client("claude", args)

            data = json.loads(args.claude_config.read_text(encoding="utf-8"))
            self.assertEqual(
                data["mcpServers"]["profile_and_optimize"]["type"],
                "stdio",
            )

    def test_cli_dry_run_executes_no_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = make_args(Path(temporary_directory), dry_run=True)
            output = io.StringIO()

            def fail_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
                raise AssertionError("dry run must not execute a client command")

            with contextlib.redirect_stdout(output):
                registered = configure_clients.try_official_cli(
                    "codex",
                    args,
                    which=lambda _: "/usr/local/bin/codex",
                    run=fail_run,
                )

            self.assertTrue(registered)
            self.assertIn("codex mcp add", output.getvalue())


if __name__ == "__main__":
    unittest.main()
