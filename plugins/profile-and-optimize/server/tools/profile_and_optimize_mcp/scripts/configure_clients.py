#!/usr/bin/env python3
"""Configure local MCP clients for the checked-in profile_and_optimize server."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

CLIENTS = ("cursor", "claude", "codex", "gemini", "antigravity")
SERVER_NAME = "profile_and_optimize"
CLI_CLIENTS = ("claude", "codex")
_TOML_TABLE_RE = re.compile(r"^\s*\[{1,2}([^\[\]]+)\]{1,2}\s*(?:#.*)?$")


def server_block(
    args: argparse.Namespace,
    *,
    client: str | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "command": str(args.python),
        "args": ["-m", "profile_and_optimize_mcp", "serve"],
        "env": {
            "PROFILE_AND_OPTIMIZE_REPO_ROOT": str(args.repo_root),
        },
    }
    if client == "claude":
        block = {"type": "stdio", **block}
    return block


def _atomic_write(path: Path, text: str) -> None:
    """Replace one config atomically while preserving its mode and symlink."""

    write_path = path.resolve(strict=False) if path.is_symlink() else path
    write_path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(write_path.stat().st_mode) if write_path.exists() else None
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=write_path.parent,
        prefix=f".{write_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, write_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _preview(path: Path, text: str) -> None:
    """Print only the proposed server entry, never the existing config."""

    print(f"# DRY-RUN would update {path}")
    print(text.rstrip())


def _write(path: Path, text: str, *, dry_run: bool, preview: str) -> None:
    if dry_run:
        _preview(path, preview)
        return
    _atomic_write(path, text)
    print(f"updated {path}")


def update_json_mcp(
    path: Path,
    args: argparse.Namespace,
    *,
    client: str | None = None,
) -> None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
    else:
        data = {}

    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"{path}: mcpServers must be a JSON object")
    servers[SERVER_NAME] = server_block(args, client=client)

    full_text = json.dumps(data, indent=2, sort_keys=False) + "\n"
    preview_text = json.dumps(
        {"mcpServers": {SERVER_NAME: server_block(args, client=client)}},
        indent=2,
        sort_keys=False,
    )
    _write(path, full_text, dry_run=args.dry_run, preview=preview_text)


def _toml_string(value: object) -> str:
    """JSON strings are valid TOML basic strings and escape paths safely."""

    return json.dumps(str(value), ensure_ascii=False)


def _codex_block(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            f"[mcp_servers.{SERVER_NAME}]",
            f"command = {_toml_string(args.python)}",
            'args = ["-m", "profile_and_optimize_mcp", "serve"]',
            "enabled = true",
            "startup_timeout_sec = 30",
            "tool_timeout_sec = 300",
            "",
            f"[mcp_servers.{SERVER_NAME}.env]",
            f"PROFILE_AND_OPTIMIZE_REPO_ROOT = {_toml_string(args.repo_root)}",
            "",
        ]
    )


def _without_profile_tables(text: str) -> str:
    """Remove all existing profile_and_optimize TOML tables.

    This deliberately accepts duplicate target tables left by older installer
    versions. Unrelated content is kept byte-for-byte except for trailing blank
    lines before the newly generated block.
    """

    kept: list[str] = []
    skip = False
    target_prefix = f"mcp_servers.{SERVER_NAME}"
    for line in text.splitlines(keepends=True):
        match = _TOML_TABLE_RE.match(line.rstrip("\r\n"))
        if match:
            table = match.group(1).strip()
            skip = table == target_prefix or table.startswith(f"{target_prefix}.")
        if not skip:
            kept.append(line)
    return "".join(kept).rstrip()


def update_codex_toml(path: Path, args: argparse.Namespace) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = _codex_block(args)
    base = _without_profile_tables(text)
    new_text = f"{base}\n\n{block}" if base else block

    try:
        tomllib.loads(new_text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"refusing to write invalid TOML to {path}: {error}") from error

    _write(path, new_text, dry_run=args.dry_run, preview=block)


def _official_add_command(client: str, executable: str, args: argparse.Namespace) -> list[str]:
    environment = server_block(args)["env"]
    launch = [str(args.python), "-m", "profile_and_optimize_mcp", "serve"]
    if client == "claude":
        command = [
            executable,
            "mcp",
            "add",
            "--scope",
            "user",
            "--transport",
            "stdio",
            SERVER_NAME,
        ]
        for key, value in environment.items():
            command.extend(["--env", f"{key}={value}"])
        return [*command, "--", *launch]
    if client == "codex":
        command = [executable, "mcp", "add"]
        for key, value in environment.items():
            command.extend(["--env", f"{key}={value}"])
        return [*command, SERVER_NAME, "--", *launch]
    raise ValueError(f"no official MCP registration command for {client}")


def try_official_cli(
    client: str,
    args: argparse.Namespace,
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    """Register a new server with a client CLI when it is safe to do so.

    Existing registrations use the atomic config updater instead. This avoids a
    remove-then-add window and makes repeat installs deterministic.
    """

    executable = which(client)
    if executable is None:
        return False

    add_command = _official_add_command(client, executable, args)
    if args.dry_run:
        print(f"# DRY-RUN would register {SERVER_NAME} with the {client} CLI")
        print(shlex.join(add_command))
        return True

    get_result = run(
        [executable, "mcp", "get", SERVER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    if get_result.returncode == 0:
        return False

    add_result = run(
        add_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if add_result.returncode == 0:
        print(f"registered {SERVER_NAME} with the {client} CLI")
        return True

    print(
        f"warning: {client} CLI registration failed with exit {add_result.returncode}. "
        "Using the atomic config-file updater."
    )
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client",
        action="append",
        choices=(*CLIENTS, "all"),
        required=True,
        help="Client to configure. May be passed multiple times.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python",
        type=Path,
        default=Path.home() / ".local/share/profile-and-optimize-mcp-venv/bin/python",
    )
    parser.add_argument(
        "--registration",
        choices=("auto", "file"),
        default="auto",
        help=(
            "Use a supported client CLI for a new Claude or Codex registration, "
            "with atomic file fallback. Use 'file' to skip client CLIs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print only the proposed server entry or CLI command. Write nothing.",
    )
    parser.add_argument("--cursor-config", type=Path, default=Path.home() / ".cursor/mcp.json")
    parser.add_argument("--claude-config", type=Path, default=Path.home() / ".claude.json")
    parser.add_argument("--codex-config", type=Path, default=Path.home() / ".codex/config.toml")
    parser.add_argument("--gemini-config", type=Path, default=Path.home() / ".gemini/settings.json")
    parser.add_argument(
        "--antigravity-config",
        type=Path,
        default=Path.home() / ".gemini/config/mcp_config.json",
        help=(
            "Path to Antigravity MCP config. Defaults to the official global "
            "path. Pass a workspace .agents/mcp_config.json when desired."
        ),
    )
    return parser.parse_args(argv)


def _file_config_path(client: str, args: argparse.Namespace) -> Path:
    return {
        "cursor": args.cursor_config,
        "claude": args.claude_config,
        "codex": args.codex_config,
        "gemini": args.gemini_config,
        "antigravity": args.antigravity_config,
    }[client]


def configure_client(client: str, args: argparse.Namespace) -> None:
    if args.registration == "auto" and client in CLI_CLIENTS and try_official_cli(client, args):
        return

    config_path = _file_config_path(client, args)
    if client == "codex":
        update_codex_toml(config_path, args)
    else:
        update_json_mcp(config_path, args, client=client)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    clients = args.client
    if "all" in clients:
        clients = list(CLIENTS)
    else:
        clients = list(dict.fromkeys(clients))

    args.repo_root = args.repo_root.expanduser().resolve()
    # Do not call resolve() here. Venv Python is often a symlink to the base
    # interpreter, and clients must launch the venv path to load its packages.
    args.python = args.python.expanduser()
    if not args.python.is_absolute():
        args.python = (Path.cwd() / args.python).absolute()

    for client in clients:
        configure_client(client, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
