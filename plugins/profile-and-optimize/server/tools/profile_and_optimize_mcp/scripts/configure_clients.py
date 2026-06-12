#!/usr/bin/env python3
"""Configure local MCP clients for the checked-in profile_and_optimize MCP server."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


CLIENTS = ("cursor", "claude", "codex", "gemini", "antigravity")


def server_block(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "command": str(args.python),
        "args": ["-m", "profile_and_optimize_mcp", "serve"],
        "env": {
            "PROFILE_AND_OPTIMIZE_REPO_ROOT": str(args.repo_root),
            "PROFILE_AND_OPTIMIZE_LOGIN_HOST": args.login_host,
        },
    }


def _write(path: Path, text: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"# DRY-RUN would write {path}")
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"updated {path}")


def update_json_mcp(path: Path, args: argparse.Namespace) -> None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    servers = data.setdefault("mcpServers", {})
    servers["profile_and_optimize"] = server_block(args)
    _write(path, json.dumps(data, indent=2, sort_keys=False) + "\n", dry_run=args.dry_run)


def update_codex_toml(path: Path, args: argparse.Namespace) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(
        r"(?ms)^\\[mcp_servers\\.mlperf\\]\\n.*?(?=^\\[[^\\n]+\\]\\n|\\Z)"
    )
    text = pattern.sub("", text).rstrip()
    block = f"""

[mcp_servers.profile_and_optimize]
command = "{args.python}"
args = ["-m", "profile_and_optimize_mcp", "serve"]
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 300

[mcp_servers.profile_and_optimize.env]
PROFILE_AND_OPTIMIZE_REPO_ROOT = "{args.repo_root}"
PROFILE_AND_OPTIMIZE_LOGIN_HOST = "{args.login_host}"
""".lstrip()
    new_text = (text + "\n\n" + block if text else block).rstrip() + "\n"
    _write(path, new_text, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client",
        action="append",
        choices=(*CLIENTS, "all"),
        default=[],
        help="client to configure; may be passed multiple times. Default: cursor.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python",
        type=Path,
        default=Path.home() / ".local/share/profile-and-optimize-mcp-venv/bin/python",
    )
    parser.add_argument(
        "--login-host",
        default=os.environ.get(
            "PROFILE_AND_OPTIMIZE_LOGIN_HOST", f"{os.environ.get('USER', 'operator')}@192.0.2.10"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cursor-config", type=Path, default=Path.home() / ".cursor/mcp.json")
    parser.add_argument(
        "--claude-config", type=Path, default=Path.home() / ".claude/settings.json"
    )
    parser.add_argument(
        "--codex-config", type=Path, default=Path.home() / ".codex/config.toml"
    )
    parser.add_argument(
        "--gemini-config", type=Path, default=Path.home() / ".gemini/settings.json"
    )
    parser.add_argument(
        "--antigravity-config",
        type=Path,
        default=Path.home() / ".config/antigravity/mcp_config.json",
        help=(
            "Path to Antigravity raw MCP config. In Antigravity, use Agent "
            "window -> Manage MCP Servers -> View raw config to confirm."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clients = args.client or ["cursor"]
    if "all" in clients:
        clients = list(CLIENTS)
    args.repo_root = args.repo_root.resolve()
    # Preserve venv interpreter paths. Path.resolve() follows the venv
    # symlink to Homebrew's base interpreter, which makes Cursor bypass the
    # venv site-packages and fail to import profile_and_optimize_mcp.
    args.python = args.python.expanduser()
    if not args.python.is_absolute():
        args.python = (Path.cwd() / args.python).absolute()

    for client in clients:
        if client == "cursor":
            update_json_mcp(args.cursor_config, args)
        elif client == "claude":
            update_json_mcp(args.claude_config, args)
        elif client == "codex":
            update_codex_toml(args.codex_config, args)
        elif client == "gemini":
            update_json_mcp(args.gemini_config, args)
        elif client == "antigravity":
            update_json_mcp(args.antigravity_config, args)
        else:
            raise AssertionError(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
