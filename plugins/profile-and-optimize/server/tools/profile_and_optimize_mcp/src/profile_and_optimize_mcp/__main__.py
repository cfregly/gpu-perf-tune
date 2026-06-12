"""Console entry point for profile-and-optimize-mcp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from profile_and_optimize_mcp.server import run_stdio
else:
    from .server import run_stdio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLPerf and performance MCP server")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the stdio MCP server")
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        run_stdio()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
