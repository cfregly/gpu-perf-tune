"""Workload-agnostic perf-baseline registry CLI.

This package is a thin shim. The real implementation lives at
[`tools/perf_baseline/perf_baseline_cli.py`](../tools/perf_baseline/perf_baseline_cli.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect this library the same way it introspects the MLPerf libraries
via `<repo_root>/<library>/cli.py`.

Added in profile-and-optimize v0.4.0.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
