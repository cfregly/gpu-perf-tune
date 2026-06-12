"""Perf-report CLI library: build multi-page benchmark report PDFs.

This package is a thin shim. The real implementation lives at
[`tools/perf_tune_report/perf_tune_report_cli.py`](../tools/perf_tune_report/perf_tune_report_cli.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect this library the same way it introspects every other
contract-driven library via `<repo_root>/<library>/cli.py`.

Backs the [`inference-perf-tune-report`](../skills/inference-perf-tune-report/SKILL.md)
skill and ships the `perftunereport` console-script CLI.

Added in profile-and-optimize v1.10.0.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
